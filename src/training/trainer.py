"""
Training loop for PageCompressor + PageAggregator.
The base Qwen3-1.7B model remains frozen throughout.
"""

import time
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model.latent_extractor import extract_latent_states
from src.model.page_compressor import PageCompressor
from src.model.page_aggregator import PageAggregator
from src.model.page_store import LatentPageStore
from src.model.soft_prompt import compute_soft_prompt_loss
from src.data.chunker import DocumentChunker
from src.evaluation.metrics import compute_f1
from src.model.soft_prompt import inject_soft_prompt_and_generate
from .scheduler import get_cosine_schedule_with_warmup, EarlyStopping

logger = logging.getLogger(__name__)


class LatentPagerTrainer:
    """
    Trains PageCompressor + PageAggregator end-to-end.
    The frozen base model is used for hidden state extraction and loss computation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        compressor: PageCompressor,
        aggregator: PageAggregator,
        config: dict,
        output_dir: str = "checkpoints",
        log_dir: str = "logs",
        recon_head=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.compressor = compressor
        self.aggregator = aggregator
        self.recon_head = recon_head
        self.config = config
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = next(model.parameters()).device

        # Move trainable modules to device
        self.compressor = self.compressor.to(self.device)
        self.aggregator = self.aggregator.to(self.device)
        if self.recon_head is not None:
            self.recon_head = self.recon_head.to(self.device)

        # Chunker
        self.chunker = DocumentChunker(
            tokenizer,
            chunk_size=config.get("chunker", {}).get("chunk_size", 1024),
            overlap=config.get("chunker", {}).get("overlap", 128),
            max_chunks=config.get("chunker", {}).get("max_chunks", 64),
        )

        # Extraction config
        self.extraction_layers = config.get("latent_extractor", {}).get(
            "extraction_layers", [7, 14, 21, 27]
        )
        self.pooling = config.get("latent_extractor", {}).get("pooling", "mean")

        # Training config
        train_cfg = config.get("training", {})
        self.lr = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.epochs = train_cfg.get("epochs", 20)
        self.warmup_steps = train_cfg.get("warmup_steps", 500)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.patience = train_cfg.get("patience", 5)
        self.min_delta = train_cfg.get("min_delta", 0.001)
        self.fast_val = train_cfg.get("fast_val", False)
        self.lambda_recon = train_cfg.get("lambda_recon", 0.0)
        self.use_question_conditioning = train_cfg.get("use_question_conditioning", True)

    def _get_question_embed(self, question: str) -> torch.Tensor:
        """Get question token embeddings from the frozen model."""
        question_text = f"Question: {question}\nAnswer:"
        q_ids = self.tokenizer(question_text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            q_embed = self.model.model.embed_tokens(q_ids).squeeze(0)  # [q_len, D_model]
        return q_embed.float()

    def _extract_pages(self, document: str) -> tuple[torch.Tensor, list[dict], list[torch.Tensor]]:
        """Extract and compress all chunks of a document into latent pages.

        NOTE: We do NOT use LatentPageStore here because it calls .detach().cpu()
        which would break the gradient chain. Instead we collect page vectors
        in a list and stack them, preserving gradients for backprop.

        Returns:
            all_pages: [num_pages, d_page] with gradients preserved
            chunks: list of chunk dicts
            original_states: list of [num_layers, D_model] tensors (detached, for recon loss)
        """
        chunks = self.chunker.chunk(document)
        page_vectors = []
        original_states = []

        for chunk in chunks:
            input_ids = torch.tensor(
                [chunk["token_ids"]], device=self.device
            )
            attention_mask = torch.ones_like(input_ids)

            # Extract hidden states from frozen model
            with torch.no_grad():
                latent_states = extract_latent_states(
                    self.model,
                    input_ids,
                    attention_mask,
                    self.extraction_layers,
                    self.pooling,
                )  # [num_layers, D_model]

            # Save original states for reconstruction loss
            original_states.append(latent_states.detach())

            # Compress (trainable — grad flows through here)
            page_vector = self.compressor(latent_states)  # [d_page]
            page_vectors.append(page_vector)

        all_pages = torch.stack(page_vectors)  # [num_pages, d_page]
        return all_pages, chunks, original_states

    def _compute_recon_loss(self, all_pages: torch.Tensor, original_states: list[torch.Tensor]) -> torch.Tensor:
        """Compute reconstruction loss: decode page vectors back to hidden states."""
        if self.recon_head is None:
            return torch.tensor(0.0, device=self.device)

        recon_loss = 0.0
        for page_vec, orig_state in zip(all_pages, original_states):
            reconstructed = self.recon_head(page_vec)  # [num_layers, D_model]
            recon_loss += nn.functional.mse_loss(reconstructed, orig_state)
        return recon_loss / len(original_states)

    def train(
        self,
        train_data: list[dict],
        val_data: list[dict],
    ) -> dict:
        """
        Main training loop.

        Args:
            train_data: list of {"document", "question", "gold_answer", ...}
            val_data: list of {"document", "question", "gold_answer", ...}

        Returns: dict with training history
        """
        # Freeze base model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Optimizer for trainable params only
        trainable_params = list(self.compressor.parameters()) + list(
            self.aggregator.parameters()
        )
        if self.recon_head is not None:
            trainable_params += list(self.recon_head.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = len(train_data) * self.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.warmup_steps, total_steps
        )
        early_stopping = EarlyStopping(
            patience=self.patience, min_delta=self.min_delta, mode="min"
        )

        writer = SummaryWriter(str(self.log_dir))
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        best_val_f1 = -1.0
        global_step = 0
        nan_count = 0

        logger.info(f"Starting training: {self.epochs} epochs, {len(train_data)} samples/epoch")
        logger.info(f"  lambda_recon={self.lambda_recon}, recon_head={'yes' if self.recon_head else 'no'}")

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.compressor.train()
            self.aggregator.train()
            if self.recon_head is not None:
                self.recon_head.train()

            epoch_loss = 0.0
            num_samples = 0

            for sample in tqdm(train_data, desc=f"Epoch {epoch+1}/{self.epochs}"):
                optimizer.zero_grad()

                try:
                    # Extract and compress pages
                    all_pages, chunks, original_states = self._extract_pages(sample["document"])

                    # Get question embedding for conditioned aggregation
                    q_embed = None
                    if self.use_question_conditioning:
                        q_embed = self._get_question_embed(sample["question"])

                    # Aggregate into soft prompt
                    soft_prompt = self.aggregator(all_pages, q_embed)  # [num_soft_tokens, D_model]

                    # Compute QA loss against gold answer
                    qa_loss = compute_soft_prompt_loss(
                        self.model,
                        self.tokenizer,
                        soft_prompt,
                        f"Question: {sample['question']}\nAnswer:",
                        sample["gold_answer"],
                    )

                    # Compute reconstruction loss
                    if self.lambda_recon > 0 and self.recon_head is not None:
                        recon_loss = self._compute_recon_loss(all_pages, original_states)
                        loss = (1 - self.lambda_recon) * qa_loss + self.lambda_recon * recon_loss
                    else:
                        loss = qa_loss

                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_count += 1
                        logger.warning(f"NaN/Inf loss at step {global_step}")
                        if nan_count >= 3:
                            logger.error("3+ consecutive NaN losses, stopping")
                            return history
                        continue
                    else:
                        nan_count = 0

                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        trainable_params, self.gradient_clip
                    )
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()
                    num_samples += 1
                    global_step += 1

                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                    # Memory management
                    del all_pages, soft_prompt, loss, original_states
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM on sample, skipping. Error: {e}")
                        torch.cuda.empty_cache()
                        continue
                    raise

            avg_train_loss = epoch_loss / max(num_samples, 1)
            history["train_loss"].append(avg_train_loss)
            history["lr"].append(scheduler.get_last_lr()[0])

            # Validation
            val_loss, val_f1 = self._validate(val_data)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/f1", val_f1, epoch)

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save checkpoint (by val_f1 which is the actual evaluation metric)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self._save_checkpoint("best_model.pt", epoch, val_loss, val_f1)

            self._save_checkpoint(f"epoch_{epoch+1}.pt", epoch, val_loss, val_f1)

            # Early stopping
            if early_stopping.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        writer.close()
        return history

    @torch.no_grad()
    def _validate(self, val_data: list[dict], max_samples: int = 50) -> tuple[float, float]:
        """Run validation and return (loss, f1)."""
        self.compressor.eval()
        self.aggregator.eval()

        total_loss = 0.0
        total_f1 = 0.0
        num_samples = 0

        for sample in val_data[:max_samples]:
            try:
                all_pages, chunks, _ = self._extract_pages(sample["document"])
                q_embed = None
                if self.use_question_conditioning:
                    q_embed = self._get_question_embed(sample["question"])
                soft_prompt = self.aggregator(all_pages, q_embed)

                # Loss (without grad)
                loss = compute_soft_prompt_loss(
                    self.model,
                    self.tokenizer,
                    soft_prompt,
                    f"Question: {sample['question']}\nAnswer:",
                    sample["gold_answer"],
                )
                total_loss += loss.item()

                # Generate answer for F1 (skip if fast_val mode)
                if not self.fast_val:
                    answer = inject_soft_prompt_and_generate(
                        self.model,
                        self.tokenizer,
                        soft_prompt,
                        f"Question: {sample['question']}\nAnswer:",
                        max_new_tokens=128,
                    )
                    f1 = compute_f1(answer, sample["gold_answer"])
                    total_f1 += f1

                num_samples += 1

                del all_pages, soft_prompt
                torch.cuda.empty_cache()

            except RuntimeError:
                torch.cuda.empty_cache()
                continue

        avg_loss = total_loss / max(num_samples, 1)
        avg_f1 = total_f1 / max(num_samples, 1)
        return avg_loss, avg_f1

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float, val_f1: float):
        """Save compressor + aggregator checkpoint."""
        path = self.output_dir / filename
        save_dict = {
            "epoch": epoch,
            "compressor_state_dict": self.compressor.state_dict(),
            "aggregator_state_dict": self.aggregator.state_dict(),
            "val_loss": val_loss,
            "val_f1": val_f1,
            "config": self.config,
        }
        if self.recon_head is not None:
            save_dict["recon_head_state_dict"] = self.recon_head.state_dict()
        torch.save(save_dict, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load compressor + aggregator from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.compressor.load_state_dict(ckpt["compressor_state_dict"])
        self.aggregator.load_state_dict(ckpt["aggregator_state_dict"])
        if self.recon_head is not None and "recon_head_state_dict" in ckpt:
            self.recon_head.load_state_dict(ckpt["recon_head_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt
