#!/usr/bin/env python3
"""
Phase 3a: Pre-train PageCompressor with Reconstruction Objective

Trains the compressor to preserve information by reconstructing original
hidden states from compressed page vectors. No QA labels needed — uses
all document chunks as self-supervised training data.
"""

import sys
import os
import json
import random
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.latent_extractor import extract_latent_states
from src.model.page_compressor import PageCompressor
from src.model.reconstruction_head import ReconstructionHead
from src.data.chunker import DocumentChunker
from src.data.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seeds(config["seeds"]["torch"])

    # Load model
    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
        device_map=config["model"]["device_map"],
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = next(model.parameters()).device
    d_model = model.config.hidden_size

    extraction_layers = config["latent_extractor"]["extraction_layers"]
    pooling = config["latent_extractor"]["pooling"]
    d_page = config["page_compressor"]["d_page"]
    num_ext_layers = len(extraction_layers)

    # Create compressor and reconstruction head
    compressor = PageCompressor(num_layers=num_ext_layers, d_model=d_model, d_page=d_page).to(device)
    recon_head = ReconstructionHead(d_page=d_page, num_layers=num_ext_layers, d_model=d_model).to(device)

    total_params = sum(p.numel() for p in compressor.parameters()) + sum(p.numel() for p in recon_head.parameters())
    logger.info(f"Pre-training params: {total_params:,} (compressor + recon head)")

    # Load ALL data (no QA labels needed, just documents)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    splits = DatasetBuilder.load(data_dir)
    all_documents = []
    for split_name in ["train", "val", "test"]:
        for sample in splits[split_name]:
            all_documents.append(sample["document"])
    # Deduplicate
    all_documents = list(set(all_documents))
    logger.info(f"Loaded {len(all_documents)} unique documents for pre-training")

    # Extract all chunks
    chunker = DocumentChunker(
        tokenizer,
        chunk_size=config.get("chunker", {}).get("chunk_size", 1024),
        overlap=config.get("chunker", {}).get("overlap", 128),
        max_chunks=config.get("chunker", {}).get("max_chunks", 64),
    )

    logger.info("Extracting hidden states for all chunks...")
    all_states = []  # list of [num_layers, D_model] tensors
    for doc in tqdm(all_documents, desc="Extracting chunks"):
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            input_ids = torch.tensor([chunk["token_ids"]], device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                latent_states = extract_latent_states(
                    model, input_ids, attention_mask, extraction_layers, pooling
                )  # [num_layers, D_model]
            all_states.append(latent_states.cpu())
            torch.cuda.empty_cache()

    logger.info(f"Extracted {len(all_states)} chunks for pre-training")

    # Pre-training loop
    epochs = 50
    lr = 5e-4
    trainable_params = list(compressor.parameters()) + list(recon_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Cosine schedule
    total_steps = len(all_states) * epochs
    from src.training.scheduler import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    logger.info(f"Starting pre-training: {epochs} epochs, {len(all_states)} chunks/epoch")

    best_loss = float("inf")
    for epoch in range(epochs):
        compressor.train()
        recon_head.train()

        # Shuffle chunk order each epoch
        indices = list(range(len(all_states)))
        random.shuffle(indices)

        epoch_loss = 0.0
        for idx in indices:
            optimizer.zero_grad()

            states = all_states[idx].to(device)  # [num_layers, D_model]
            page_vector = compressor(states)  # [d_page]
            reconstructed = recon_head(page_vector)  # [num_layers, D_model]

            loss = nn.functional.mse_loss(reconstructed, states)
            loss.backward()

            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(all_states)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Recon Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save pretrained compressor and recon head
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "pretrained_compressor.pt")
    torch.save({
        "compressor_state_dict": compressor.state_dict(),
        "recon_head_state_dict": recon_head.state_dict(),
        "final_recon_loss": best_loss,
        "config": config,
    }, save_path)

    logger.info(f"Pre-training complete. Best recon loss: {best_loss:.6f}")
    logger.info(f"Saved pretrained compressor to {save_path}")


if __name__ == "__main__":
    main()
