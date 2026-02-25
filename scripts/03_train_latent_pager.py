#!/usr/bin/env python3
"""
Phase 3: Latent Pager Training

Trains the PageCompressor + PageAggregator modules while keeping
the base Qwen3-1.7B frozen. Implements all training hyperparameters
from Section 7.3 of the spec.
"""

import sys
import os
import json
import random
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.page_compressor import PageCompressor
from src.model.page_aggregator import PageAggregator
from src.model.reconstruction_head import ReconstructionHead
from src.data.dataset_builder import DatasetBuilder
from src.training.trainer import LatentPagerTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_curves(history: dict, output_path: str):
    """Plot and save training loss and validation F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_f1"], "g-", label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {output_path}")


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

    d_model = model.config.hidden_size
    num_extraction_layers = len(config["latent_extractor"]["extraction_layers"])
    d_page = config["page_compressor"]["d_page"]

    logger.info(f"D_model={d_model}, num_extraction_layers={num_extraction_layers}, d_page={d_page}")

    # Create trainable modules
    compressor = PageCompressor(
        num_layers=num_extraction_layers,
        d_model=d_model,
        d_page=d_page,
    )
    aggregator = PageAggregator(
        d_page=d_page,
        d_model=d_model,
        num_soft_tokens=config["page_aggregator"]["num_soft_tokens"],
        num_heads=config["page_aggregator"]["num_heads"],
        num_agg_layers=config["page_aggregator"]["num_agg_layers"],
    )

    # Create reconstruction head
    recon_head = ReconstructionHead(
        d_page=d_page,
        num_layers=num_extraction_layers,
        d_model=d_model,
    )

    # Load pretrained compressor if available
    pretrained_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pretrained_compressor.pt")
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained compressor from {pretrained_path}")
        pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        compressor.load_state_dict(pretrained["compressor_state_dict"])
        recon_head.load_state_dict(pretrained["recon_head_state_dict"])
        logger.info(f"  Pretrained recon loss: {pretrained.get('final_recon_loss', 'N/A')}")
    else:
        logger.info("No pretrained compressor found, training from scratch")

    total_params = sum(p.numel() for p in compressor.parameters()) + sum(
        p.numel() for p in aggregator.parameters()
    ) + sum(p.numel() for p in recon_head.parameters())
    logger.info(f"Total trainable parameters: {total_params:,}")

    # Load dataset
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    splits = DatasetBuilder.load(data_dir)
    train_data = splits["train"]
    val_data = splits["val"]
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Create trainer
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")

    trainer = LatentPagerTrainer(
        model=model,
        tokenizer=tokenizer,
        compressor=compressor,
        aggregator=aggregator,
        config=config,
        output_dir=checkpoint_dir,
        log_dir=log_dir,
        recon_head=recon_head,
    )

    # Train
    logger.info("Starting training...")
    history = trainer.train(train_data, val_data)

    # Phase 3 blocker check
    if len(history.get("train_loss", [])) > 2:
        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]
        if final_loss >= initial_loss:
            logger.warning(
                f"PHASE 3 WARNING: Training loss did not decrease "
                f"(initial={initial_loss:.4f}, final={final_loss:.4f}). "
                f"Check architecture or learning rate."
            )

    # Save training curves
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "latent_pager")
    os.makedirs(results_dir, exist_ok=True)

    curves_path = os.path.join(results_dir, "training_curves.png")
    if history.get("train_loss"):
        plot_training_curves(history, curves_path)

    # Save training history
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save config used
    config_out_path = os.path.join(results_dir, "config.json")
    with open(config_out_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "d_model": d_model,
            "d_page": d_page,
            "num_extraction_layers": num_extraction_layers,
            "extraction_layers": config["latent_extractor"]["extraction_layers"],
            "pooling": config["latent_extractor"]["pooling"],
            "num_soft_tokens": config["page_aggregator"]["num_soft_tokens"],
            "num_agg_layers": config["page_aggregator"]["num_agg_layers"],
            "training": config["training"],
            "total_trainable_params": total_params,
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("PHASE 3 CHECKPOINT: TRAINING COMPLETE")
    if history.get("train_loss"):
        logger.info(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"  Final Val F1: {history['val_f1'][-1]:.4f}")
        logger.info(f"  Best Val F1: {max(history['val_f1']):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
