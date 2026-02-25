#!/usr/bin/env python3
"""
Phase 1: Infrastructure Setup and Verification

- Loads Qwen3-1.7B and verifies config
- Tests hidden state extraction
- Prepares and saves the dataset
- Logs all config values
"""

import sys
import os
import json
import random
import logging
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seeds(config["seeds"]["torch"])

    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "phase1")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Step 1: Log environment ----
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            env_info["gpus"].append({
                "name": torch.cuda.get_device_name(i),
                "memory_total_mb": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024),
            })

    logger.info(f"Environment: {json.dumps(env_info, indent=2)}")

    # ---- Step 2: Load model and tokenizer ----
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

    # ---- Step 3: Record model config ----
    model_config = {
        "model_name": model_name,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
        "head_dim": getattr(model.config, "head_dim", None),
        "intermediate_size": model.config.intermediate_size,
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "hidden_act": getattr(model.config, "hidden_act", None),
        "rms_norm_eps": getattr(model.config, "rms_norm_eps", None),
        "torch_dtype": str(model.config.torch_dtype),
    }
    logger.info(f"Model config:\n{json.dumps(model_config, indent=2)}")

    # ---- Step 4: Verify hidden state extraction ----
    logger.info("Testing hidden state extraction...")
    test_input = tokenizer("Hello world, this is a test.", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**test_input, output_hidden_states=True)

    num_layers = len(out.hidden_states)
    hidden_shape = out.hidden_states[-1].shape
    logger.info(f"Num hidden state layers (including embedding): {num_layers}")
    logger.info(f"Hidden state shape: {hidden_shape}")
    logger.info(f"D_model (hidden_size): {model.config.hidden_size}")

    # Verify extraction layers are valid
    extraction_layers = config["latent_extractor"]["extraction_layers"]
    max_layer_idx = num_layers - 1
    for l in extraction_layers:
        assert l <= max_layer_idx, f"Layer {l} > max {max_layer_idx}"
    logger.info(f"Extraction layers {extraction_layers} verified (max={max_layer_idx})")

    # Verify embedding access
    embed_layer = model.model.embed_tokens
    test_embeds = embed_layer(test_input.input_ids)
    logger.info(f"Embedding layer accessible, output shape: {test_embeds.shape}")

    hidden_state_check = {
        "num_hidden_state_layers": num_layers,
        "hidden_state_shape": list(hidden_shape),
        "extraction_layers_valid": True,
        "embedding_access_valid": True,
    }

    # ---- Step 5: Test generation ----
    logger.info("Testing generation...")
    gen_input = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_out = model.generate(**gen_input, max_new_tokens=20, do_sample=False)
    generated_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    logger.info(f"Generation test: '{generated_text}'")

    # ---- Step 6: Prepare dataset ----
    logger.info("Building dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    builder = DatasetBuilder(
        tokenizer=tokenizer,
        source=config["dataset"]["source"],
        min_doc_tokens=config["dataset"]["min_doc_tokens"],
        max_doc_tokens=config["dataset"]["max_doc_tokens"],
        seed=config["seeds"]["random"],
    )

    splits = builder.build(
        train_samples=config["dataset"]["train_samples"],
        val_samples=config["dataset"]["val_samples"],
        test_samples=config["dataset"]["test_samples"],
        test_max_doc_tokens=config["dataset"]["test_max_doc_tokens"],
    )

    builder.save(splits, data_dir)

    dataset_stats = {
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
    }
    for split_name, samples in splits.items():
        if samples:
            token_counts = [s["num_tokens"] for s in samples]
            dataset_stats[f"{split_name}_min_tokens"] = min(token_counts)
            dataset_stats[f"{split_name}_max_tokens"] = max(token_counts)
            dataset_stats[f"{split_name}_mean_tokens"] = sum(token_counts) / len(token_counts)

            # Task type distribution
            task_dist = {}
            for s in samples:
                t = s["task_type"]
                task_dist[t] = task_dist.get(t, 0) + 1
            dataset_stats[f"{split_name}_task_distribution"] = task_dist

    logger.info(f"Dataset stats:\n{json.dumps(dataset_stats, indent=2)}")

    # ---- Save all Phase 1 outputs ----
    phase1_output = {
        "environment": env_info,
        "model_config": model_config,
        "hidden_state_check": hidden_state_check,
        "generation_test": generated_text,
        "dataset_stats": dataset_stats,
        "experiment_config": config,
        "status": "PASS",
    }

    output_path = os.path.join(output_dir, "phase1_report.json")
    with open(output_path, "w") as f:
        json.dump(phase1_output, f, indent=2)

    logger.info(f"Phase 1 complete. Report saved to {output_path}")
    logger.info("=" * 60)
    logger.info("PHASE 1 CHECKPOINT: ALL COMPONENTS VERIFIED")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  D_model: {model.config.hidden_size}")
    logger.info(f"  Num layers: {model.config.num_hidden_layers}")
    logger.info(f"  Dataset: {dataset_stats['train_count']}/{dataset_stats['val_count']}/{dataset_stats['test_count']}")
    logger.info("=" * 60)

    return phase1_output


if __name__ == "__main__":
    main()
