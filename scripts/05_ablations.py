#!/usr/bin/env python3
"""
Phase 5: Ablation Studies

Runs ablation experiments varying one factor at a time:
- d_page: {128, 256, 512, 1024, 2048}
- num_soft_tokens: {8, 16, 32, 64, 128}
- extraction layers: {last_only, quartiles, all_layers}
- pooling: {mean, last_token}
- number of chunks: {4, 8, 16, 32, 64}
- aggregator depth: {1, 2, 4}
"""

import sys
import os
import json
import copy
import random
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.latent_extractor import extract_latent_states
from src.model.page_compressor import PageCompressor
from src.model.page_aggregator import PageAggregator
from src.model.page_store import LatentPageStore
from src.model.soft_prompt import inject_soft_prompt_and_generate
from src.data.chunker import DocumentChunker
from src.data.dataset_builder import DatasetBuilder
from src.evaluation.metrics import compute_all_metrics
from src.training.trainer import LatentPagerTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_short_training(model, tokenizer, compressor, aggregator, config, train_data, val_data, epochs=3):
    """Short training run for ablation. Uses fast_val to skip generation."""
    abl_config = copy.deepcopy(config)
    abl_config["training"]["epochs"] = epochs
    abl_config["training"]["patience"] = epochs  # Don't early stop during ablation
    abl_config["training"]["fast_val"] = True  # Skip generation in validation

    trainer = LatentPagerTrainer(
        model=model,
        tokenizer=tokenizer,
        compressor=compressor,
        aggregator=aggregator,
        config=abl_config,
        output_dir=os.path.join("checkpoints", "ablation_temp"),
        log_dir=os.path.join("logs", "ablation_temp"),
    )

    history = trainer.train(train_data, val_data[:20])
    return history


def evaluate_model(model, tokenizer, compressor, aggregator, test_data, config, max_samples=30):
    """Quick evaluation on a subset."""
    device = next(model.parameters()).device
    compressor = compressor.to(device).eval()
    aggregator = aggregator.to(device).eval()

    chunker = DocumentChunker(
        tokenizer,
        chunk_size=config.get("chunker", {}).get("chunk_size", 1024),
        overlap=config.get("chunker", {}).get("overlap", 128),
    )
    extraction_layers = config.get("latent_extractor", {}).get(
        "extraction_layers", [7, 14, 21, 27]
    )
    pooling = config.get("latent_extractor", {}).get("pooling", "mean")

    all_metrics = []
    for sample in tqdm(test_data[:max_samples], desc="Ablation eval"):
        try:
            chunks = chunker.chunk(sample["document"])
            page_store = LatentPageStore()

            for chunk in chunks:
                input_ids = torch.tensor([chunk["token_ids"]], device=device)
                attention_mask = torch.ones_like(input_ids)
                with torch.no_grad():
                    latent_states = extract_latent_states(
                        model, input_ids, attention_mask, extraction_layers, pooling
                    )
                    page_vector = compressor(latent_states)
                page_store.write(chunk["chunk_id"], page_vector)

            all_pages = page_store.read_all().to(device)
            with torch.no_grad():
                # Get question embeddings for conditioned aggregation
                question_text = f"Question: {sample['question']}\nAnswer:"
                q_ids = tokenizer(question_text, return_tensors="pt").input_ids.to(device)
                q_embed = model.model.embed_tokens(q_ids).squeeze(0).float()
                soft_prompt = aggregator(all_pages, q_embed)
                answer = inject_soft_prompt_and_generate(
                    model, tokenizer, soft_prompt,
                    f"Question: {sample['question']}\nAnswer:",
                    max_new_tokens=128,
                )

            metrics = compute_all_metrics(answer, sample["gold_answer"], sample["document"])
            all_metrics.append(metrics)
            torch.cuda.empty_cache()
        except RuntimeError:
            torch.cuda.empty_cache()
            continue

    if not all_metrics:
        return {"f1": 0, "rouge_l": 0, "hallucination_rate": 1}

    agg = {}
    for key in all_metrics[0]:
        agg[key] = float(np.mean([m[key] for m in all_metrics]))
    return agg


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seeds(config["seeds"]["torch"])

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

    d_model = model.config.hidden_size
    num_hidden_layers = model.config.num_hidden_layers

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    splits = DatasetBuilder.load(data_dir)
    # Use smaller subsets for ablation (optimized for speed)
    train_data = splits["train"][:100]
    val_data = splits["val"][:20]
    test_data = splits["test"][:30]

    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "latent_pager", "ablations")
    os.makedirs(output_dir, exist_ok=True)

    ablation_results = {}

    def _save_partial():
        with open(os.path.join(output_dir, "all_ablations.json"), "w") as f:
            json.dump(ablation_results, f, indent=2, default=str)

    # ---- Ablation 1: d_page ----
    logger.info("=" * 40 + " ABLATION: d_page " + "=" * 40)
    d_page_results = {}
    for d_page in [128, 256, 512, 1024, 2048]:
        logger.info(f"Testing d_page={d_page}")
        set_seeds(42)

        num_ext_layers = len(config["latent_extractor"]["extraction_layers"])
        comp = PageCompressor(num_layers=num_ext_layers, d_model=d_model, d_page=d_page)
        agg = PageAggregator(
            d_page=d_page, d_model=d_model,
            num_soft_tokens=config["page_aggregator"]["num_soft_tokens"],
            num_heads=config["page_aggregator"]["num_heads"],
            num_agg_layers=config["page_aggregator"]["num_agg_layers"],
        )

        abl_config = copy.deepcopy(config)
        abl_config["page_compressor"]["d_page"] = d_page
        history = run_short_training(model, tokenizer, comp, agg, abl_config, train_data, val_data)
        metrics = evaluate_model(model, tokenizer, comp, agg, test_data, abl_config)

        d_page_results[d_page] = {
            "metrics": metrics,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        }
        logger.info(f"  d_page={d_page}: F1={metrics.get('f1', 0):.4f}")

    ablation_results["d_page"] = d_page_results
    _save_partial()

    # ---- Ablation 2: num_soft_tokens ----
    logger.info("=" * 40 + " ABLATION: num_soft_tokens " + "=" * 40)
    soft_token_results = {}
    for nst in [8, 16, 32, 64, 128]:
        logger.info(f"Testing num_soft_tokens={nst}")
        set_seeds(42)

        d_page = config["page_compressor"]["d_page"]
        num_ext_layers = len(config["latent_extractor"]["extraction_layers"])
        comp = PageCompressor(num_layers=num_ext_layers, d_model=d_model, d_page=d_page)
        agg = PageAggregator(
            d_page=d_page, d_model=d_model,
            num_soft_tokens=nst,
            num_heads=config["page_aggregator"]["num_heads"],
            num_agg_layers=config["page_aggregator"]["num_agg_layers"],
        )

        abl_config = copy.deepcopy(config)
        abl_config["page_aggregator"]["num_soft_tokens"] = nst
        history = run_short_training(model, tokenizer, comp, agg, abl_config, train_data, val_data)
        metrics = evaluate_model(model, tokenizer, comp, agg, test_data, abl_config)

        soft_token_results[nst] = {
            "metrics": metrics,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        }
        logger.info(f"  num_soft_tokens={nst}: F1={metrics.get('f1', 0):.4f}")

    ablation_results["num_soft_tokens"] = soft_token_results
    _save_partial()

    # ---- Ablation 3: Extraction layers ----
    logger.info("=" * 40 + " ABLATION: extraction_layers " + "=" * 40)
    layer_configs = {
        "last_only": [num_hidden_layers],
        "quartiles": [
            num_hidden_layers // 4,
            num_hidden_layers // 2,
            3 * num_hidden_layers // 4,
            num_hidden_layers,
        ],
        "all_even": list(range(2, num_hidden_layers + 1, 2)),
    }
    layer_results = {}
    for name, layers in layer_configs.items():
        logger.info(f"Testing extraction_layers={name}: {layers}")
        set_seeds(42)

        d_page = config["page_compressor"]["d_page"]
        comp = PageCompressor(num_layers=len(layers), d_model=d_model, d_page=d_page)
        agg = PageAggregator(
            d_page=d_page, d_model=d_model,
            num_soft_tokens=config["page_aggregator"]["num_soft_tokens"],
            num_heads=config["page_aggregator"]["num_heads"],
            num_agg_layers=config["page_aggregator"]["num_agg_layers"],
        )

        abl_config = copy.deepcopy(config)
        abl_config["latent_extractor"]["extraction_layers"] = layers
        history = run_short_training(model, tokenizer, comp, agg, abl_config, train_data, val_data)
        metrics = evaluate_model(model, tokenizer, comp, agg, test_data, abl_config)

        layer_results[name] = {
            "layers": layers,
            "metrics": metrics,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        }
        logger.info(f"  {name}: F1={metrics.get('f1', 0):.4f}")

    ablation_results["extraction_layers"] = layer_results
    _save_partial()

    # ---- Ablation 4: Pooling ----
    logger.info("=" * 40 + " ABLATION: pooling " + "=" * 40)
    pooling_results = {}
    for pooling in ["mean", "last_token"]:
        logger.info(f"Testing pooling={pooling}")
        set_seeds(42)

        d_page = config["page_compressor"]["d_page"]
        num_ext_layers = len(config["latent_extractor"]["extraction_layers"])
        comp = PageCompressor(num_layers=num_ext_layers, d_model=d_model, d_page=d_page)
        agg = PageAggregator(
            d_page=d_page, d_model=d_model,
            num_soft_tokens=config["page_aggregator"]["num_soft_tokens"],
            num_heads=config["page_aggregator"]["num_heads"],
            num_agg_layers=config["page_aggregator"]["num_agg_layers"],
        )

        abl_config = copy.deepcopy(config)
        abl_config["latent_extractor"]["pooling"] = pooling
        history = run_short_training(model, tokenizer, comp, agg, abl_config, train_data, val_data)
        metrics = evaluate_model(model, tokenizer, comp, agg, test_data, abl_config)

        pooling_results[pooling] = {
            "metrics": metrics,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        }
        logger.info(f"  pooling={pooling}: F1={metrics.get('f1', 0):.4f}")

    ablation_results["pooling"] = pooling_results
    _save_partial()

    # ---- Ablation 5: Aggregator depth ----
    logger.info("=" * 40 + " ABLATION: aggregator_depth " + "=" * 40)
    depth_results = {}
    for depth in [1, 2, 4]:
        logger.info(f"Testing num_agg_layers={depth}")
        set_seeds(42)

        d_page = config["page_compressor"]["d_page"]
        num_ext_layers = len(config["latent_extractor"]["extraction_layers"])
        comp = PageCompressor(num_layers=num_ext_layers, d_model=d_model, d_page=d_page)
        agg = PageAggregator(
            d_page=d_page, d_model=d_model,
            num_soft_tokens=config["page_aggregator"]["num_soft_tokens"],
            num_heads=config["page_aggregator"]["num_heads"],
            num_agg_layers=depth,
        )

        abl_config = copy.deepcopy(config)
        abl_config["page_aggregator"]["num_agg_layers"] = depth
        history = run_short_training(model, tokenizer, comp, agg, abl_config, train_data, val_data)
        metrics = evaluate_model(model, tokenizer, comp, agg, test_data, abl_config)

        depth_results[depth] = {
            "metrics": metrics,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        }
        logger.info(f"  num_agg_layers={depth}: F1={metrics.get('f1', 0):.4f}")

    ablation_results["aggregator_depth"] = depth_results
    _save_partial()

    # Individual files for spec compliance
    with open(os.path.join(output_dir, "d_page_sweep.json"), "w") as f:
        json.dump(d_page_results, f, indent=2, default=str)

    with open(os.path.join(output_dir, "pooling_comparison.json"), "w") as f:
        json.dump(pooling_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("PHASE 5 CHECKPOINT: ABLATIONS COMPLETE")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
