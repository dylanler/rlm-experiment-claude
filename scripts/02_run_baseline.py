#!/usr/bin/env python3
"""
Phase 2: Baseline Evaluation

Runs the TextBufferBaseline on the test set with multiple chunk sizes.
Records accuracy, ROUGE-L, hallucination rate, latency, and memory.
"""

import sys
import os
import json
import time
import random
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.baseline.text_buffer import TextBufferBaseline
from src.data.chunker import DocumentChunker
from src.data.dataset_builder import DatasetBuilder
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_baseline_eval(
    model, tokenizer, test_data, chunk_size, max_buffer_tokens=4096
):
    """Run baseline on test data with given chunk_size."""
    baseline = TextBufferBaseline(
        model, tokenizer, chunk_size=chunk_size, max_buffer_tokens=max_buffer_tokens
    )
    chunker = DocumentChunker(tokenizer, chunk_size=chunk_size, overlap=128)

    predictions = []
    all_metrics = []
    total_time = 0
    peak_memory = 0

    for i, sample in enumerate(tqdm(test_data, desc=f"Baseline (chunk={chunk_size})")):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        chunks = chunker.chunk(sample["document"])
        answer = baseline.run(
            document=sample["document"],
            question=sample["question"],
            chunks=chunks,
        )

        elapsed = time.time() - start_time
        total_time += elapsed

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_memory = max(peak_memory, peak_mem)

        metrics = compute_all_metrics(
            prediction=answer,
            gold_answer=sample["gold_answer"],
            source_document=sample["document"],
        )

        predictions.append({
            "id": sample["id"],
            "question": sample["question"],
            "gold_answer": sample["gold_answer"],
            "prediction": answer,
            "num_chunks": len(chunks),
            "latency_seconds": elapsed,
            "metrics": metrics,
            "task_type": sample.get("task_type", "unknown"),
        })
        all_metrics.append(metrics)

        if (i + 1) % 10 == 0:
            avg_f1 = np.mean([m["f1"] for m in all_metrics])
            logger.info(f"  [{i+1}/{len(test_data)}] Running F1: {avg_f1:.4f}")

    # Aggregate metrics
    agg_metrics = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        agg_metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }

    # Per task-type metrics
    task_metrics = {}
    for pred in predictions:
        tt = pred["task_type"]
        if tt not in task_metrics:
            task_metrics[tt] = []
        task_metrics[tt].append(pred["metrics"])

    per_task = {}
    for tt, metrics_list in task_metrics.items():
        per_task[tt] = {}
        for key in metrics_list[0]:
            values = [m[key] for m in metrics_list]
            per_task[tt][key] = {"mean": float(np.mean(values)), "count": len(values)}

    return {
        "chunk_size": chunk_size,
        "num_samples": len(test_data),
        "aggregate_metrics": agg_metrics,
        "per_task_metrics": per_task,
        "total_time_seconds": total_time,
        "avg_latency_seconds": total_time / len(test_data),
        "peak_memory_gb": peak_memory,
    }, predictions


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

    # Load dataset
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    splits = DatasetBuilder.load(data_dir)
    test_data = splits["test"]
    logger.info(f"Loaded {len(test_data)} test samples")

    # Phase 2 blocker check
    if len(test_data) == 0:
        logger.error("PHASE 2 BLOCKER: No test data available")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "baseline")
    os.makedirs(output_dir, exist_ok=True)

    # Run primary chunk_size on full test set, others on subset
    primary_cs = 1024
    other_chunk_sizes = [512, 2048]
    subset_size = 50  # smaller subset for non-primary chunk sizes
    all_results = {}

    # Primary evaluation (full test set)
    logger.info(f"Running baseline with primary chunk_size={primary_cs} on full test set ({len(test_data)} samples)")
    results, predictions = run_baseline_eval(
        model, tokenizer, test_data, chunk_size=primary_cs
    )
    all_results[str(primary_cs)] = results

    pred_path = os.path.join(output_dir, f"predictions_chunk{primary_cs}.jsonl")
    with open(pred_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info(
        f"  chunk_size={primary_cs}: F1={results['aggregate_metrics']['f1']['mean']:.4f}, "
        f"ROUGE-L={results['aggregate_metrics']['rouge_l']['mean']:.4f}, "
        f"Hallucination={results['aggregate_metrics']['hallucination_rate']['mean']:.4f}"
    )

    # Secondary evaluations (subset only)
    for cs in other_chunk_sizes:
        logger.info(f"Running baseline with chunk_size={cs} on subset ({subset_size} samples)")
        results_sub, predictions_sub = run_baseline_eval(
            model, tokenizer, test_data[:subset_size], chunk_size=cs
        )
        all_results[str(cs)] = results_sub

        pred_path = os.path.join(output_dir, f"predictions_chunk{cs}.jsonl")
        with open(pred_path, "w") as f:
            for pred in predictions_sub:
                f.write(json.dumps(pred) + "\n")

        logger.info(
            f"  chunk_size={cs}: F1={results_sub['aggregate_metrics']['f1']['mean']:.4f}, "
            f"ROUGE-L={results_sub['aggregate_metrics']['rouge_l']['mean']:.4f}, "
            f"Hallucination={results_sub['aggregate_metrics']['hallucination_rate']['mean']:.4f}"
        )

    # Use chunk_size=1024 as the primary baseline
    primary = all_results["1024"]

    # Phase 2 blocker: check if accuracy is too low
    primary_f1 = primary["aggregate_metrics"]["f1"]["mean"]
    if primary_f1 < 0.05:
        logger.warning(
            f"PHASE 2 WARNING: Baseline F1={primary_f1:.4f} < 0.05. "
            f"Model may be too weak. Consider simplifying dataset."
        )

    # Save results
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)

    config_out_path = os.path.join(output_dir, "config.json")
    with open(config_out_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "chunk_sizes": [primary_cs] + other_chunk_sizes,
            "max_buffer_tokens": config["baseline"]["max_buffer_tokens"],
            "primary_chunk_size": 1024,
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("PHASE 2 CHECKPOINT: BASELINE ESTABLISHED")
    logger.info(f"  Primary (chunk=1024) F1: {primary_f1:.4f}")
    logger.info(f"  Primary ROUGE-L: {primary['aggregate_metrics']['rouge_l']['mean']:.4f}")
    logger.info(f"  Primary Hallucination: {primary['aggregate_metrics']['hallucination_rate']['mean']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
