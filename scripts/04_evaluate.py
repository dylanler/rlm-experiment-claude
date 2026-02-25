#!/usr/bin/env python3
"""
Phase 4: Evaluation and Comparison

Runs the trained Latent Pager system on the test set.
Computes all metrics from Section 6.2.
Compares against baseline results from Phase 2.
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

from src.model.latent_extractor import extract_latent_states
from src.model.page_compressor import PageCompressor
from src.model.page_aggregator import PageAggregator
from src.model.page_store import LatentPageStore
from src.model.soft_prompt import inject_soft_prompt_and_generate
from src.data.chunker import DocumentChunker
from src.data.dataset_builder import DatasetBuilder
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.consistency import global_consistency
from src.evaluation.significance import paired_bootstrap_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_latent_pager_inference(
    model, tokenizer, compressor, aggregator, sample, config
):
    """Run latent pager inference on a single sample."""
    device = next(model.parameters()).device
    chunker = DocumentChunker(
        tokenizer,
        chunk_size=config.get("chunker", {}).get("chunk_size", 1024),
        overlap=config.get("chunker", {}).get("overlap", 128),
    )
    extraction_layers = config.get("latent_extractor", {}).get(
        "extraction_layers", [7, 14, 21, 27]
    )
    pooling = config.get("latent_extractor", {}).get("pooling", "mean")

    chunks = chunker.chunk(sample["document"])
    page_store = LatentPageStore()

    for chunk in chunks:
        input_ids = torch.tensor([chunk["token_ids"]], device=device)
        attention_mask = torch.ones_like(input_ids)

        latent_states = extract_latent_states(
            model, input_ids, attention_mask, extraction_layers, pooling
        )
        page_vector = compressor(latent_states)
        page_store.write(chunk["chunk_id"], page_vector)

    all_pages = page_store.read_all().to(device)

    # Get question embeddings for conditioned aggregation (if enabled)
    q_embed = None
    if config.get("training", {}).get("use_question_conditioning", True):
        question_text = f"Question: {sample['question']}\nAnswer:"
        q_ids = tokenizer(question_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            q_embed = model.model.embed_tokens(q_ids).squeeze(0).float()  # [q_len, D_model]

    soft_prompt = aggregator(all_pages, q_embed)

    answer = inject_soft_prompt_and_generate(
        model,
        tokenizer,
        soft_prompt,
        f"Question: {sample['question']}\nAnswer:",
        max_new_tokens=config.get("evaluation", {}).get("max_new_tokens", 256),
    )

    return answer, len(chunks)


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

    # Load trained compressor + aggregator
    compressor = PageCompressor(
        num_layers=num_extraction_layers, d_model=d_model, d_page=d_page
    )
    aggregator = PageAggregator(
        d_page=d_page,
        d_model=d_model,
        num_soft_tokens=config["page_aggregator"]["num_soft_tokens"],
        num_heads=config["page_aggregator"]["num_heads"],
        num_agg_layers=config["page_aggregator"]["num_agg_layers"],
    )

    # Allow overriding checkpoint via command line
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "best_model.pt"
        )
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Run 03_train_latent_pager.py first")
        sys.exit(1)

    device = next(model.parameters()).device
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    compressor.load_state_dict(ckpt["compressor_state_dict"])
    aggregator.load_state_dict(ckpt["aggregator_state_dict"])
    compressor = compressor.to(device).eval()
    aggregator = aggregator.to(device).eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Load dataset
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    splits = DatasetBuilder.load(data_dir)
    test_data = splits["test"]
    logger.info(f"Loaded {len(test_data)} test samples")

    # Run evaluation
    predictions = []
    all_metrics = []
    total_time = 0
    peak_memory = 0

    for i, sample in enumerate(tqdm(test_data, desc="Latent Pager Eval")):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        try:
            with torch.no_grad():
                answer, num_chunks = run_latent_pager_inference(
                    model, tokenizer, compressor, aggregator, sample, config
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"OOM on sample {sample['id']}, skipping")
                torch.cuda.empty_cache()
                continue
            raise

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
            "num_chunks": num_chunks,
            "latency_seconds": elapsed,
            "metrics": metrics,
            "task_type": sample.get("task_type", "unknown"),
        })
        all_metrics.append(metrics)

        if (i + 1) % 10 == 0:
            avg_f1 = np.mean([m["f1"] for m in all_metrics])
            logger.info(f"  [{i+1}/{len(test_data)}] Running F1: {avg_f1:.4f}")

        torch.cuda.empty_cache()

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

    # Save latent pager results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "latent_pager")
    os.makedirs(results_dir, exist_ok=True)

    lp_results = {
        "num_samples": len(predictions),
        "aggregate_metrics": agg_metrics,
        "per_task_metrics": per_task,
        "total_time_seconds": total_time,
        "avg_latency_seconds": total_time / max(len(predictions), 1),
        "peak_memory_gb": peak_memory,
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(lp_results, f, indent=2)

    with open(os.path.join(results_dir, "predictions.jsonl"), "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # ---- Comparison with baseline ----
    baseline_metrics_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "baseline", "metrics.json"
    )
    if os.path.exists(baseline_metrics_path):
        with open(baseline_metrics_path) as f:
            baseline_results = json.load(f)

        baseline = baseline_results.get("1024", {})
        comparison_dir = os.path.join(
            os.path.dirname(__file__), "..", "results", "comparison"
        )
        os.makedirs(comparison_dir, exist_ok=True)

        # Load baseline predictions for significance testing
        baseline_preds_path = os.path.join(
            os.path.dirname(__file__), "..", "results", "baseline", "predictions_chunk1024.jsonl"
        )
        baseline_preds = {}
        if os.path.exists(baseline_preds_path):
            with open(baseline_preds_path) as f:
                for line in f:
                    p = json.loads(line)
                    baseline_preds[p["id"]] = p

        # Paired significance tests
        sig_results = {}
        for metric_key in ["f1", "rouge_l", "hallucination_rate"]:
            scores_baseline = []
            scores_latent = []
            for pred in predictions:
                if pred["id"] in baseline_preds:
                    scores_baseline.append(baseline_preds[pred["id"]]["metrics"][metric_key])
                    scores_latent.append(pred["metrics"][metric_key])

            if scores_baseline:
                sig = paired_bootstrap_test(scores_baseline, scores_latent)
                sig_results[metric_key] = sig
                logger.info(
                    f"Significance test ({metric_key}): "
                    f"diff={sig['diff']:.4f}, p={sig['p_value']:.4f}, "
                    f"significant={sig['significant']}"
                )

        with open(os.path.join(comparison_dir, "significance_tests.json"), "w") as f:
            json.dump(sig_results, f, indent=2)

        # Consistency test
        doc_answers = {}
        for pred in predictions:
            doc_id = pred["id"].rsplit("_", 1)[0] if "_" in pred["id"] else pred["id"]
            if doc_id not in doc_answers:
                doc_answers[doc_id] = {"answers": [], "document": ""}
            doc_answers[doc_id]["answers"].append(pred["prediction"])

        if doc_answers:
            consistency_scores = []
            for doc_id, data in doc_answers.items():
                if len(data["answers"]) >= 2:
                    score = global_consistency(data["answers"], data.get("document", ""))
                    consistency_scores.append(score)

            if consistency_scores:
                lp_results["global_consistency"] = {
                    "mean": float(np.mean(consistency_scores)),
                    "std": float(np.std(consistency_scores)),
                }

        # Summary table
        bl_agg = baseline.get("aggregate_metrics", {})
        lp_agg = agg_metrics

        summary = "# Comparison: Latent Pager vs Text Buffer Baseline\n\n"
        summary += "| Metric | Text Buffer (Baseline) | Latent Pager | Difference | Significant |\n"
        summary += "|---|---|---|---|---|\n"

        for metric_key in ["f1", "rouge_l", "exact_match", "hallucination_rate"]:
            bl_val = bl_agg.get(metric_key, {}).get("mean", 0)
            lp_val = lp_agg.get(metric_key, {}).get("mean", 0)
            diff = lp_val - bl_val
            sig = sig_results.get(metric_key, {}).get("significant", "N/A")
            summary += f"| {metric_key} | {bl_val:.4f} | {lp_val:.4f} | {diff:+.4f} | {sig} |\n"

        summary += f"\n| Avg Latency (s) | {baseline.get('avg_latency_seconds', 0):.2f} | {lp_results['avg_latency_seconds']:.2f} | | |\n"
        summary += f"| Peak Memory (GB) | {baseline.get('peak_memory_gb', 0):.2f} | {lp_results['peak_memory_gb']:.2f} | | |\n"

        # Per-task breakdown
        summary += "\n## Per-Task Type Breakdown\n\n"
        all_task_types = set(list(per_task.keys()) + list(baseline.get("per_task_metrics", {}).keys()))
        for tt in sorted(all_task_types):
            summary += f"\n### {tt}\n\n"
            summary += "| Metric | Baseline | Latent Pager |\n|---|---|---|\n"
            bl_tt = baseline.get("per_task_metrics", {}).get(tt, {})
            lp_tt = per_task.get(tt, {})
            for mk in ["f1", "rouge_l", "hallucination_rate"]:
                bl_v = bl_tt.get(mk, {}).get("mean", 0)
                lp_v = lp_tt.get(mk, {}).get("mean", 0)
                summary += f"| {mk} | {bl_v:.4f} | {lp_v:.4f} |\n"

        with open(os.path.join(comparison_dir, "summary_table.md"), "w") as f:
            f.write(summary)

        logger.info(f"Comparison summary saved to {comparison_dir}/summary_table.md")
    else:
        logger.warning("No baseline results found. Run 02_run_baseline.py first.")

    logger.info("=" * 60)
    logger.info("PHASE 4 CHECKPOINT: EVALUATION COMPLETE")
    logger.info(f"  Latent Pager F1: {agg_metrics['f1']['mean']:.4f}")
    logger.info(f"  Latent Pager ROUGE-L: {agg_metrics['rouge_l']['mean']:.4f}")
    logger.info(f"  Latent Pager Hallucination: {agg_metrics['hallucination_rate']['mean']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
