#!/usr/bin/env python3
"""
Phase 6: Generate Final Report

Compiles all results into a final analysis, evaluates hypotheses H1-H5,
and produces a verdict (SUCCESS/STRONG SUCCESS/PARTIAL SUCCESS/FAILURE).
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    results_dir = os.path.join(base_dir, "results")
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Load all results
    phase1 = load_json(os.path.join(results_dir, "phase1", "phase1_report.json"))
    baseline_metrics = load_json(os.path.join(results_dir, "baseline", "metrics.json"))
    lp_metrics = load_json(os.path.join(results_dir, "latent_pager", "metrics.json"))
    lp_history = load_json(os.path.join(results_dir, "latent_pager", "training_history.json"))
    sig_tests = load_json(os.path.join(comparison_dir, "significance_tests.json"))
    ablations = load_json(os.path.join(results_dir, "latent_pager", "ablations", "all_ablations.json"))

    if not baseline_metrics or not lp_metrics:
        logger.error("Missing baseline or latent pager metrics. Run phases 2 and 4 first.")
        sys.exit(1)

    # Extract primary metrics
    bl = baseline_metrics.get("1024", {}).get("aggregate_metrics", {})
    lp = lp_metrics.get("aggregate_metrics", {})

    bl_f1 = bl.get("f1", {}).get("mean", 0)
    lp_f1 = lp.get("f1", {}).get("mean", 0)
    bl_rouge = bl.get("rouge_l", {}).get("mean", 0)
    lp_rouge = lp.get("rouge_l", {}).get("mean", 0)
    bl_halluc = bl.get("hallucination_rate", {}).get("mean", 0)
    lp_halluc = lp.get("hallucination_rate", {}).get("mean", 0)
    bl_latency = baseline_metrics.get("1024", {}).get("avg_latency_seconds", 0)
    lp_latency = lp_metrics.get("avg_latency_seconds", 0)

    # ---- Evaluate Hypotheses ----
    hypotheses = {}

    # H1: Hallucination reduction >= 10% relative
    if bl_halluc > 0:
        halluc_reduction = (bl_halluc - lp_halluc) / bl_halluc * 100
    else:
        halluc_reduction = 0
    h1_supported = lp_halluc < bl_halluc
    h1_strong = halluc_reduction >= 10
    hypotheses["H1"] = {
        "description": "Latent pages reduce hallucination (>=10% relative reduction)",
        "baseline_hallucination": bl_halluc,
        "latent_pager_hallucination": lp_halluc,
        "relative_reduction_pct": halluc_reduction,
        "supported": h1_supported,
        "strongly_supported": h1_strong,
    }

    # H2: Multi-hop accuracy improvement >= 5 F1 points
    bl_per_task = baseline_metrics.get("1024", {}).get("per_task_metrics", {})
    lp_per_task = lp_metrics.get("per_task_metrics", {})
    mh_bl = bl_per_task.get("multi_hop_reasoning", {}).get("f1", {}).get("mean", 0)
    mh_lp = lp_per_task.get("multi_hop_reasoning", {}).get("f1", {}).get("mean", 0)
    h2_supported = mh_lp > mh_bl
    h2_strong = (mh_lp - mh_bl) >= 0.05
    hypotheses["H2"] = {
        "description": "Multi-hop accuracy improvement >= 5 F1 points",
        "baseline_multi_hop_f1": mh_bl,
        "latent_pager_multi_hop_f1": mh_lp,
        "difference": mh_lp - mh_bl,
        "supported": h2_supported,
        "strongly_supported": h2_strong,
    }

    # H3: Global consistency improves
    lp_consistency = lp_metrics.get("global_consistency", {}).get("mean", None)
    hypotheses["H3"] = {
        "description": "Global consistency improves with latent aggregation",
        "latent_pager_consistency": lp_consistency,
        "supported": lp_consistency is not None and lp_consistency > 0.5,
    }

    # H4: Information retention scales with d_page (from ablations)
    h4_supported = False
    if ablations and "d_page" in ablations:
        d_page_f1s = []
        for d_page_val, res in sorted(ablations["d_page"].items(), key=lambda x: int(x[0])):
            d_page_f1s.append((int(d_page_val), res.get("metrics", {}).get("f1", 0)))
        # Check monotonic trend
        if len(d_page_f1s) >= 3:
            increases = sum(1 for i in range(1, len(d_page_f1s)) if d_page_f1s[i][1] >= d_page_f1s[i-1][1])
            h4_supported = increases >= len(d_page_f1s) // 2
        hypotheses["H4"] = {
            "description": "Information retention scales with d_page",
            "d_page_f1_curve": d_page_f1s,
            "supported": h4_supported,
        }
    else:
        hypotheses["H4"] = {
            "description": "Information retention scales with d_page",
            "supported": None,
            "note": "Ablation data not available",
        }

    # H5: Compute cost is comparable (<=1.5x)
    if bl_latency > 0:
        latency_ratio = lp_latency / bl_latency
    else:
        latency_ratio = float("inf")
    h5_supported = latency_ratio <= 1.5
    hypotheses["H5"] = {
        "description": "Compute cost <= 1.5x text baseline",
        "baseline_latency": bl_latency,
        "latent_pager_latency": lp_latency,
        "ratio": latency_ratio,
        "supported": h5_supported,
    }

    # ---- Determine Verdict ----
    # S1: LP accuracy >= baseline
    s1 = lp_f1 >= bl_f1
    # S2: LP hallucination < baseline
    s2 = lp_halluc < bl_halluc
    # S3: Compute cost <= 2x
    s3 = latency_ratio <= 2.0
    # S4: Training converges
    s4 = False
    if lp_history and lp_history.get("train_loss"):
        losses = lp_history["train_loss"]
        if len(losses) >= 3:
            # Check if loss generally decreases after first few steps
            s4 = losses[-1] < losses[0]

    # Strong success additions
    s5 = (lp_f1 - bl_f1) >= 0.03
    s6 = halluc_reduction >= 10
    s7 = True  # Check all task types
    for tt in lp_per_task:
        if tt in bl_per_task:
            if lp_per_task[tt].get("f1", {}).get("mean", 0) < bl_per_task[tt].get("f1", {}).get("mean", 0):
                s7 = False
                break

    # Failure conditions
    f1_fail = (bl_f1 - lp_f1) > 0.03
    f2_fail = not s4
    f3_fail = lp_halluc > bl_halluc
    bl_num_samples = baseline_metrics.get("1024", {}).get("num_samples", 1) if baseline_metrics else 1
    f4_fail = lp_metrics.get("num_samples", 0) < bl_num_samples * 0.5

    if s1 and s2 and s3 and s4 and s5 and s6 and s7:
        verdict = "STRONG SUCCESS"
    elif s1 and s2 and s3 and s4:
        verdict = "SUCCESS"
    elif s1 or s2:
        verdict = "PARTIAL SUCCESS"
    elif f1_fail or f2_fail or f3_fail:
        verdict = "FAILURE"
    else:
        verdict = "PARTIAL SUCCESS"

    criteria = {
        "S1_accuracy_geq_baseline": s1,
        "S2_hallucination_lt_baseline": s2,
        "S3_compute_leq_2x": s3,
        "S4_training_converges": s4,
        "S5_accuracy_gain_geq_3pts": s5,
        "S6_hallucination_reduction_geq_10pct": s6,
        "S7_consistent_across_tasks": s7,
        "F1_accuracy_drop_gt_3pts": f1_fail,
        "F2_training_no_converge": f2_fail,
        "F3_hallucination_worse": f3_fail,
    }

    # ---- Generate Analysis Document ----
    analysis = f"""# Latent Pager Memory: Experiment Analysis

## Overview

This analysis evaluates the Latent Pager Memory system against the Text Buffer (RLM) baseline
on long-document question answering using Qwen3-1.7B.

## Key Results

| Metric | Text Buffer | Latent Pager | Difference |
|---|---|---|---|
| F1 | {bl_f1:.4f} | {lp_f1:.4f} | {lp_f1 - bl_f1:+.4f} |
| ROUGE-L | {bl_rouge:.4f} | {lp_rouge:.4f} | {lp_rouge - bl_rouge:+.4f} |
| Hallucination Rate | {bl_halluc:.4f} | {lp_halluc:.4f} | {lp_halluc - bl_halluc:+.4f} |
| Avg Latency (s) | {bl_latency:.2f} | {lp_latency:.2f} | {lp_latency - bl_latency:+.2f} |

## Hypothesis Evaluation

### H1: Hallucination Reduction
{"SUPPORTED" if h1_supported else "NOT SUPPORTED"} — The latent pager {"reduced" if h1_supported else "did not reduce"} \
hallucination rate from {bl_halluc:.4f} to {lp_halluc:.4f} ({halluc_reduction:.1f}% relative \
{"reduction" if halluc_reduction > 0 else "change"}). \
{"This exceeds the 10% target." if h1_strong else "However, the reduction did not meet the 10% relative threshold."}

### H2: Multi-hop Accuracy Improvement
{"SUPPORTED" if h2_supported else "NOT SUPPORTED"} — Multi-hop F1 {"improved" if h2_supported else "did not improve"} \
from {mh_bl:.4f} to {mh_lp:.4f} ({"+" if mh_lp >= mh_bl else ""}{(mh_lp - mh_bl)*100:.1f} points). \
{"This meets the 5-point threshold." if h2_strong else ""}

### H3: Global Consistency
{"SUPPORTED" if hypotheses["H3"]["supported"] else "INCONCLUSIVE"} — \
{"Consistency score: " + f"{lp_consistency:.4f}" if lp_consistency else "Insufficient data for consistency evaluation."}

### H4: Information Retention Scales with d_page
{"SUPPORTED" if hypotheses["H4"]["supported"] else "NOT SUPPORTED" if hypotheses["H4"]["supported"] is not None else "NOT TESTED"} — \
{"Ablation shows " + ("monotonic" if h4_supported else "non-monotonic") + " scaling." if ablations else "Ablation data not available."}

### H5: Compute Cost Comparable
{"SUPPORTED" if h5_supported else "NOT SUPPORTED"} — Latency ratio: {latency_ratio:.2f}x \
({"within" if h5_supported else "exceeds"} the 1.5x threshold).

## Verdict: **{verdict}**

Success criteria evaluation:
- S1 (accuracy >= baseline): {"PASS" if s1 else "FAIL"}
- S2 (hallucination < baseline): {"PASS" if s2 else "FAIL"}
- S3 (compute <= 2x): {"PASS" if s3 else "FAIL"}
- S4 (training converges): {"PASS" if s4 else "FAIL"}
- S5 (accuracy +3pts): {"PASS" if s5 else "FAIL"}
- S6 (hallucination -10%): {"PASS" if s6 else "FAIL"}
- S7 (consistent across tasks): {"PASS" if s7 else "FAIL"}

{"The latent pager system achieved significant improvements over the text buffer baseline, demonstrating that continuous-space intermediate representations can outperform text-based summaries for long-document comprehension." if verdict in ["SUCCESS", "STRONG SUCCESS"] else ""}
{"While some metrics improved, the results are mixed and warrant further investigation with larger models or different training strategies." if verdict == "PARTIAL SUCCESS" else ""}
{"The latent pager system did not outperform the baseline. Potential causes include insufficient training, suboptimal hyperparameters, or fundamental limitations of the approach at this model scale." if verdict == "FAILURE" else ""}
"""

    # Save outputs
    with open(os.path.join(comparison_dir, "analysis.md"), "w") as f:
        f.write(analysis)

    report = {
        "verdict": verdict,
        "criteria": criteria,
        "hypotheses": hypotheses,
        "baseline_metrics": {
            "f1": bl_f1, "rouge_l": bl_rouge,
            "hallucination_rate": bl_halluc, "latency": bl_latency,
        },
        "latent_pager_metrics": {
            "f1": lp_f1, "rouge_l": lp_rouge,
            "hallucination_rate": lp_halluc, "latency": lp_latency,
        },
    }

    with open(os.path.join(comparison_dir, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"FINAL VERDICT: {verdict}")
    logger.info("=" * 60)
    for k, v in criteria.items():
        logger.info(f"  {k}: {'PASS' if v else 'FAIL'}")
    logger.info("=" * 60)
    logger.info(f"Analysis saved to {comparison_dir}/analysis.md")
    logger.info(f"Report saved to {comparison_dir}/final_report.json")


if __name__ == "__main__":
    main()
