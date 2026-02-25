# Comparison: Latent Pager vs Text Buffer Baseline

| Metric | Text Buffer (Baseline) | Latent Pager | Difference | Significant |
|---|---|---|---|---|
| f1 | 0.0182 | 0.0257 | +0.0075 | True |
| rouge_l | 0.0177 | 0.0260 | +0.0083 | True |
| exact_match | 0.0000 | 0.0000 | +0.0000 | N/A |
| hallucination_rate | 0.2920 | 0.5795 | +0.2875 | True |

| Avg Latency (s) | 19.55 | 7.65 | | |
| Peak Memory (GB) | 1.02 | 1.82 | | |

## Per-Task Type Breakdown


### multi_hop_reasoning

| Metric | Baseline | Latent Pager |
|---|---|---|
| f1 | 0.0155 | 0.0195 |
| rouge_l | 0.0142 | 0.0192 |
| hallucination_rate | 0.2647 | 0.4906 |

### single_fact_extraction

| Metric | Baseline | Latent Pager |
|---|---|---|
| f1 | 0.0206 | 0.0314 |
| rouge_l | 0.0210 | 0.0323 |
| hallucination_rate | 0.3172 | 0.6615 |
