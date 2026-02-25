"""
Statistical significance testing: paired bootstrap test for comparing
two systems' metric distributions.
"""

import numpy as np


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    num_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap significance test.

    Tests whether system B is significantly better than system A.

    Args:
        scores_a: per-sample scores for system A (baseline)
        scores_b: per-sample scores for system B (treatment)
        num_bootstrap: number of bootstrap samples
        seed: random seed

    Returns: dict with:
        - mean_a, mean_b: mean scores
        - diff: mean_b - mean_a
        - p_value: probability that B is NOT better than A
        - significant: whether p < 0.05
        - ci_lower, ci_upper: 95% confidence interval for the difference
    """
    rng = np.random.RandomState(seed)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)

    assert len(scores_a) == len(scores_b), "Score arrays must have equal length"

    observed_diff = scores_b.mean() - scores_a.mean()

    diffs = []
    count_a_better = 0
    for _ in range(num_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample_a = scores_a[indices]
        sample_b = scores_b[indices]
        diff = sample_b.mean() - sample_a.mean()
        diffs.append(diff)
        if diff <= 0:
            count_a_better += 1

    diffs = np.array(diffs)
    p_value = count_a_better / num_bootstrap

    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    return {
        "mean_a": float(scores_a.mean()),
        "mean_b": float(scores_b.mean()),
        "diff": float(observed_diff),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "num_bootstrap": num_bootstrap,
    }
