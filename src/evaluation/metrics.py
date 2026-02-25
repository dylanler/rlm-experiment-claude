"""
Evaluation metrics: F1, Exact Match, ROUGE-L, and hallucination rate.
"""

import re
import string
from collections import Counter

from rouge_score import rouge_scorer


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def compute_exact_match(prediction: str, gold: str) -> float:
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F-measure."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(gold, prediction)
    return scores["rougeL"].fmeasure


def compute_hallucination_rate(
    generated_answer: str,
    source_document: str,
    gold_answer: str,
) -> float:
    """
    Compute hallucination rate using n-gram overlap heuristic.

    Decomposes generated answer into sentences/claims.
    For each claim, checks if it overlaps with the source document or gold answer.
    Claims with no significant overlap are considered hallucinated.

    Returns: fraction of claims that are hallucinated (0.0 to 1.0)
    """
    claims = _split_into_claims(generated_answer)
    if not claims:
        return 0.0

    source_lower = source_document.lower()
    gold_lower = gold_answer.lower()

    hallucinated = 0
    for claim in claims:
        claim_lower = claim.lower().strip()
        if not claim_lower:
            continue

        # Check if claim is supported by source or gold
        claim_tokens = set(normalize_answer(claim).split())
        source_tokens = set(normalize_answer(source_document).split())
        gold_tokens = set(normalize_answer(gold_answer).split())

        if not claim_tokens:
            continue

        # Overlap with source
        source_overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
        # Overlap with gold
        gold_overlap = len(claim_tokens & gold_tokens) / len(claim_tokens)

        # If less than 50% token overlap with both source and gold, consider hallucinated
        if source_overlap < 0.5 and gold_overlap < 0.5:
            hallucinated += 1

    total_claims = len([c for c in claims if c.strip()])
    if total_claims == 0:
        return 0.0

    return hallucinated / total_claims


def _split_into_claims(text: str) -> list[str]:
    """Split text into atomic claims (sentences)."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip().split()) >= 3]


def compute_all_metrics(
    prediction: str,
    gold_answer: str,
    source_document: str,
) -> dict:
    """Compute all metrics for a single prediction."""
    return {
        "exact_match": compute_exact_match(prediction, gold_answer),
        "f1": compute_f1(prediction, gold_answer),
        "rouge_l": compute_rouge_l(prediction, gold_answer),
        "hallucination_rate": compute_hallucination_rate(
            prediction, source_document, gold_answer
        ),
    }
