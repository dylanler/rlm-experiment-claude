"""
Global consistency checker: evaluates whether multiple answers about the same
document are mutually consistent.
"""

import re
from collections import Counter


def global_consistency(answers: list[str], document: str) -> float:
    """
    Given multiple answers about the same document, check that
    answers are mutually consistent using token overlap heuristic.

    For each pair of answers, checks for contradictions by looking
    at entity/fact overlap and divergence patterns.

    Returns: fraction of answer pairs that are consistent (0.0 to 1.0)
    """
    if len(answers) < 2:
        return 1.0

    consistent_pairs = 0
    total_pairs = 0

    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            total_pairs += 1
            if _are_consistent(answers[i], answers[j], document):
                consistent_pairs += 1

    return consistent_pairs / total_pairs if total_pairs > 0 else 1.0


def _are_consistent(answer_a: str, answer_b: str, document: str) -> bool:
    """
    Check if two answers are consistent with each other.

    Uses simple heuristics:
    1. Extract entities/numbers from both answers
    2. Check if shared entities have contradictory contexts
    3. Check if both answers are grounded in the document
    """
    entities_a = _extract_entities(answer_a)
    entities_b = _extract_entities(answer_b)

    shared_entities = entities_a & entities_b
    if not shared_entities:
        # No shared entities — can't detect contradiction
        return True

    # Check if both answers' facts are grounded in the document
    doc_lower = document.lower()
    a_grounded = sum(1 for e in entities_a if e in doc_lower) / max(len(entities_a), 1)
    b_grounded = sum(1 for e in entities_b if e in doc_lower) / max(len(entities_b), 1)

    # If both are well-grounded, they're likely consistent
    return a_grounded > 0.3 and b_grounded > 0.3


def _extract_entities(text: str) -> set[str]:
    """Extract simple entities: numbers, capitalized words, quoted strings."""
    entities = set()

    # Numbers
    numbers = re.findall(r"\b\d+\.?\d*\b", text)
    entities.update(numbers)

    # Capitalized multi-word phrases
    cap_phrases = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
    entities.update(p.lower() for p in cap_phrases)

    # Quoted strings
    quoted = re.findall(r'"([^"]+)"', text)
    entities.update(q.lower() for q in quoted)

    return entities
