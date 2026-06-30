from __future__ import annotations

_EQUIVALENT_LABELS = frozenset({("dog_bark", "dog"), ("dog", "dog_bark")})


def labels_match(predicted: str | None, ground_truth: str | None) -> bool:
    if not predicted or not ground_truth:
        return False
    if predicted == ground_truth:
        return True
    return (predicted, ground_truth) in _EQUIVALENT_LABELS
