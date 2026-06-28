import math
from typing import Any

def assess_prediction(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Assess prediction confidence and uncertainty (entropy).
    Maps predictions to reliability categories based on config thresholds.
    """
    app_cfg = cfg.get("app", {})
    unknown_thresh = app_cfg.get("unknown_threshold", 0.40)
    conf_high = app_cfg.get("confidence_high", 0.70)
    conf_medium = app_cfg.get("confidence_medium", 0.40)

    # 1. Get top guess
    top_label = result.get("top_label", "unknown")
    top_confidence = float(result.get("top_confidence", 0.0))

    # 2. Calculate Shannon Entropy
    probs = result.get("probabilities", {})
    entropy = 0.0
    for p in probs.values():
        if p > 0.0:
            entropy -= p * math.log2(p)

    n_classes = len(probs)
    max_entropy = math.log2(n_classes) if n_classes > 1 else 1.0
    entropy_normalized = entropy / max_entropy

    # 3. Determine if out-of-distribution (unknown)
    is_unknown = top_confidence < unknown_thresh

    # 4. Determine reliability level
    if is_unknown:
        reliability_level = "Low"
    elif top_confidence >= conf_high:
        reliability_level = "High"
    elif top_confidence >= conf_medium:
        reliability_level = "Medium"
    else:
        reliability_level = "Low"

    # 5. Determine uncertainty level
    if entropy_normalized < 0.35:
        uncertainty_level = "Low"
    elif entropy_normalized < 0.70:
        uncertainty_level = "Medium"
    else:
        uncertainty_level = "High"

    # 6. Format display name
    if is_unknown:
        display_name = "Unknown / Uncertain"
    else:
        display_name = top_label.replace("_", " ").title()

    # 7. Construct reliability message
    if is_unknown:
        reliability_message = (
            "The model is unsure. The signal features do not strongly match any known class profiles, "
            "indicating an out-of-distribution or noisy sound."
        )
    elif reliability_level == "High":
        reliability_message = (
            "The model is highly confident in this classification. The acoustic features are distinct "
            "and match the trained class profile."
        )
    elif reliability_level == "Medium":
        reliability_message = (
            "The model is moderately confident. There may be background noise or features that partially "
            "overlap with other sound classes."
        )
    else:
        reliability_message = (
            "The model classification confidence is low. Acoustic features are ambiguous or degraded."
        )

    # 8. Construct calibration note
    calibration_note = (
        f"Calibrated Thresholds: >{int(conf_high*100)}% High, <{int(unknown_thresh*100)}% Unknown. "
        f"Normalized Entropy: {entropy_normalized:.3f} ({uncertainty_level} uncertainty)."
    )

    return {
        "reliability_level": reliability_level,
        "is_unknown": is_unknown,
        "display_name": display_name,
        "display_label": display_name,
        "best_guess_label": top_label,
        "entropy_normalized": entropy_normalized,
        "uncertainty_level": uncertainty_level,
        "reliability_message": reliability_message,
        "calibration_note": calibration_note,
        "confidence": top_confidence,
    }
