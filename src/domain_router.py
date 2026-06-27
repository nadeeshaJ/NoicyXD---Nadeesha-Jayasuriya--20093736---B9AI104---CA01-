"""Smart domain routing between urban and animal expert models.

Used by: app/streamlit_app.py — auto-detects whether uploaded audio is
urban (UrbanSound8K) or animal (ESC-50) and routes to the correct checkpoint.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.predict import predict_audio


URBAN_ONLY_HINTS = {
    # Classes that only exist in UrbanSound8K — strong signal for urban domain
    "air_conditioner",
    "car_horn",
    "children_playing",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
}

ANIMAL_ONLY_HINTS = {
    # Classes that only exist in ESC-50 Animals — strong signal for animal domain
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects",
    "sheep",
    "crow",
}

OVERLAP_HINTS = {"dog", "dog_bark"}  # appears in both datasets — no domain bonus


def _domain_score(result: dict[str, Any], domain: str) -> float:
    top_label = result["top_label"]
    top_conf = float(result["top_confidence"])

    if domain == "urban":
        if top_label in URBAN_ONLY_HINTS:
            return top_conf + 0.08       # boost when urban-only class detected
        if top_label in OVERLAP_HINTS:
            return top_conf                # neutral — could be either domain
        if top_label in ANIMAL_ONLY_HINTS:
            return top_conf * 0.35         # penalise — urban model shouldn't predict cow
        return top_conf * 0.75

    if top_label in ANIMAL_ONLY_HINTS:
        return top_conf + 0.08
    if top_label in OVERLAP_HINTS:
        return top_conf
    if top_label in URBAN_ONLY_HINTS:
        return top_conf * 0.35
    return top_conf * 0.75


@torch.no_grad()
def route_audio_domain(
    urban_model: nn.Module,
    urban_class_names: list[str],
    urban_model_name: str,
    animal_model: nn.Module,
    animal_class_names: list[str],
    animal_model_name: str,
    audio_bytes: bytes,
    device: torch.device,
    cfg: dict,
    margin: float = 0.03,
) -> dict[str, Any]:
    """
    Run lightweight dual-expert probes and route to the stronger domain.

    Both experts use the deployed MobileNetV2 checkpoints. The router compares
    adjusted domain scores so urban-only classes (e.g. siren) and animal-only
    classes (e.g. cow) are not mis-routed through overlapping labels like dog.
    """
    urban_probe = predict_audio(
        urban_model,
        urban_class_names,
        urban_model_name,
        audio_bytes,
        device=device,
        cfg=cfg,
        top_k=3,
    )
    animal_probe = predict_audio(
        animal_model,
        animal_class_names,
        animal_model_name,
        audio_bytes,
        device=device,
        cfg=cfg,
        top_k=3,
    )

    # Adjust raw confidence scores using domain-specific class hints
    urban_score = _domain_score(urban_probe, "urban")
    animal_score = _domain_score(animal_probe, "animal")

    if urban_score >= animal_score + margin:
        chosen = "urban"
        reason = (
            f"Urban expert score {urban_score:.3f} vs animal {animal_score:.3f} "
            f"(top: {urban_probe['top_label']} {urban_probe['top_confidence']:.1%})."
        )
    elif animal_score >= urban_score + margin:
        chosen = "animal"
        reason = (
            f"Animal expert score {animal_score:.3f} vs urban {urban_score:.3f} "
            f"(top: {animal_probe['top_label']} {animal_probe['top_confidence']:.1%})."
        )
    else:
        chosen = "urban" if urban_score >= animal_score else "animal"
        reason = (
            f"Scores tied within margin — defaulting to {chosen} "
            f"(urban {urban_score:.3f}, animal {animal_score:.3f})."
        )

    return {
        "domain": chosen,
        "reason": reason,
        "urban_probe": urban_probe,
        "animal_probe": animal_probe,
        "urban_score": urban_score,
        "animal_score": animal_score,
    }
