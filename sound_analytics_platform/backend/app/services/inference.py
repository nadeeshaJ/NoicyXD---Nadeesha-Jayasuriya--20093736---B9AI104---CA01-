from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from app.services.supabase_client import ensure_ml_path


class InferenceService:
    def __init__(self) -> None:
        ensure_ml_path()
        from src.confidence import assess_prediction
        from src.domain_router import route_audio_domain
        from src.predict import (
            available_models_for_mode,
            load_benchmark_table,
            load_mode_model,
            plot_mel_spectrogram,
            plot_waveform,
            predict_audio,
            predict_with_gradcam,
        )
        from src.utils import load_config

        self.route_audio_domain = route_audio_domain
        self.available_models_for_mode = available_models_for_mode
        self.load_benchmark_table = load_benchmark_table
        self.load_mode_model = load_mode_model
        self.plot_mel_spectrogram = plot_mel_spectrogram
        self.plot_waveform = plot_waveform
        self.predict_audio = predict_audio
        self.predict_with_gradcam = predict_with_gradcam
        self.load_config = load_config
        self.assess_prediction = assess_prediction

        self.cfg = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_cache: dict[tuple[str, str], tuple[Any, list[str], str]] = {}

    def get_model_bundle(self, mode: str, model_name: str):
        key = (mode, model_name)
        if key not in self._model_cache:
            model, class_names, _, chosen_name, _ = self.load_mode_model(
                mode,
                self.cfg,
                self.device,
                model_name=model_name,
            )
            self._model_cache[key] = (model, class_names, chosen_name)
        return self._model_cache[key]

    def get_router_experts(self):
        urban = self.get_model_bundle("urban", "mobilenetv2")
        animal = self.get_model_bundle("animal", "mobilenetv2")
        return urban, animal

    @staticmethod
    def figure_to_base64(fig: plt.Figure) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @staticmethod
    def ndarray_to_base64_png(array: np.ndarray) -> str:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @staticmethod
    def _serialize_router(router_info: dict[str, Any]) -> dict[str, Any]:
        urban = router_info["urban_metrics"]
        animal = router_info["animal_metrics"]
        return {
            "domain": router_info["domain"],
            "reason": router_info["reason"],
            "primary_reason": router_info.get("primary_reason", router_info["reason"]),
            "hint_note": router_info.get("hint_note"),
            "urban_score": router_info["urban_strength"],
            "animal_score": router_info["animal_strength"],
            "confidence_gap": router_info["confidence_gap"],
            "selected_uncertainty": router_info["selected_uncertainty"],
            "urban_metrics": urban,
            "animal_metrics": animal,
            "urban_probe": {
                "top_label": router_info["urban_probe"]["top_label"],
                "top_confidence": router_info["urban_probe"]["top_confidence"],
            },
            "animal_probe": {
                "top_label": router_info["animal_probe"]["top_label"],
                "top_confidence": router_info["animal_probe"]["top_confidence"],
            },
        }

    def run_prediction(
        self,
        audio_bytes: bytes,
        mode: str,
        model_name: str,
        gradcam: bool = True,
    ) -> dict[str, Any]:
        effective_mode = mode
        router_info = None

        if mode == "auto":
            (urban_model, urban_classes, urban_name), (animal_model, animal_classes, animal_name) = self.get_router_experts()
            router_info = self.route_audio_domain(
                urban_model,
                urban_classes,
                urban_name,
                animal_model,
                animal_classes,
                animal_name,
                audio_bytes,
                self.device,
                self.cfg,
            )
            effective_mode = router_info["domain"]

        available = self.available_models_for_mode(effective_mode, self.cfg)
        chosen_model = model_name if model_name in available else (
            self.cfg["deployment"][effective_mode]["model_name"]
        )

        model, class_names, resolved_name = self.get_model_bundle(effective_mode, chosen_model)

        if gradcam:
            result = self.predict_with_gradcam(
                model,
                class_names,
                resolved_name,
                audio_bytes,
                device=self.device,
                cfg=self.cfg,
                top_k=3,
                measure_latency=True,
            )
        else:
            result = self.predict_audio(
                model,
                class_names,
                resolved_name,
                audio_bytes,
                device=self.device,
                cfg=self.cfg,
                top_k=3,
                measure_latency=True,
            )

        assessment = self.assess_prediction(result, self.cfg)

        payload: dict[str, Any] = {
            "processing_mode": mode,
            "effective_mode": effective_mode,
            "model_key": resolved_name,
            "top_label": result["top_label"],
            "top_confidence": result["top_confidence"],
            "predictions": result["predictions"],
            "probabilities": result["probabilities"],
            "inference_ms": result.get("inference_ms"),
            "device_used": str(self.device),
            "assessment": assessment,
            "waveform_png": self.figure_to_base64(self.plot_waveform(result["waveform"], result["sample_rate"])),
            "mel_png": self.figure_to_base64(self.plot_mel_spectrogram(result["mel_spectrogram"])),
            "rgb_png": self.ndarray_to_base64_png(result["rgb_image"]),
        }

        if router_info:
            payload["router"] = self._serialize_router(router_info)

        if gradcam and "gradcam_figure" in result:
            payload["gradcam_png"] = self.figure_to_base64(result["gradcam_figure"])
            payload["gradcam_summary"] = result.get("gradcam_summary")

        benchmarks = self.load_benchmark_table(self.cfg)
        if resolved_name in benchmarks:
            payload["benchmark"] = benchmarks[resolved_name]

        return payload

    def resolve_effective_mode(self, audio_bytes: bytes, mode: str) -> tuple[str, dict[str, Any] | None]:
        if mode != "auto":
            return mode, None
        (urban_model, urban_classes, urban_name), (animal_model, animal_classes, animal_name) = self.get_router_experts()
        router_info = self.route_audio_domain(
            urban_model,
            urban_classes,
            urban_name,
            animal_model,
            animal_classes,
            animal_name,
            audio_bytes,
            self.device,
            self.cfg,
        )
        return router_info["domain"], router_info

    def preview_audio(
        self,
        audio_bytes: bytes,
        input_source: str = "upload",
        filename: str | None = None,
    ) -> dict[str, Any]:
        import io

        import librosa
        import numpy as np

        from src.predict import preprocess_uploaded_audio

        audio_cfg = self.cfg["audio"]
        target_sr = audio_cfg["sample_rate"]
        target_duration = audio_cfg["duration_sec"]

        try:
            raw, raw_sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
        except Exception as exc:
            raise ValueError(f"Unable to decode audio file: {exc}") from exc

        if raw.ndim == 1:
            channels = "mono"
            raw_mono = raw
        else:
            channels = f"{raw.shape[0]} channels -> mono"
            raw_mono = librosa.to_mono(raw)

        original_duration = len(raw_mono) / float(raw_sr)
        y, sr, mel_norm, _ = preprocess_uploaded_audio(audio_bytes, self.cfg)

        checks = [
            {
                "name": "Sample rate",
                "target": f"{target_sr} Hz",
                "actual": f"{sr} Hz",
                "passed": sr == target_sr,
            },
            {
                "name": "Channels",
                "target": "mono",
                "actual": channels,
                "passed": True,
            },
            {
                "name": "Duration",
                "target": f"{target_duration:.1f} s",
                "actual": f"{len(y) / sr:.2f} s processed",
                "passed": abs((len(y) / sr) - target_duration) < 0.05,
            },
            {
                "name": "Normalization",
                "target": "Mel min-max [0, 1]",
                "actual": f"{float(np.min(mel_norm)):.2f} – {float(np.max(mel_norm)):.2f}",
                "passed": float(np.min(mel_norm)) >= 0.0 and float(np.max(mel_norm)) <= 1.0,
            },
        ]

        return {
            "valid": all(item["passed"] for item in checks),
            "filename": filename,
            "input_source": input_source,
            "original_duration_sec": round(original_duration, 3),
            "processed_duration_sec": round(len(y) / sr, 3),
            "sample_rate": sr,
            "channels": channels,
            "waveform_png": self.figure_to_base64(self.plot_waveform(y, sr)),
            "mel_png": self.figure_to_base64(self.plot_mel_spectrogram(mel_norm)),
            "validation_checks": checks,
        }

    def compare_models(
        self,
        audio_bytes: bytes,
        mode: str,
    ) -> dict[str, Any]:
        effective_mode, _ = self.resolve_effective_mode(audio_bytes, mode)
        available = self.available_models_for_mode(effective_mode, self.cfg)
        benchmarks = self.load_benchmark_table(self.cfg)
        labels = self.cfg["app"]["model_labels"]

        comparisons: list[dict[str, Any]] = []
        for model_name in available:
            model, class_names, resolved_name = self.get_model_bundle(effective_mode, model_name)
            result = self.predict_audio(
                model,
                class_names,
                resolved_name,
                audio_bytes,
                device=self.device,
                cfg=self.cfg,
                top_k=3,
                measure_latency=True,
            )
            bench = benchmarks.get(resolved_name, {})
            comparisons.append(
                {
                    "model_key": resolved_name,
                    "display_name": labels.get(resolved_name, resolved_name),
                    "top_label": result["top_label"],
                    "top_confidence": float(result["top_confidence"]),
                    "inference_ms": result.get("inference_ms"),
                    "checkpoint_size_mb": bench.get("model_file_size_mb"),
                    "benchmark_latency_ms": bench.get("inference_ms_mean"),
                }
            )

        comparisons.sort(key=lambda row: row.get("top_confidence") or 0.0, reverse=True)
        return {"effective_mode": effective_mode, "comparisons": comparisons}


inference_service = InferenceService()
