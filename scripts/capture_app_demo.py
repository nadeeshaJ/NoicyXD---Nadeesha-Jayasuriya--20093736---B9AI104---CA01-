"""Capture static demo previews mimicking the Streamlit app output.

Run:
    python scripts/capture_app_demo.py

Outputs:
    reports/figures/step5/app_demo_urban.png
    reports/figures/step5/app_demo_animal.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch

from src.predict import load_mode_model, plot_mel_spectrogram, plot_waveform, predict_audio
from src.utils import load_config, project_path

OUT_DIR = project_path("reports", "figures", "step5")
SAMPLES = {
    "urban": project_path(
        "data", "raw", "urbansound8k", "audio", "fold10", "100795-3-0-0.wav"
    ),
    "animal": project_path("data", "raw", "esc50", "audio", "3-180977-A-0.wav"),
}


def capture_mode(mode: str, audio_path: Path) -> Path:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, deploy_cfg, _, _ = load_mode_model(mode, cfg, device)
    result = predict_audio(
        model,
        class_names,
        deploy_cfg["model_name"],
        audio_path,
        device=device,
        cfg=cfg,
        top_k=3,
    )

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 0.8])

    ax_wave = fig.add_subplot(gs[0, 0])
    wave_fig = plot_waveform(result["waveform"], result["sample_rate"])
    wave_ax = wave_fig.axes[0]
    ax_wave.plot(
        wave_ax.lines[0].get_xdata(),
        wave_ax.lines[0].get_ydata(),
        color="#2563eb",
        linewidth=0.8,
    )
    ax_wave.set_title("Waveform")
    ax_wave.set_xlabel("Time (s)")
    plt.close(wave_fig)

    ax_mel = fig.add_subplot(gs[0, 1])
    mel_fig = plot_mel_spectrogram(result["mel_spectrogram"])
    mel_ax = mel_fig.axes[0]
    img = mel_ax.images[0].get_array()
    ax_mel.imshow(img, aspect="auto", origin="lower", cmap="magma")
    ax_mel.set_title("Mel-spectrogram")
    plt.close(mel_fig)

    ax_rgb = fig.add_subplot(gs[1, :])
    ax_rgb.imshow(result["rgb_image"])
    ax_rgb.set_title("Model input (224x224 RGB)")
    ax_rgb.axis("off")

    ax_pred = fig.add_subplot(gs[2, :])
    ax_pred.axis("off")
    title = mode.title() + " mode"
    top = result["predictions"][0]
    lines = [
        f"{title} — sample: {audio_path.name}",
        f"Top prediction: {top['label'].replace('_', ' ').title()} ({top['confidence']:.1%})",
        "",
        "Top 3:",
    ]
    for pred in result["predictions"]:
        bar = "#" * int(pred["confidence"] * 30)
        lines.append(f"  {pred['label']:20s} {pred['confidence']:6.1%}  {bar}")
    ax_pred.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)

    out_path = OUT_DIR / f"app_demo_{mode}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Environmental Sound Classification — Streamlit App Preview", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    for mode, path in SAMPLES.items():
        out = capture_mode(mode, path)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
