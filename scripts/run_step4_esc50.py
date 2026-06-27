"""Step 4: ESC-50 transfer learning, cross-domain comparison, and error analysis.

Run:
    python scripts/run_step4_esc50.py

Compares three ESC-50 strategies:
    1. UrbanSound8K MobileNetV2 → fine-tune on animals
    2. Custom CNN from scratch on animals
    3. ImageNet-only MobileNetV2 on animals (best ESC-50 F1)

Outputs:
    experiments/esc50_animals/{run}/best_model.pt, test_metrics.json
    reports/step4/step4_summary.json, cross_domain_comparison.csv
    reports/figures/step4/cross_domain_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.error_analysis import run_error_analysis_for_run
from src.train import train_model
from src.utils import load_config, project_path, save_json, set_seed

FIG_DIR = project_path("reports", "figures", "step4")
REPORTS_DIR = project_path("reports", "step4")
STEP3_CSV = project_path("reports", "step3", "model_comparison_urbansound8k.csv")
BEST_URBAN_MODEL = "mobilenetv2"
URBAN_CHECKPOINT = project_path(
    "experiments", "urbansound8k", BEST_URBAN_MODEL, "best_model.pt"
)


def plot_cross_domain(comparison_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=comparison_df, x="approach", y="macro_f1", hue="domain", ax=axes[0])
    axes[0].set_title("Macro F1 — Urban vs Animal Domains")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=comparison_df, x="approach", y="f1_drop", ax=axes[1], color="indianred")
    axes[1].set_title("Cross-Domain Performance Drop (Urban F1 - ESC-50 F1)")
    axes[1].set_ylabel("F1 drop")

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "cross_domain_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    set_seed(42)
    cfg = load_config()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not URBAN_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing UrbanSound8K checkpoint: {URBAN_CHECKPOINT}")

    print("Step 4.1 — Transfer learning: UrbanSound8K MobileNetV2 -> ESC-50 Animals")
    # Load urban-trained backbone, fine-tune on animal classes (cross-domain transfer)
    transfer_summary = train_model(
        model_name=BEST_URBAN_MODEL,
        dataset_key="esc50_animals",
        cfg=cfg,
        epochs=20,
        run_name="mobilenetv2_urbansound_transfer",
        backbone_checkpoint=URBAN_CHECKPOINT,
        figure_step="step4",
    )

    print("\nStep 4.2 — Baseline: Custom CNN from scratch on ESC-50 Animals")
    scratch_summary = train_model(
        model_name="custom_cnn",
        dataset_key="esc50_animals",
        cfg=cfg,
        epochs=20,
        run_name="custom_cnn_from_scratch",
        figure_step="step4",
    )

    print("\nStep 4.3 — Comparison: ImageNet-only MobileNetV2 on ESC-50")
    imagenet_summary = train_model(
        model_name=BEST_URBAN_MODEL,
        dataset_key="esc50_animals",
        cfg=cfg,
        epochs=20,
        run_name="mobilenetv2_imagenet_only",
        figure_step="step4",
    )

    urban_df = pd.read_csv(STEP3_CSV)
    urban_best = urban_df[urban_df["model"] == BEST_URBAN_MODEL].iloc[0]

    esc_rows = [
        {
            "approach": "Urban transfer (MobileNetV2)",
            "run_name": transfer_summary["run_name"],
            "domain": "ESC-50 Animals",
            **transfer_summary["test_metrics"],
        },
        {
            "approach": "From scratch (Custom CNN)",
            "run_name": scratch_summary["run_name"],
            "domain": "ESC-50 Animals",
            **scratch_summary["test_metrics"],
        },
        {
            "approach": "ImageNet only (MobileNetV2)",
            "run_name": imagenet_summary["run_name"],
            "domain": "ESC-50 Animals",
            **imagenet_summary["test_metrics"],
        },
    ]
    esc_df = pd.DataFrame(esc_rows)
    esc_df.to_csv(REPORTS_DIR / "esc50_model_comparison.csv", index=False)

    cross_rows = [
        {
            "approach": "MobileNetV2 (UrbanSound8K)",
            "domain": "UrbanSound8K",
            "macro_f1": urban_best["macro_f1"],
            "accuracy": urban_best["accuracy"],
            "f1_drop": 0.0,
        },
        {
            "approach": "Urban transfer (MobileNetV2)",
            "domain": "ESC-50 Animals",
            "macro_f1": transfer_summary["test_metrics"]["macro_f1"],
            "accuracy": transfer_summary["test_metrics"]["accuracy"],
            "f1_drop": urban_best["macro_f1"] - transfer_summary["test_metrics"]["macro_f1"],
        },
        {
            "approach": "From scratch (Custom CNN)",
            "domain": "ESC-50 Animals",
            "macro_f1": scratch_summary["test_metrics"]["macro_f1"],
            "accuracy": scratch_summary["test_metrics"]["accuracy"],
            "f1_drop": urban_best["macro_f1"] - scratch_summary["test_metrics"]["macro_f1"],
        },
        {
            "approach": "ImageNet only (MobileNetV2)",
            "domain": "ESC-50 Animals",
            "macro_f1": imagenet_summary["test_metrics"]["macro_f1"],
            "accuracy": imagenet_summary["test_metrics"]["accuracy"],
            "f1_drop": urban_best["macro_f1"] - imagenet_summary["test_metrics"]["macro_f1"],
        },
    ]
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(REPORTS_DIR / "cross_domain_comparison.csv", index=False)
    plot_cross_domain(cross_df)

    print("\nStep 4.4 — Error analysis")
    error_transfer = run_error_analysis_for_run(
        cfg, "esc50_animals", BEST_URBAN_MODEL,
        "mobilenetv2_urbansound_transfer", FIG_DIR, REPORTS_DIR,
    )
    error_scratch = run_error_analysis_for_run(
        cfg, "esc50_animals", "custom_cnn",
        "custom_cnn_from_scratch", FIG_DIR, REPORTS_DIR,
    )

    best_esc = max(
        [transfer_summary, scratch_summary, imagenet_summary],
        key=lambda s: s["test_metrics"]["macro_f1"],
    )

    step4_summary = {
        "step": 4,
        "title": "ESC-50 Transfer Learning and Cross-Domain Analysis",
        "urban_source_model": BEST_URBAN_MODEL,
        "urban_source_macro_f1": float(urban_best["macro_f1"]),
        "esc50_runs": [transfer_summary, scratch_summary, imagenet_summary],
        "best_esc50_run": best_esc["run_name"],
        "best_esc50_macro_f1": best_esc["test_metrics"]["macro_f1"],
        "cross_domain": cross_rows,
        "error_analysis": {
            "transfer": error_transfer,
            "scratch": error_scratch,
        },
    }
    save_json(step4_summary, REPORTS_DIR / "step4_summary.json")

    print("\nStep 4 complete.")
    print(f"  Best ESC-50 run: {best_esc['run_name']} (macro F1={best_esc['test_metrics']['macro_f1']:.4f})")
    print(f"  Reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
