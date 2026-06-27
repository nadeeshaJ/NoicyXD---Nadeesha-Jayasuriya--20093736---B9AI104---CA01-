"""Step 6: Error analysis, case studies, inference benchmarking, and cross-domain summary.

Run:
    python scripts/run_step6_error_analysis.py

Outputs:
    reports/step6/step6_summary.json, inference_benchmarks.json
    reports/figures/step6/confusion_matrix_*.png, case_studies/*.png
    reports/figures/step6/mel_vs_mfcc_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.error_analysis import (
    plot_mel_vs_mfcc,
    run_error_analysis_for_run,
    run_urban_benchmarks,
)
from src.utils import load_config, project_path, save_json

FIG_DIR = project_path("reports", "figures", "step6")
REPORTS_DIR = project_path("reports", "step6")
STEP4_DIR = project_path("reports", "step4")

URBAN_MODELS = (
    ("custom_cnn", "custom_cnn"),
    ("resnet50", "resnet50"),
    ("mobilenetv2", "mobilenetv2"),
)


def build_cross_domain_table() -> list[dict]:
    cross_path = STEP4_DIR / "cross_domain_comparison.csv"
    if cross_path.exists():
        return pd.read_csv(cross_path).to_dict(orient="records")

    with (STEP4_DIR / "step4_summary.json").open(encoding="utf-8") as f:
        summary = json.load(f)
    return summary.get("cross_domain", [])


def main() -> None:
    cfg = load_config()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 6.1 — UrbanSound8K confusion matrix deep dive (3 models)")
    urban_results = []
    for model_name, run_name in URBAN_MODELS:
        # Analyse misclassifications and generate case-study figures per model
        print(f"  Analysing {run_name} ...")
        result = run_error_analysis_for_run(
            cfg,
            "urbansound8k",
            model_name,
            run_name,
            FIG_DIR,
            REPORTS_DIR,
            max_examples=6,
            case_studies=5 if run_name == cfg["experiments"]["best_urban_run"] else 0,
        )
        urban_results.append(result)
        print(
            f"    accuracy={result['test_accuracy']:.4f}, "
            f"misclassified={result['misclassified_count']}, "
            f"top pair={result['top_confused_pairs'][0] if result['top_confused_pairs'] else 'none'}"
        )

    print("\nStep 6.2 — ESC-50 best model error analysis")
    best_esc = cfg["experiments"]["best_esc50_run"]
    esc_result = run_error_analysis_for_run(
        cfg,
        "esc50_animals",
        "mobilenetv2",
        best_esc,
        FIG_DIR,
        REPORTS_DIR,
        max_examples=6,
        case_studies=0,
    )

    print("\nStep 6.3 — Inference benchmarking (UrbanSound8K models)")
    benchmarks = run_urban_benchmarks(cfg, REPORTS_DIR)
    for row in benchmarks:
        print(
            f"  {row['model']}: params={row['total_parameters']:,}, "
            f"size={row['model_file_size_mb']} MB, "
            f"inference={row['inference_ms_mean']:.2f} ms ({row['device']})"
        )

    print("\nStep 6.4 — Mel-spectrogram vs MFCC comparison figure")
    best_urban = cfg["experiments"]["best_urban_run"]
    best_cases_path = REPORTS_DIR / f"error_analysis_{best_urban}.json"
    sample_audio = None
    if best_cases_path.exists():
        with best_cases_path.open(encoding="utf-8") as f:
            best_analysis = json.load(f)
        cases = best_analysis.get("case_studies") or best_analysis.get("misclassified_examples", [])
        if cases:
            sample_audio = cases[0].get("audio_path")

    if sample_audio and Path(sample_audio).exists():
        plot_mel_vs_mfcc(Path(sample_audio), cfg, FIG_DIR / "mel_vs_mfcc_comparison.png")
        print(f"  Saved mel_vs_mfcc_comparison.png using {Path(sample_audio).name}")
    else:
        print("  Skipped MFCC figure (no sample audio found).")

    cross_domain = build_cross_domain_table()
    pd.DataFrame(cross_domain).to_csv(REPORTS_DIR / "cross_domain_summary.csv", index=False)

    best_urban_analysis = next(r for r in urban_results if r["run_name"] == best_urban)
    step6_summary = {
        "step": 6,
        "title": "Error Analysis and Advanced Evaluation",
        "best_urban_run": best_urban,
        "best_esc50_run": best_esc,
        "urban_error_analysis": urban_results,
        "esc50_error_analysis": esc_result,
        "inference_benchmarks": benchmarks,
        "cross_domain": cross_domain,
        "case_studies": best_urban_analysis.get("case_studies", []),
        "figures_dir": str(FIG_DIR),
    }
    save_json(step6_summary, REPORTS_DIR / "step6_summary.json")

    print("\nStep 6 complete.")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
