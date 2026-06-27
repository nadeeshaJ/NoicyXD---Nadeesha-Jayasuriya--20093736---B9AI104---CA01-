"""Copy required CA1 code + outputs to Code_Submission folder with detailed README.

Run:
    python scripts/build_code_submission.py

Creates:
    D:/DBS - Sem 2/Deep Learning/CA01/Code_Submission/
    (config, src, scripts, notebooks, app, outputs/figures, outputs/metrics, README.md)
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_SUBMISSION = PROJECT_ROOT.parent / "Code_Submission"

PIPELINE_SCRIPTS = {
    "setup_datasets.py": (
        "Prepare UrbanSound8K folder layout and download ESC-50 Animals from Hugging Face.",
        "data/raw/urbansound8k/, data/raw/esc50/",
    ),
    "run_step1_eda.py": (
        "Exploratory data analysis on both datasets (class counts, duration, folds).",
        "reports/step1/, reports/figures/step1/*.png",
    ),
    "run_step2_preprocess.py": (
        "Create train/val/test splits; convert audio to Mel-spec PNG images; draw pipeline diagram.",
        "data/processed/, data/splits/, reports/figures/step2/preprocessing_pipeline_diagram.png",
    ),
    "run_step3_train.py": (
        "Train Custom CNN, ResNet50, MobileNetV2 on UrbanSound8K; save architecture + comparison figures.",
        "experiments/urbansound8k/, reports/figures/step3/, reports/step3/",
    ),
    "run_ca1_ablation_studies.py": (
        "CA1 hyperparameter ablations: MobileNetV2 head sizes (64/128/256) and Custom CNN dropout.",
        "reports/final/ca1_ablation_summary.json, reports/figures/final/ablation_*.png",
    ),
    "run_step4_esc50.py": (
        "ESC-50 transfer learning: urban transfer, from-scratch, ImageNet-only comparison.",
        "experiments/esc50_animals/, reports/step4/, reports/figures/step4/",
    ),
    "run_step6_error_analysis.py": (
        "Confusion matrices, case studies, inference benchmarking, cross-domain summary.",
        "reports/step6/, reports/figures/step6/",
    ),
    "verify_step5_deployment.py": (
        "Verify Streamlit app loads models and runs sample predictions.",
        "reports/step5/step5_verification.json, reports/figures/step5/app_demo_*.png",
    ),
    "capture_app_demo.py": (
        "Capture Streamlit screenshot images for the report.",
        "reports/figures/step5/app_demo_urban.png, app_demo_animal.png",
    ),
}

REPORT_SCRIPTS = {
    "generate_model_summaries.py": (
        "Generate torchinfo model.summary() PNG figures for all three models.",
        "reports/figures/final/model_summary_*.png",
    ),
    "generate_final_report.py": (
        "Build main Word report, cover sheet, and development notes.",
        "reports/final/Final_Assignment_Report.docx, Cover_Sheet.docx, Final_Development_Notes.docx",
    ),
    "generate_phase9_presentation.py": (
        "Generate presentation slides, demo script, and speaker notes.",
        "reports/final/Presentation_Slides.pptx, Live_Demo_Script.docx",
    ),
    "generate_ca1_checklist.py": (
        "Generate CA1 pre-submission compliance checklist.",
        "reports/final/CA1_PreSubmission_Checklist.docx",
    ),
    "generate_step1_report.py": "Step 1 Word report from EDA outputs.",
    "generate_step2_report.py": "Step 2 Word report from preprocessing outputs.",
    "generate_step3_report.py": "Step 3 Word report from model training outputs.",
    "generate_step4_report.py": "Step 4 Word report from ESC-50 experiments.",
    "generate_step5_report.py": "Step 5 Word report from deployment verification.",
    "generate_step6_report.py": "Step 6 Word report from error analysis.",
}

SRC_MODULES = {
    "audio_utils.py": "Load, resample, pad/trim audio clips.",
    "spectrogram.py": "STFT, Mel filterbank, dB conversion, RGB image export.",
    "preprocess.py": "Batch preprocessing of dataset splits to PNG images.",
    "augmentation.py": "SpecAugment time/frequency masking and noise augmentation.",
    "splits.py": "UrbanSound8K fold-10 splits and ESC-50 stratified splits.",
    "dataset.py": "PyTorch Dataset for Mel-spec PNG images.",
    "models/custom_cnn.py": "Model 1 — conventional CNN baseline from scratch.",
    "models/resnet50_model.py": "Model 2 — ResNet50 transfer learning with 10-class head.",
    "models/mobilenetv2_model.py": "Model 3 — MobileNetV2 transfer learning with custom head.",
    "train.py": "Training loop, early stopping, two-phase transfer learning.",
    "evaluate.py": "Test evaluation, classification report, confusion matrix.",
    "predict.py": "Single-file inference for Streamlit app.",
    "error_analysis.py": "Misclassification analysis and inference benchmarks.",
    "gradcam.py": "Grad-CAM explainability overlays.",
    "utils.py": "Config loading, paths, JSON helpers.",
}

NOTEBOOKS = {
    "01_eda_urbansound8k.ipynb": "Interactive EDA on UrbanSound8K.",
    "02_preprocessing_demo.ipynb": "Demo of waveform → Mel-spec → RGB conversion.",
    "03_results_analysis.ipynb": "Review model metrics and result tables.",
    "04_ca1_model_training.ipynb": "CA1-style notebook aligned with course 5-cell workflow.",
}

REPORT_JSON = [
    "reports/step1/step1_summary.json",
    "reports/step1/dataset_inventory.csv",
    "reports/step2/step2_summary.json",
    "reports/step2/step2_validation.json",
    "reports/step2/preprocessing_validation.csv",
    "reports/step3/step3_summary.json",
    "reports/step3/model_specifications.json",
    "reports/step3/model_comparison_urbansound8k.csv",
    "reports/step4/step4_summary.json",
    "reports/step4/esc50_model_comparison.csv",
    "reports/step4/cross_domain_comparison.csv",
    "reports/step5/step5_verification.json",
    "reports/step6/step6_summary.json",
    "reports/step6/inference_benchmarks.json",
    "reports/step6/cross_domain_summary.csv",
    "reports/final/ca1_ablation_summary.json",
]

EXPERIMENT_METRICS = [
    "experiments/urbansound8k/custom_cnn/test_metrics.json",
    "experiments/urbansound8k/custom_cnn/training_summary.json",
    "experiments/urbansound8k/resnet50/test_metrics.json",
    "experiments/urbansound8k/resnet50/training_summary.json",
    "experiments/urbansound8k/mobilenetv2/test_metrics.json",
    "experiments/urbansound8k/mobilenetv2/training_summary.json",
    "experiments/esc50_animals/custom_cnn_from_scratch/test_metrics.json",
    "experiments/esc50_animals/custom_cnn_from_scratch/training_summary.json",
    "experiments/esc50_animals/mobilenetv2_urbansound_transfer/test_metrics.json",
    "experiments/esc50_animals/mobilenetv2_urbansound_transfer/training_summary.json",
    "experiments/esc50_animals/mobilenetv2_imagenet_only/test_metrics.json",
    "experiments/esc50_animals/mobilenetv2_imagenet_only/training_summary.json",
]


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree(src: Path, dst: Path, pattern: str = "*") -> int:
    count = 0
    if not src.exists():
        return count
    for path in src.rglob(pattern):
        if path.is_file():
            copy_file(path, dst / path.relative_to(src))
            count += 1
    return count


def _script_section(title: str, scripts: dict) -> list[str]:
    lines = [f"## {title}", ""]
    for name, info in scripts.items():
        if isinstance(info, tuple):
            desc, outputs = info
        else:
            desc, outputs = info, "reports/stepN/ (Word report)"
        lines.append(f"### `{name}`")
        lines.append(f"- **Purpose:** {desc}")
        lines.append(f"- **Run:** `python scripts/{name}`")
        lines.append(f"- **Outputs:** `{outputs}`")
        lines.append("")
    return lines


def write_readme(manifest: dict) -> None:
    lines = [
        "# B9AI104 Deep Learning — CA1 Code Submission",
        "",
        "**Student:** Nadeesha Jayasuriya  ",
        f"**Generated:** {date.today().strftime('%d %B %Y')}  ",
        "**Project:** Environmental Sound Classification via Mel-Spectrogram Images",
        "",
        "---",
        "",
        "## What this folder contains",
        "",
        "This is the **code + result outputs** package for CA1. It includes:",
        "",
        "- Python source code (`src/`)",
        "- Runnable pipeline scripts (`scripts/`)",
        "- Jupyter notebooks (`notebooks/`)",
        "- Streamlit demo app (`app/`)",
        "- Generated figures and metrics (`outputs/`)",
        "- Shared configuration (`config/config.yaml`)",
        "",
        "**Submit separately on Moodle (Word documents):**",
        "- Final Assignment Report",
        "- Development Notes",
        "- Cover Sheet",
        "- Presentation Slides",
        "",
        "**Not included (too large — regenerate locally):**",
        "- Raw audio datasets (`data/raw/`)",
        "- Processed Mel-spec images (`data/processed/`)",
        "- Model checkpoint files (`experiments/**/*.pt`)",
        "",
        "---",
        "",
        "## Folder structure",
        "",
        "```",
        "Code_Submission/",
        "├── README.md              ← this file",
        "├── requirements.txt       ← Python packages",
        "├── config/config.yaml     ← pipeline settings",
        "├── src/                   ← core modules",
        "├── scripts/               ← run pipeline + generate figures/reports",
        "├── notebooks/             ← Jupyter notebooks",
        "├── app/streamlit_app.py   ← web demo",
        "└── outputs/",
        "    ├── figures/           ← all PNG diagrams and result plots",
        "    └── metrics/           ← JSON/CSV result summaries",
        "```",
        "",
        "---",
        "",
        "## Quick start (reproduce from scratch)",
        "",
        "1. Install dependencies:",
        "   ```bash",
        "   pip install -r requirements.txt",
        "   ```",
        "",
        "2. Download datasets into `data/raw/` (UrbanSound8K + ESC-50 Animals):",
        "   ```bash",
        "   python scripts/setup_datasets.py",
        "   ```",
        "",
        "3. Run the pipeline in order:",
        "   ```bash",
        "   python scripts/run_step1_eda.py",
        "   python scripts/run_step2_preprocess.py",
        "   python scripts/run_step3_train.py",
        "   python scripts/run_ca1_ablation_studies.py",
        "   python scripts/run_step4_esc50.py",
        "   python scripts/run_step6_error_analysis.py",
        "   python scripts/verify_step5_deployment.py",
        "   ```",
        "",
        "4. Generate diagrams and reports:",
        "   ```bash",
        "   python scripts/generate_model_summaries.py",
        "   python scripts/generate_final_report.py",
        "   python scripts/generate_phase9_presentation.py",
        "   ```",
        "",
        "5. Run Streamlit demo:",
        "   ```bash",
        "   python -m streamlit run app/streamlit_app.py",
        "   ```",
        "",
        "> **Note:** Run commands from the full project root (`noicy_XD/`) where `data/` and `experiments/` exist.",
        "> This Code_Submission folder is a copy for Moodle upload; paths in scripts assume project layout.",
        "",
        "---",
        "",
        "## Key results (UrbanSound8K, fold-10 test)",
        "",
        "| Model | Accuracy | Macro F1 | Role |",
        "|-------|----------|----------|------|",
        "| Custom CNN | 75.0% | 0.767 | Model 1 — conventional baseline |",
        "| ResNet50 | 81.2% | 0.811 | Model 2 — transfer learning |",
        "| MobileNetV2 | 82.7% | 0.831 | Model 3 — customised transfer (deployed) |",
        "",
        "---",
        "",
    ]
    lines.extend(_script_section("Pipeline scripts (`scripts/`)", PIPELINE_SCRIPTS))
    lines.append("---")
    lines.append("")
    lines.extend(_script_section("Report & diagram scripts (`scripts/`)", REPORT_SCRIPTS))
    lines.append("---")
    lines.append("")
    lines.append("## Core modules (`src/`)")
    lines.append("")
    for name, desc in SRC_MODULES.items():
        lines.append(f"- **`{name}`** — {desc}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Notebooks (`notebooks/`)")
    lines.append("")
    for name, desc in NOTEBOOKS.items():
        lines.append(f"- **`{name}`** — {desc}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Outputs included (`outputs/`)")
    lines.append("")
    lines.append("### `outputs/figures/` — report diagrams and plots")
    lines.append("")
    lines.append("| Subfolder | Contents |")
    lines.append("|-----------|----------|")
    lines.append("| `step1/` | EDA charts (class distribution, duration, waveforms) |")
    lines.append("| `step2/` | Preprocessing pipeline diagram (Fig 2.1), examples, SpecAugment |")
    lines.append("| `step3/` | Architecture diagrams, model comparison, training curves, confusion matrices |")
    lines.append("| `step4/` | ESC-50 cross-domain comparison, transfer learning curves |")
    lines.append("| `step5/` | Streamlit app demo screenshots |")
    lines.append("| `step6/` | Error analysis, case studies, inference comparison, MFCC vs Mel |")
    lines.append("| `final/` | Model summary PNGs, ablation study charts |")
    lines.append("")
    lines.append(f"**Total figures copied:** {manifest['figure_count']}")
    lines.append("")
    lines.append("### `outputs/metrics/` — numeric results")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `step3_summary.json` | Best model, macro F1, model specs |")
    lines.append("| `model_comparison_urbansound8k.csv` | Accuracy/F1 comparison table |")
    lines.append("| `inference_benchmarks.json` | Params, file size, inference ms, F1 |")
    lines.append("| `ca1_ablation_summary.json` | Head-size and dropout ablation results |")
    lines.append("| `experiments__*` JSON files | Per-model test_metrics and training_summary |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Important report figures → source script")
    lines.append("")
    lines.append("| Report figure | Generated by | File |")
    lines.append("|---------------|--------------|------|")
    lines.append("| Figure 2.1 — Preprocessing pipeline | `run_step2_preprocess.py` | `outputs/figures/step2/preprocessing_pipeline_diagram.png` |")
    lines.append("| Figure 2.2 — Custom CNN architecture | `run_step3_train.py` | `outputs/figures/step3/architecture_custom_cnn.png` |")
    lines.append("| Figure 2.3 — Custom CNN model summary | `generate_model_summaries.py` | `outputs/figures/final/model_summary_custom_cnn.png` |")
    lines.append("| Figure 2.5 — Transfer learning strategy | `run_step3_train.py` | `outputs/figures/step3/architecture_transfer_learning.png` |")
    lines.append("| Figure 5.1 — Model comparison | `run_step3_train.py` | `outputs/figures/step3/model_comparison_urbansound8k.png` |")
    lines.append("| Figures 5.2–5.4 — Training curves | `run_step3_train.py` | `outputs/figures/step3/urbansound8k/*/training_history.png` |")
    lines.append("| Figure 5.6 — Confusion matrix | `run_step6_error_analysis.py` | `outputs/figures/step6/confusion_matrix_mobilenetv2.png` |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Package built automatically — {manifest['file_count']} files, {manifest['figure_count']} figures.*")
    lines.append("")

    (CODE_SUBMISSION / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if CODE_SUBMISSION.exists():
        shutil.rmtree(CODE_SUBMISSION)
    CODE_SUBMISSION.mkdir(parents=True)

    file_count = 0
    copy_file(PROJECT_ROOT / "requirements.txt", CODE_SUBMISSION / "requirements.txt")
    file_count += 1

    file_count += copy_tree(PROJECT_ROOT / "config", CODE_SUBMISSION / "config")
    file_count += copy_tree(PROJECT_ROOT / "src", CODE_SUBMISSION / "src", "*.py")

    scripts_dst = CODE_SUBMISSION / "scripts"
    for name in list(PIPELINE_SCRIPTS) + list(REPORT_SCRIPTS) + ["build_code_submission.py"]:
        if copy_file(PROJECT_ROOT / "scripts" / name, scripts_dst / name):
            file_count += 1

    file_count += copy_tree(PROJECT_ROOT / "notebooks", CODE_SUBMISSION / "notebooks", "*.ipynb")
    file_count += copy_tree(PROJECT_ROOT / "app", CODE_SUBMISSION / "app", "*.py")

    fig_count = copy_tree(
        PROJECT_ROOT / "reports" / "figures",
        CODE_SUBMISSION / "outputs" / "figures",
        "*.png",
    )
    file_count += fig_count

    metrics_dst = CODE_SUBMISSION / "outputs" / "metrics"
    for rel in REPORT_JSON:
        if copy_file(PROJECT_ROOT / rel, metrics_dst / Path(rel).name):
            file_count += 1
    for rel in EXPERIMENT_METRICS:
        name = rel.replace("/", "__").replace("\\", "__")
        if copy_file(PROJECT_ROOT / rel, metrics_dst / name):
            file_count += 1

    manifest = {
        "generated": date.today().isoformat(),
        "destination": str(CODE_SUBMISSION),
        "file_count": file_count,
        "figure_count": fig_count,
    }
    (metrics_dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    file_count += 1

    write_readme(manifest)

    print(f"Code submission created: {CODE_SUBMISSION}")
    print(f"  README: {CODE_SUBMISSION / 'README.md'}")
    print(f"  Files:  {file_count}")
    print(f"  Figures: {fig_count}")


if __name__ == "__main__":
    main()
