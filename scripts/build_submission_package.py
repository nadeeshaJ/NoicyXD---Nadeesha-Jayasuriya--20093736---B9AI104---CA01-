"""Assemble CA1 submission folder: code, scripts, outputs, and manifest.

Run:
    python scripts/build_submission_package.py

Creates:
    submission/  (full Moodle package with documents + code + figures)
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = PROJECT_ROOT / "submission"

# Core pipeline scripts (data → train → evaluate → figures → reports)
PIPELINE_SCRIPTS = [
    "setup_datasets.py",
    "run_step1_eda.py",
    "run_step2_preprocess.py",
    "run_step3_train.py",
    "run_step4_esc50.py",
    "run_step6_error_analysis.py",
    "run_ca1_ablation_studies.py",
    "verify_step5_deployment.py",
    "capture_app_demo.py",
]

# Report / diagram generation scripts
REPORT_SCRIPTS = [
    "generate_model_summaries.py",
    "generate_final_report.py",
    "generate_phase9_presentation.py",
    "generate_ca1_checklist.py",
    "generate_step1_report.py",
    "generate_step2_report.py",
    "generate_step3_report.py",
    "generate_step4_report.py",
    "generate_step5_report.py",
    "generate_step6_report.py",
    "build_submission_package.py",
]

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
    "reports/final/final_report_manifest.json",
    "reports/final/phase9_manifest.json",
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
            rel = path.relative_to(src)
            copy_file(path, dst / rel)
            count += 1
    return count


def write_readme(manifest: dict) -> None:
    lines = [
        "B9AI104 Deep Learning — CA1 Submission Package",
        f"Generated: {date.today().strftime('%d %B %Y')}",
        "Student: Nadeesha Jayasuriya",
        "",
        "WHAT TO SUBMIT ON MOODLE (documents):",
        "  1. Final_Assignment_Report.docx",
        "  2. Final_Development_Notes.docx",
        "  3. Cover_Sheet.docx",
        "  4. Presentation_Slides.pptx",
        "  5. This submission/ folder (or zip) — code + results + figures",
        "",
        "FOLDER STRUCTURE:",
        "  config/          — shared pipeline settings (config.yaml)",
        "  src/             — core Python modules (preprocess, models, train, evaluate)",
        "  scripts/         — runnable pipeline and figure/report generators",
        "  notebooks/       — EDA, preprocessing demo, results analysis, CA1 training notebook",
        "  app/             — Streamlit deployment demo",
        "  outputs/figures/ — all report diagrams and result plots",
        "  outputs/metrics/ — JSON/CSV summaries and experiment metrics",
        "  documents/       — Word report, dev notes, cover sheet, presentation",
        "  requirements.txt — Python dependencies",
        "",
        "HOW TO REPRODUCE (from project root, after datasets are in data/raw/):",
        "  pip install -r requirements.txt",
        "  python scripts/setup_datasets.py",
        "  python scripts/run_step1_eda.py",
        "  python scripts/run_step2_preprocess.py",
        "  python scripts/run_step3_train.py",
        "  python scripts/run_ca1_ablation_studies.py",
        "  python scripts/run_step4_esc50.py",
        "  python scripts/run_step6_error_analysis.py",
        "  python scripts/generate_model_summaries.py",
        "  python scripts/generate_final_report.py",
        "  python scripts/generate_phase9_presentation.py",
        "  python -m streamlit run app/streamlit_app.py",
        "",
        "NOT INCLUDED (too large for submission zip):",
        "  - Raw audio datasets (download UrbanSound8K + HF ESC-50 Animals)",
        "  - Processed Mel-spec PNG images (regenerated by run_step2_preprocess.py)",
        "  - Model checkpoints (.pt files in experiments/)",
        "",
        "KEY RESULTS (UrbanSound8K fold-10 test):",
        "  Custom CNN:    accuracy 75.0%, macro F1 0.767",
        "  ResNet50:      accuracy 81.2%, macro F1 0.811",
        "  MobileNetV2:   accuracy 82.7%, macro F1 0.831  (deployed model)",
        "",
        f"Files copied: {manifest['file_count']}",
        f"Figures: {manifest['figure_count']}",
    ]
    (SUBMISSION / "SUBMISSION_README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if SUBMISSION.exists():
        shutil.rmtree(SUBMISSION)
    SUBMISSION.mkdir(parents=True)

    file_count = 0

    # Root files
    for name in ("requirements.txt", "README.md"):
        if copy_file(PROJECT_ROOT / name, SUBMISSION / name):
            file_count += 1

    # Config + source
    file_count += copy_tree(PROJECT_ROOT / "config", SUBMISSION / "config")
    file_count += copy_tree(PROJECT_ROOT / "src", SUBMISSION / "src", "*.py")

    # Scripts
    scripts_dst = SUBMISSION / "scripts"
    scripts_dst.mkdir(parents=True)
    for name in PIPELINE_SCRIPTS + REPORT_SCRIPTS:
        if copy_file(PROJECT_ROOT / "scripts" / name, scripts_dst / name):
            file_count += 1

    # Notebooks
    file_count += copy_tree(PROJECT_ROOT / "notebooks", SUBMISSION / "notebooks", "*.ipynb")

    # Streamlit app
    file_count += copy_tree(PROJECT_ROOT / "app", SUBMISSION / "app", "*.py")

    # Figures → outputs/figures
    fig_count = copy_tree(
        PROJECT_ROOT / "reports" / "figures",
        SUBMISSION / "outputs" / "figures",
        "*.png",
    )
    file_count += fig_count

    # Metrics / summaries → outputs/metrics
    metrics_dst = SUBMISSION / "outputs" / "metrics"
    for rel in REPORT_JSON:
        if copy_file(PROJECT_ROOT / rel, metrics_dst / Path(rel).name):
            file_count += 1

    for rel in EXPERIMENT_METRICS:
        name = rel.replace("/", "__").replace("\\", "__")
        if copy_file(PROJECT_ROOT / rel, metrics_dst / name):
            file_count += 1

    manifest = {
        "generated": date.today().isoformat(),
        "student": "Nadeesha Jayasuriya",
        "module": "B9AI104 Deep Learning",
        "file_count": file_count,
        "figure_count": fig_count,
        "pipeline_scripts": PIPELINE_SCRIPTS,
        "report_scripts": REPORT_SCRIPTS,
        "notebooks": [
            "01_eda_urbansound8k.ipynb",
            "02_preprocessing_demo.ipynb",
            "03_results_analysis.ipynb",
            "04_ca1_model_training.ipynb",
        ],
        "documents_submit_separately": [
            "Final_Assignment_Report.docx",
            "Final_Development_Notes.docx",
            "Cover_Sheet.docx",
            "Presentation_Slides.pptx",
        ],
    }
    manifest_path = SUBMISSION / "outputs" / "submission_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    file_count += 1

    write_readme(manifest)

    # Documents (if present in reports/final or common locations)
    docs_dst = SUBMISSION / "documents"
    docs_dst.mkdir(parents=True, exist_ok=True)
    doc_candidates = [
        PROJECT_ROOT / "reports" / "final" / name
        for name in (
            "Final_Assignment_Report.docx",
            "Final_Assignment_Report_updated.docx",
            "Final_Development_Notes.docx",
            "Cover_Sheet.docx",
            "Presentation_Slides.pptx",
            "Live_Demo_Script.docx",
            "Presentation_Speaker_Notes.docx",
            "CA1_PreSubmission_Checklist.docx",
        )
    ]
    downloads_report = Path.home() / "Downloads" / "Final_Assignment_Report_updated.docx"
    doc_candidates.append(downloads_report)
    for src in doc_candidates:
        if src.exists() and copy_file(src, docs_dst / src.name):
            file_count += 1

    print(f"Submission package created: {SUBMISSION}")
    print(f"  Files copied: {file_count}")
    print(f"  Figures: {fig_count}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
