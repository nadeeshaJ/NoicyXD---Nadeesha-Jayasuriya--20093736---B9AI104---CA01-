"""
Copy trained best_model.pt files into experiments/ for deployment.

Git does not store .pt weights. After clone or on a server, run:

    python scripts/setup_checkpoints.py --source "D:/path/to/experiments"

Or copy from your original training folder (e.g. noicy_XD/experiments).

Also writes test_metrics.json sidecars when present in the source tree.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint_utils import DEPLOYMENT_CHECKPOINTS, verify_all_deployment_checkpoints

DEFAULT_WINDOWS_SOURCE = Path(r"d:\DBS - Sem 2\Deep Learning\CA01\noicy_XD\experiments")


def copy_checkpoint_tree(source_root: Path, rel_path: str) -> bool:
    src = source_root / Path(rel_path).relative_to("experiments")
    dst = PROJECT_ROOT / rel_path
    if not src.exists():
        print(f"  SKIP (not in source): {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  OK  {rel_path} ({dst.stat().st_size / (1024 * 1024):.2f} MB)")

    for sidecar in ("test_metrics.json", "training_summary.json"):
        side_src = src.parent / sidecar
        if side_src.exists():
            shutil.copy2(side_src, dst.parent / sidecar)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Install trained checkpoints for deployment.")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to folder containing urbansound8k/ and esc50_animals/ (usually .../experiments)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing checkpoints; do not copy.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run untrained-weight probe after copy/verify (loads PyTorch).",
    )
    args = parser.parse_args()

    if not args.verify_only:
        source = args.source
        if source is None and DEFAULT_WINDOWS_SOURCE.exists():
            source = DEFAULT_WINDOWS_SOURCE
            print(f"Using default source: {source}")
        if source is None or not source.exists():
            print("ERROR: Provide --source PATH to your trained experiments folder.")
            print('Example: python scripts/setup_checkpoints.py --source "D:/.../experiments"')
            sys.exit(1)

        print("Copying deployment checkpoints...")
        copied = 0
        for _mode, _model, rel_path in DEPLOYMENT_CHECKPOINTS:
            if copy_checkpoint_tree(source, rel_path):
                copied += 1
        print(f"Copied {copied}/{len(DEPLOYMENT_CHECKPOINTS)} files.\n")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = verify_all_deployment_checkpoints(run_probe=args.probe, device=device)

    print(report["summary"])
    for row in report["checkpoints"]:
        flag = {"ok": "OK", "missing": "MISSING", "invalid": "INVALID", "suspect": "SUSPECT"}.get(
            row["status"], row["status"]
        )
        size = f"{row['size_mb']} MB" if row["size_mb"] else "—"
        print(f"  [{flag}] {row['path']} ({size}) — {row['message']}")

    if not report["deploy_ready"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
