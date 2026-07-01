"""Pre-publish checks — code in git, Docker inputs, checkpoints, frontend build.

Run before git push / docker compose up:

    python scripts/verify_publish.py
    python scripts/verify_publish.py --build   # also run npm run build
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint_utils import DEPLOYMENT_CHECKPOINTS, verify_all_deployment_checkpoints

# Paths that must exist on disk and be tracked in git for a clean clone + Docker build.
GIT_REQUIRED: list[str] = [
    "requirements.txt",
    "config/config.yaml",
    "docker-compose.yml",
    "docker/backend/Dockerfile",
    "docker/frontend/Dockerfile",
    "docker/nginx/default.conf",
    "docker/.env.example",
    "data/splits/urbansound8k/test_processed.csv",
    "data/splits/esc50_animals/test_processed.csv",
    "reports/step1/step1_summary.json",
    "reports/step6/inference_benchmarks.json",
    "reports/step5/step5_verification.json",
    "sound_analytics_platform/backend/requirements.txt",
    "sound_analytics_platform/backend/app/main.py",
    "sound_analytics_platform/backend/app/routes.py",
    "sound_analytics_platform/backend/app/schemas.py",
    "sound_analytics_platform/backend/app/services/inference.py",
    "sound_analytics_platform/backend/app/services/export_report.py",
    "sound_analytics_platform/frontend/package.json",
    "sound_analytics_platform/frontend/package-lock.json",
    "sound_analytics_platform/frontend/index.html",
    "sound_analytics_platform/frontend/vite.config.ts",
    "sound_analytics_platform/supabase/migrations/001_initial_schema.sql",
    "sound_analytics_platform/supabase/migrations/002_dataset_input_source.sql",
    "sound_analytics_platform/supabase/migrations/002_prediction_metadata.sql",
    "sound_analytics_platform/supabase/migrations/003_ground_truth_audit.sql",
    "sound_analytics_platform/supabase/migrations/004_test_macro_recall.sql",
    "scripts/setup_checkpoints.py",
    "scripts/verify_publish.py",
    "experiments/README.md",
]

# Intentionally not in git — must exist on the server at runtime.
RUNTIME_ONLY: list[str] = [
    rel for _mode, _model, rel in DEPLOYMENT_CHECKPOINTS
]

passed: list[str] = []
failed: list[str] = []
warned: list[str] = []


def ok(msg: str) -> None:
    passed.append(msg)
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    failed.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    warned.append(msg)
    print(f"  WARN  {msg}")


def is_tracked(rel: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def check_git_files() -> None:
    print("\n=== Git-tracked deploy files ===")
    for rel in GIT_REQUIRED:
        path = PROJECT_ROOT / rel
        if not path.exists():
            fail(f"Missing on disk: {rel}")
            continue
        if not is_tracked(rel):
            fail(f"Exists but not in git: {rel}")
            continue
        ok(rel)


def check_uncommitted() -> None:
    print("\n=== Uncommitted changes (publish blockers) ===")
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    deploy_prefixes = (
        "sound_analytics_platform/",
        "src/",
        "config/",
        "docker/",
        "reports/step6/",
        "data/splits/",
        ".gitignore",
    )
    deploy_changes = [
        ln for ln in lines if any(p in ln for p in deploy_prefixes)
    ]
    if deploy_changes:
        for ln in deploy_changes:
            warn(f"Uncommitted: {ln.strip()}")
    else:
        ok("No uncommitted deploy-related changes")


def check_checkpoints() -> None:
    print("\n=== Model checkpoints (runtime — not in git) ===")
    report = verify_all_deployment_checkpoints()
    for row in report["checkpoints"]:
        if row["status"] == "ok":
            ok(f"{row['path']} ({row['size_mb']} MB)")
        else:
            fail(f"{row['path']}: {row['message']}")
    if not report["deploy_ready"]:
        warn(
            "After git clone on server, run: "
            "python scripts/setup_checkpoints.py --source /path/to/experiments"
        )


def check_docker_inputs() -> None:
    print("\n=== Docker build context ===")
    for rel in (
        "requirements.txt",
        "config/",
        "src/",
        "data/splits/",
        "reports/",
        "sound_analytics_platform/backend/",
        "sound_analytics_platform/frontend/package.json",
        "sound_analytics_platform/frontend/package-lock.json",
    ):
        path = PROJECT_ROOT / rel
        if path.exists():
            ok(f"Docker context: {rel}")
        else:
            fail(f"Docker context missing: {rel}")


def check_frontend_build(run_build: bool) -> None:
    print("\n=== Frontend build ===")
    frontend = PROJECT_ROOT / "sound_analytics_platform" / "frontend"
    if not (frontend / "node_modules").exists():
        warn("node_modules missing — run: cd sound_analytics_platform/frontend && npm ci")
        if not run_build:
            return
    if not run_build:
        ok("Skipped (use --build to run npm run build)")
        return
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    result = subprocess.run(
        [npm, "run", "build"],
        cwd=frontend,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok("npm run build succeeded")
    else:
        fail("npm run build failed")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify repo is ready to publish/deploy.")
    parser.add_argument("--build", action="store_true", help="Run npm run build")
    args = parser.parse_args()

    print("Publish verification — noicyXD / sound_analytics_platform")
    check_git_files()
    check_uncommitted()
    check_docker_inputs()
    check_checkpoints()
    check_frontend_build(args.build)

    print("\n=== Summary ===")
    print(f"  Passed: {len(passed)}  Failed: {len(failed)}  Warnings: {len(warned)}")
    if failed:
        print("\nFix FAIL items before publishing.")
        sys.exit(1)
    if warned:
        print("\nReview WARN items — commit deploy changes and install checkpoints on server.")
    else:
        print("\nReady to publish.")


if __name__ == "__main__":
    main()
