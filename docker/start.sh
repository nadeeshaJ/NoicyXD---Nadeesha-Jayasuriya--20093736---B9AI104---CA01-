#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  cp docker/.env.example .env
  echo "Created .env from docker/.env.example — edit Supabase keys and PUBLIC_URL, then re-run."
  exit 1
fi

# shellcheck disable=SC1091
set -a
source .env
set +a

if [[ -n "${PUBLIC_URL:-}" ]]; then
  export CORS_ORIGINS="${PUBLIC_URL},http://localhost,http://127.0.0.1"
fi

if ! find experiments -name 'best_model.pt' -print -quit >/dev/null 2>&1; then
  echo "WARNING: No best_model.pt files found under experiments/."
  echo "Upload trained checkpoints before running inference, e.g.:"
  echo "  experiments/urbansound8k/mobilenetv2/best_model.pt"
fi

docker compose up -d --build

echo ""
echo "Stack started."
echo "  UI:     ${PUBLIC_URL:-http://localhost}"
echo "  Health: curl ${PUBLIC_URL:-http://localhost}/api/health"
echo "  Logs:   docker compose logs -f"
