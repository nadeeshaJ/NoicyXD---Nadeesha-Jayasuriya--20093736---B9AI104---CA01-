# Start the Docker stack from repo root (Windows / local testing).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".env")) {
    Copy-Item "docker\.env.example" ".env"
    Write-Host "Created .env from docker\.env.example — edit Supabase keys and PUBLIC_URL, then re-run."
    exit 1
}

$envContent = Get-Content ".env" -Raw
if ($envContent -match '(?m)^PUBLIC_URL=(.+)$') {
    $publicUrl = $Matches[1].Trim()
    if ($publicUrl) {
        $env:CORS_ORIGINS = "$publicUrl,http://localhost,http://127.0.0.1"
    }
}

$checkpoints = Get-ChildItem -Path "experiments" -Recurse -Filter "best_model.pt" -ErrorAction SilentlyContinue
if (-not $checkpoints) {
    Write-Error @"
No best_model.pt files under experiments/.
Inference will fail. Install checkpoints first:
  python scripts/setup_checkpoints.py --source /path/to/trained/experiments
See experiments/README.md
"@
}

docker compose up -d --build

Write-Host ""
Write-Host "Stack started."
Write-Host "  UI:     http://localhost"
Write-Host "  Health: curl http://localhost/api/health"
Write-Host "  Logs:   docker compose logs -f"
