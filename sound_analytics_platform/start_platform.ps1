# Start Sound Analytics Platform (backend + frontend)
# Usage: powershell -ExecutionPolicy Bypass -File start_platform.ps1

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"

Write-Host "Starting FastAPI backend on http://localhost:8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Backend'; python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 3

Write-Host "Starting React frontend on http://localhost:5173 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Frontend'; npm run dev"

Start-Sleep -Seconds 4
Start-Process "http://localhost:5173"
Write-Host "Done. Open http://localhost:5173 if the browser did not launch."
