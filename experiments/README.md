# Model checkpoints (not in Git)

Trained `.pt` weights are **too large for Git** (~125 MB total). They live here at runtime only.

## Required files for deployment

| Path | Size (approx) |
|------|----------------|
| `urbansound8k/mobilenetv2/best_model.pt` | 8.8 MB |
| `urbansound8k/resnet50/best_model.pt` | 90 MB |
| `urbansound8k/custom_cnn/best_model.pt` | 25 MB |
| `esc50_animals/mobilenetv2_imagenet_only/best_model.pt` | 8.8 MB |

`test_metrics.json` sidecars **are** in Git (for benchmark display).

## Install after clone

```powershell
python scripts/setup_checkpoints.py --source "D:\DBS - Sem 2\Deep Learning\CA01\noicy_XD\experiments"
python scripts/setup_checkpoints.py --verify-only
```

## Docker

`docker-compose.yml` mounts `./experiments` into the backend container. Upload checkpoints to the server under `experiments/` before starting inference.

## Verify before go-live

```powershell
python scripts/verify_publish.py --build
```
