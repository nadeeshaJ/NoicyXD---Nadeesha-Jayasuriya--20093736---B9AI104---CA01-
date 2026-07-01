Server raw audio install — noicyXD
================================

Package: server_data_raw.zip (1037 WAV files)

WHY: data/raw is not in git. Without these files the Datasets tab
and Play Sound return: Audio file not found ... under /app/data/raw/...

ON THE SERVER (repo root, same folder as docker-compose.yml):

  1. Upload server_data_raw.zip to the server
  2. unzip -o deploy/server_data_raw.zip
     (creates data/raw/urbansound8k/audio/fold10/ and data/raw/esc50/audio/)
  3. Verify:
       ls data/raw/urbansound8k/audio/fold10/159742-8-0-0.wav
       ls data/raw/esc50/audio/*.wav | head
  4. Restart:
       docker compose up -d

Docker mounts ./data -> /app/data (see docker-compose.yml).

Minimum contents:
  - urbansound8k/audio/fold10/*.wav  (837 files — test set)
  - esc50/audio/*.wav                (200 files — animal mode)

Also required separately (not in this zip):
  - experiments/**/best_model.pt     (run setup_checkpoints.py on server)
