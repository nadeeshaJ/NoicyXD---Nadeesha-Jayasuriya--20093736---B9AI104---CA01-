from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from supabase import Client, create_client

from app.config import settings


def get_ml_project_root() -> Path:
    backend_root = Path(__file__).resolve().parents[2]
    ml_root = (backend_root / settings.ml_project_root).resolve()
    if not ml_root.exists():
        raise FileNotFoundError(f"ML project root not found: {ml_root}")
    return ml_root


def ensure_ml_path() -> Path:
    ml_root = get_ml_project_root()
    root_str = str(ml_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return ml_root


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    key = settings.supabase_service_role_key or settings.supabase_anon_key
    if not key:
        raise RuntimeError("SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY is required.")
    return create_client(settings.supabase_url, key)
