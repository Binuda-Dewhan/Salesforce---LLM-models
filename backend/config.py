"""
backend/config.py
=================
Single canonical environment configuration loader for the entire backend.

Design decisions:
- Loads the .env file ONCE at import time; never again per-request.
- All modules must import from here instead of defining their own load_env_file().
- Provides typed config accessors for each setting so nothing reads os.getenv() inline.
- The `reload_env()` function is available for tests that need a fresh load.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env Loader
# ---------------------------------------------------------------------------

def _find_env_file() -> Optional[Path]:
    """Walks up from the config file's location to find a .env file."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / ".env",   # repo root (expected location)
        here / ".env",          # inside backend/ (fallback)
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_env(*, overwrite: bool = False) -> None:
    """
    Parses the nearest .env file and loads values into os.environ.

    Args:
        overwrite: If True, existing env vars are overwritten by .env values.
                   Defaults to False (env vars already set take precedence).
    """
    env_path = _find_env_file()
    if env_path is None:
        logger.debug("No .env file found; skipping env load.")
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, raw_val = line.partition("=")
                key = key.strip()
                # Strip inline comments and surrounding quotes
                val = raw_val.split("#")[0].strip().strip('"').strip("'")
                if overwrite or key not in os.environ or not os.environ[key]:
                    os.environ[key] = val
        logger.debug("Loaded environment from %s", env_path)
    except OSError as exc:
        logger.warning("Could not read .env file at %s: %s", env_path, exc)


def reload_env() -> None:
    """Forces a fresh load of the .env file, overwriting existing values. For tests only."""
    load_env(overwrite=True)


# Load once at module import time
load_env()


# ---------------------------------------------------------------------------
# Typed Config Accessors — use these instead of os.getenv() inline
# ---------------------------------------------------------------------------

class _Config:
    """Lazy, typed config accessors. Reads from os.environ (already loaded above)."""

    @staticmethod
    def salesforce_instance_url() -> str:
        return os.getenv("SF_INSTANCE_URL", "").strip().rstrip("/")

    @staticmethod
    def salesforce_access_token() -> str:
        return os.getenv("SF_ACCESS_TOKEN", "").strip()

    @staticmethod
    def salesforce_username() -> str:
        return os.getenv("SALESFORCE_USERNAME", "").strip()

    @staticmethod
    def salesforce_password() -> str:
        return os.getenv("SALESFORCE_PASSWORD", "").strip()

    @staticmethod
    def salesforce_security_token() -> str:
        return os.getenv("SALESFORCE_SECURITY_TOKEN", "").strip()

    @staticmethod
    def salesforce_api_version() -> str:
        return os.getenv("SALESFORCE_API_VERSION", "v61.0").strip()

    @staticmethod
    def gemini_api_key() -> str:
        return os.getenv("GEMINI_API_KEY", "").strip()

    @staticmethod
    def mistral_api_key() -> str:
        return os.getenv("MISTRAL_API_KEY", "").strip()

    @staticmethod
    def use_mcp() -> bool:
        return os.getenv("USE_MCP", "false").lower() == "true"

    @staticmethod
    def mcp_server_url() -> str:
        return os.getenv("MCP_SERVER_URL", "http://localhost:5000").rstrip("/")

    @staticmethod
    def cors_allowed_origins() -> list:
        raw = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
        return [o.strip() for o in raw.split(",") if o.strip()]

    @staticmethod
    def lora_adapter_path() -> Optional[str]:
        """Returns explicit adapter path if set via env var, else None (auto-discover)."""
        return os.getenv("LORA_ADAPTER_PATH", "").strip() or None


cfg = _Config()
