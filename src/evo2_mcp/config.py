"""Runtime configuration loaded from environment variables.

All knobs are opt-in and documented in `.env.example`. The API key is only
ever read from the ``NVIDIA_API_KEY`` environment variable (or a ``.env``
file) — never from code or a config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


# Official hosted endpoint (deprecated on build.nvidia.com but still the
# documented target for this project). The NIM container uses
# /biology/arc/evo2/forward on port 8000; point EVO2_MCP_BASE_URL at it.
DEFAULT_BASE_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-7b"

DEFAULT_MAX_SEQUENCE_LENGTH = 1_000_000  # Evo2 7B was trained to 1M context.


def _split_dirs(raw: str) -> list[Path]:
    if not raw:
        return []
    sep = ";" if os.name == "nt" and ";" in raw else ":" if ":" in raw else os.pathsep
    return [Path(p).expanduser().resolve() for p in raw.split(sep) if p.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_key: Optional[str]
    base_url: str
    timeout: float
    max_retries: int
    max_concurrency: int
    allowed_dirs: tuple[Path, ...]
    output_dir: Path
    allow_ambiguous: bool
    max_sequence_length: int
    raw_inline_max: int
    max_per_position: int
    logits_layer: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.environ.get("NVIDIA_API_KEY") or None,
            base_url=os.environ.get("EVO2_MCP_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            timeout=float(os.environ.get("EVO2_MCP_TIMEOUT", "120")),
            max_retries=int(os.environ.get("EVO2_MCP_MAX_RETRIES", "4")),
            max_concurrency=int(os.environ.get("EVO2_MCP_MAX_CONCURRENCY", "2")),
            allowed_dirs=tuple(_split_dirs(os.environ.get("EVO2_MCP_ALLOWED_DIRS", ""))),
            output_dir=Path(os.environ.get("EVO2_MCP_OUTPUT_DIR", "./output")).expanduser().resolve(),
            allow_ambiguous=_env_bool("EVO2_MCP_ALLOW_AMBIGUOUS"),
            max_sequence_length=int(os.environ.get(
                "EVO2_MCP_MAX_SEQUENCE_LENGTH", str(DEFAULT_MAX_SEQUENCE_LENGTH)
            )),
            raw_inline_max=int(os.environ.get("EVO2_MCP_RAW_INLINE_MAX", "4096")),
            max_per_position=int(os.environ.get("EVO2_MCP_MAX_PER_POSITION", "5000")),
            logits_layer=os.environ.get("EVO2_MCP_LOGITS_LAYER", "auto"),
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise Evo2ConfigError(
                "NVIDIA_API_KEY is not set. Export it in the environment or a .env file. "
                "Get a key from https://build.nvidia.com/ (top-right → Get API Key). "
                "This server never falls back to a hard-coded key."
            )
        return self.api_key


class Evo2ConfigError(RuntimeError):
    """Raised when required configuration (typically the API key) is missing."""
