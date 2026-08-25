"""Shared pytest fixtures. All tests are fully mocked — no real API calls."""

from __future__ import annotations

import base64
import io

import httpx
import numpy as np
import pytest

from evo2_mcp.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        api_key="test-key-123",
        base_url="https://health.api.nvidia.com/v1/biology/arc/evo2-7b",
        timeout=5.0,
        max_retries=2,
        max_concurrency=2,
        allowed_dirs=(),
        output_dir=tmp_path / "output",
        allow_ambiguous=False,
        max_sequence_length=1_000_000,
        raw_inline_max=4096,
        max_per_position=5000,
        logits_layer="auto",
    )


@pytest.fixture
def no_key_settings(tmp_path):
    return Settings(
        api_key=None,
        base_url="https://health.api.nvidia.com/v1/biology/arc/evo2-7b",
        timeout=5.0,
        max_retries=2,
        max_concurrency=2,
        allowed_dirs=(),
        output_dir=tmp_path / "output",
        allow_ambiguous=False,
        max_sequence_length=1_000_000,
        raw_inline_max=4096,
        max_per_position=5000,
        logits_layer="auto",
    )


@pytest.fixture
def fast_settings(monkeypatch):
    """Remove backoff sleeps so retry tests run instantly."""
    monkeypatch.setattr("evo2_mcp.api_client._backoff_seconds", lambda attempt, retry_after: 0.0)
    return None


# --- helpers ---------------------------------------------------------------

NUC_INDEX = {"A": 65, "C": 67, "G": 71, "T": 84}


def make_npz_bytes(layer_arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **layer_arrays)
    return buf.getvalue()


def npz_json_response(layer_arrays: dict[str, np.ndarray], elapsed_ms: int = 7) -> httpx.Response:
    npz = make_npz_bytes(layer_arrays)
    payload = {"data": base64.b64encode(npz).decode("ascii"), "elapsed_ms": elapsed_ms}
    return httpx.Response(200, json=payload, headers={"Content-Type": "application/json"})


def npz_zip_response(layer_arrays: dict[str, np.ndarray]) -> httpx.Response:
    return httpx.Response(
        200,
        content=make_npz_bytes(layer_arrays),
        headers={"Content-Type": "application/zip"},
    )


def uniform_logits(seq_len: int) -> np.ndarray:
    """All-zero logits -> uniform distribution over the 512 vocab."""
    return np.zeros((seq_len, 512), dtype=np.float32)


def favor_logits(seq_len: int, base: str, logit: float = 2.0) -> np.ndarray:
    """Logits that favour one base at every position."""
    arr = np.zeros((seq_len, 512), dtype=np.float32)
    arr[:, NUC_INDEX[base]] = logit
    return arr


class FakeEvo2Client:
    """In-memory stand-in for Evo2Client: returns a fixed NPZ response."""

    def __init__(self, layer_arrays: dict[str, np.ndarray]):
        self.layer_arrays = layer_arrays
        self.calls: list[tuple[str, list[str]]] = []
        self._logits_layer = "output_layer"

    @property
    def logits_layer(self) -> str:
        return self._logits_layer

    async def forward(self, *, sequence: str, output_layers: list[str]):
        self.calls.append((sequence, list(output_layers)))
        return make_npz_bytes(self.layer_arrays), {"elapsed_ms": 5}

    async def forward_logits(self, *, sequence: str):
        return await self.forward(sequence=sequence, output_layers=[self._logits_layer])
