"""OPTIONAL live tests against the real NVIDIA Evo2-7B Forward API.

These tests are skipped unless BOTH are set:
  * NVIDIA_API_KEY      (the real key)
  * EVO2_MCP_RUN_LIVE=1 (explicit opt-in, so a present key alone never
                         triggers paid API traffic)

Run with, e.g.:
    NVIDIA_API_KEY=nvapi-xxx EVO2_MCP_RUN_LIVE=1 \
        pixi run python -m pytest tests/test_live_api.py -s
"""

from __future__ import annotations

import os

import pytest

from evo2_mcp.api_client import Evo2Client
from evo2_mcp.config import Settings
from evo2_mcp.forward_output import decode_npz, find_logits, layer_summary, score_from_logits
from evo2_mcp.tools import evo2_score, evo2_variant_score

pytestmark = pytest.mark.skipif(
    not (os.environ.get("NVIDIA_API_KEY") and os.environ.get("EVO2_MCP_RUN_LIVE") == "1"),
    reason="live tests need NVIDIA_API_KEY + EVO2_MCP_RUN_LIVE=1",
)


def _live_settings(tmp_path) -> Settings:
    return Settings(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("EVO2_MCP_BASE_URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-7b"),
        timeout=float(os.environ.get("EVO2_MCP_TIMEOUT", "300")),
        max_retries=2,
        max_concurrency=1,
        allowed_dirs=(),
        output_dir=tmp_path / "output",
        allow_ambiguous=False,
        max_sequence_length=1_000_000,
        raw_inline_max=4096,
        max_per_position=5000,
        logits_layer=os.environ.get("EVO2_MCP_LOGITS_LAYER", "auto"),
    )


@pytest.mark.asyncio
async def test_live_forward_parse_npz(tmp_path):
    """Real round-trip: forward -> decode NPZ -> extract logits.

    Confirms the actual response format against the running API: layer keys,
    shapes and dtype, and that the logits seq_len matches the input length
    (required for likelihood alignment).
    """
    seq = "ACGTACGTACGTACGT"
    settings = _live_settings(tmp_path)
    async with Evo2Client(settings) as client:
        npz_bytes, meta = await client.forward_logits(sequence=seq)
        detected_layer = client.logits_layer

    layers = decode_npz(npz_bytes)
    print("\nresolved logits layer:", detected_layer)
    print("npz keys:", list(layers))
    for name, layer in layers.items():
        print(f"  {name}: shape={layer.shape} dtype={layer.dtype} "
              f"min={layer.array.min():.4f} max={layer.array.max():.4f}")
    print("api meta:", meta)

    # The logits must exist and have vocab dim 512.
    logits = find_logits(layers, preferred_name=detected_layer)
    assert logits.shape[-1] == 512
    assert logits.shape[0] == len(seq), (
        f"seq_len mismatch ({logits.shape[0]} vs {len(seq)}) — the API is "
        "padding/BOS-ing; likelihood alignment would be aborted."
    )
    assert logits.shape[0] >= 1
    # logits should be finite
    import numpy as np
    assert np.isfinite(logits).all()
    print("extracted logits shape:", logits.shape)


@pytest.mark.asyncio
async def test_live_score(tmp_path):
    settings = _live_settings(tmp_path)
    result = await evo2_score(sequence="ACGTACGTACGTACGTACGT", settings=settings)
    assert result["likelihood_computed"] is True, result
    assert result["scored_positions"] == 19
    assert result["total_log_likelihood"] < 0  # log-probabilities are <= 0
    print("\nscore:", {k: v for k, v in result.items() if k not in ("method_notes",)})


@pytest.mark.asyncio
async def test_live_variant_score(tmp_path):
    settings = _live_settings(tmp_path)
    result = await evo2_variant_score(
        sequence="ACGTACGTACGTACGTACGT",
        position=10,
        ref="C",
        alt="G",
        settings=settings,
    )
    assert "delta_log_likelihood" in result
    print("\nvariant:", {k: v for k, v in result.items() if k not in ("method_notes", "disclaimer")})


@pytest.mark.asyncio
async def test_live_score_matches_direct_logits(tmp_path):
    """Cross-check: evo2_score result == manual score from the raw logits."""
    seq = "ACGTACGTACGTACGTACGT"
    settings = _live_settings(tmp_path)

    async with Evo2Client(settings) as client:
        npz_bytes, _ = await client.forward_logits(sequence=seq)
        logits = find_logits(decode_npz(npz_bytes), preferred_name=client.logits_layer)
    direct = score_from_logits(logits, seq, include_per_position=False)

    result = await evo2_score(sequence=seq, settings=settings)
    assert result["total_log_likelihood"] == pytest.approx(direct.total_log_likelihood)
    assert result["mean_log_likelihood"] == pytest.approx(direct.mean_log_likelihood)
    print("\ncross-check total:", direct.total_log_likelihood,
          "mean:", direct.mean_log_likelihood, "OK")
