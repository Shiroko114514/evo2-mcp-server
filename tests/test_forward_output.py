"""Tests for NPZ decoding, logits extraction and likelihood math.

The likelihood reference is computed independently here (numpy log_softmax +
causal shift), mirroring ArcInstitute/evo2 `logits_to_logprobs`:
  * softmax_logprobs = log_softmax(logits, dim=-1)
  * softmax_logprobs = softmax_logprobs[:, :-1]; input_ids = input_ids[:, 1:]
  * gather target byte index (A=65, C=67, T=84, G=71)
"""

from __future__ import annotations

import numpy as np
import pytest

from evo2_mcp.forward_output import (
    VOCAB_SIZE,
    ForwardDecodeError,
    _extract_output_layer_logits,
    _log_softmax,
    decode_npz,
    find_logits,
    layer_summary,
    score_from_logits,
)

from conftest import favor_logits, make_npz_bytes, uniform_logits

SEQUENCE = "ACGTACGTAC"


def reference_score(logits: np.ndarray, sequence: str) -> tuple[float, float, list[float]]:
    """Independent numpy re-implementation of the Arc scoring semantics."""
    logp = _log_softmax(logits[:-1], axis=-1)  # causal shift: predict i+1
    target = np.array([{"A": 65, "C": 67, "G": 71, "T": 84}[c] for c in sequence[1:]])
    per_pos = logp[np.arange(len(target)), target]
    return float(per_pos.sum()), float(per_pos.mean()), per_pos.tolist()


def test_decode_npz_roundtrip():
    arr = uniform_logits(6)
    npz = make_npz_bytes({"output_layer": arr, "decoder.layers.0.mlp": arr.copy()})
    layers = decode_npz(npz)
    assert set(layers) == {"output_layer", "decoder.layers.0.mlp"}
    assert layers["output_layer"].shape == (6, 512)
    assert layers["output_layer"].dtype == "float32"


def test_decode_npz_garbage():
    with pytest.raises(ForwardDecodeError):
        decode_npz(b"this is not an npz archive")


def test_layer_summary():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    from evo2_mcp.forward_output import LayerArray
    s = layer_summary(LayerArray("x", arr))
    assert s["shape"] == [3, 4]
    assert s["dtype"] == "float32"
    assert s["min"] == 0.0 and s["max"] == 11.0 and s["mean"] == 5.5


def test_extract_logits_documented_order():
    # [seq_len, batch=1, 512] — the NVIDIA-documented order.
    arr = uniform_logits(7).reshape(7, 1, 512)
    out = _extract_output_layer_logits({"output_layer": type("L", (), {"array": arr})()})
    assert out.shape == (7, 512)


def test_extract_logits_batch_first_order():
    # [batch=1, seq_len, 512] — Arc reference order, accepted defensively.
    arr = uniform_logits(7).reshape(1, 7, 512)
    out = _extract_output_layer_logits({"output_layer": type("L", (), {"array": arr})()})
    assert out.shape == (7, 512)


def test_extract_logits_missing_layer():
    # realistic non-logits layer (embedding is 4096-wide on Evo2 7B)
    arr = np.zeros((4, 4096), dtype=np.float32)
    with pytest.raises(ForwardDecodeError, match="look like final logits"):
        _extract_output_layer_logits({"embedding": type("L", (), {"array": arr})()})


def test_extract_logits_wrong_vocab_dim():
    arr = np.zeros((4, 256), dtype=np.float32)
    with pytest.raises(ForwardDecodeError, match="unexpected shape"):
        _extract_output_layer_logits({"output_layer": type("L", (), {"array": arr})()})


def test_find_logits_hosted_dot_output_key():
    # The hosted API returns NPZ keys like 'unembed.output' (verified live).
    arr = np.zeros((1, 6, 512), dtype=np.float64)
    out = find_logits({"unembed.output": type("L", (), {"array": arr})()}, preferred_name="unembed")
    assert out.shape == (6, 512)


def test_find_logits_fallback_heuristic():
    # Without a preferred name, a unique 512-dim array is still found.
    arr = np.zeros((5, 512), dtype=np.float32)
    out = find_logits({"mystery_layer": type("L", (), {"array": arr})()})
    assert out.shape == (5, 512)


def test_score_uniform_logits_matches_reference():
    logits = uniform_logits(len(SEQUENCE))
    ref_total, ref_mean, ref_per = reference_score(logits, SEQUENCE)
    score = score_from_logits(logits, SEQUENCE, include_per_position=True)
    assert score.scored_positions == len(SEQUENCE) - 1  # position 0 unscored
    assert score.sequence_length == len(SEQUENCE)
    assert score.total_log_likelihood == pytest.approx(ref_total)
    assert score.mean_log_likelihood == pytest.approx(ref_mean)
    assert score.per_position_log_likelihood == pytest.approx(ref_per)
    # uniform 512-vocab -> -log(512) per position
    assert score.mean_log_likelihood == pytest.approx(-np.log(512.0))


def test_score_favor_logits_matches_reference():
    logits = favor_logits(len(SEQUENCE), base="A", logit=3.0)
    ref_total, ref_mean, ref_per = reference_score(logits, SEQUENCE)
    score = score_from_logits(logits, SEQUENCE, include_per_position=False)
    assert score.total_log_likelihood == pytest.approx(ref_total)
    assert score.mean_log_likelihood == pytest.approx(ref_mean)
    assert score.per_position_log_likelihood == []


def test_score_misalignment_aborts():
    # seq_len mismatch must abort rather than silently misalign.
    logits = uniform_logits(len(SEQUENCE) + 3)
    with pytest.raises(ForwardDecodeError, match="does not match"):
        score_from_logits(logits, SEQUENCE)


def test_score_too_short():
    with pytest.raises(ForwardDecodeError):
        score_from_logits(uniform_logits(1), "A")


def test_score_notes_document_semantics():
    score = score_from_logits(uniform_logits(4), "ACGT", include_per_position=True)
    assert "causal shift" in score.notes
    assert "Position 0" in score.notes
    assert "65" in score.notes  # documents the byte-index mapping
