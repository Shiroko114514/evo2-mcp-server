"""Unit tests for scripts/score_fasta.py pure helpers (no API calls)."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "score_fasta.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("score_fasta_mod", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def test_pool_embedding_mean():
    # (1, 3, 4): mean over positions
    arr = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    pooled = mod.pool_embedding(arr, method="mean")
    assert pooled.shape == (4,)
    assert pooled.dtype == np.float32
    np.testing.assert_allclose(pooled, arr.mean(axis=(0, 1)))


def test_pool_embedding_2d():
    arr = np.zeros((5, 8), dtype=np.float64)
    pooled = mod.pool_embedding(arr)
    assert pooled.shape == (8,)
    np.testing.assert_allclose(pooled, np.zeros(8, dtype=np.float32))


def test_find_embedding_array_dot_output_key():
    # hosted API returns '<layer>.output' keys
    decoded = {"norm.output": type("L", (), {"array": np.zeros((1, 3, 4096))})()}
    key, arr = mod.find_embedding_array(decoded, "norm")
    assert key == "norm.output"
    assert arr.shape == (1, 3, 4096)


def test_find_embedding_array_bare_key():
    decoded = {"norm": type("L", (), {"array": np.zeros((3, 4096))})()}
    key, _ = mod.find_embedding_array(decoded, "norm")
    assert key == "norm"


def test_find_embedding_array_missing():
    with pytest.raises(KeyError):
        mod.find_embedding_array({}, "norm")


def test_sanitize_key():
    assert mod.sanitize_key("cis/enhancers::K562_TE-629") == "cis_enhancers__K562_TE_629"
    assert mod.sanitize_key("K562_MPT_6842") == "K562_MPT_6842"


def test_source_label(tmp_path):
    p = tmp_path / "cis" / "enhancers.fa"
    assert mod.source_label(p) == "cis/enhancers"
