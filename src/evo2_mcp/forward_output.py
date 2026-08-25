"""Decoding of Evo2 forward output and log-likelihood scoring.

Authoritative references (verified 2026-08-24)
-----------------------------------------------
1. NVIDIA NIM Evo2 endpoints doc + hosted OpenAPI schema:
   https://docs.nvidia.com/nim/bionemo/evo2/latest/endpoints.html
   * Response `data` = base64-encoded NPZ archive; `output_layer` is the
     final logits of shape `[seq_len, batch_size, 512]` (512 = padded
     vocabulary size).
   * Logit-index → DNA-base mapping (ASCII byte tokenizer):
       A: 65,  C: 67,  T: 84,  G: 71.
2. Arc Institute reference `logits_to_logprobs`:
   https://github.com/ArcInstitute/evo2/blob/main/evo2/scoring.py
   * `softmax_logprobs = log_softmax(logits, dim=-1)` over the full
     512-vocab axis, then the causal-LM shift:
          softmax_logprobs = softmax_logprobs[:, :-1]
          input_ids        = input_ids[:, 1:]
   * Position 0 of the sequence has **no** likelihood under the model
     (no context to predict it from); a length-N sequence yields N-1
     scored positions — the mean is taken over those N-1 values.
3. Arc/vortex `CharLevelTokenizer(512)`:
   tokens are UTF-8 bytes (`np.frombuffer(text.encode(), np.uint8)`);
   eod_id = 0, pad_id = 1; no BOS is prepended by default in
   `score_sequences`.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger("evo2_mcp.forward_output")


# Byte-tokenizer index of each nucleotide in the 512-dim logits axis.
# 'N' is only reachable when the server is started with
# EVO2_MCP_ALLOW_AMBIGUOUS=1 (see sequence.py).
NUC_TO_INDEX: dict[str, int] = {"A": 65, "C": 67, "G": 71, "T": 84, "N": 78}
INDEX_TO_NUC: dict[int, str] = {v: k for k, v in NUC_TO_INDEX.items()}
VOCAB_SIZE = 512


class ForwardDecodeError(RuntimeError):
    """Raised when the NPZ response cannot be decoded as expected."""


@dataclass
class LayerArray:
    name: str
    array: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def dtype(self) -> str:
        return str(self.array.dtype)


def decode_npz(npz_bytes: bytes) -> dict[str, LayerArray]:
    """Load an NPZ archive from raw bytes into `{layer_name: LayerArray}`."""
    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
            return {name: LayerArray(name=name, array=npz[name]) for name in npz.files}
    except Exception as e:
        raise ForwardDecodeError(
            f"Could not read Evo2 forward response as an NPZ archive ({len(npz_bytes)} bytes): {e}"
        )


def layer_summary(layer: LayerArray) -> dict:
    """Cheap statistics safe to return to an agent even for huge tensors."""
    a = layer.array
    flat = a.reshape(-1)
    if flat.size == 0:
        return {"name": layer.name, "shape": list(layer.shape), "dtype": layer.dtype, "size": 0}
    return {
        "name": layer.name,
        "shape": list(layer.shape),
        "dtype": layer.dtype,
        "size": int(flat.size),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


# ---------------------------------------------------------------------------
# Likelihood scoring
# ---------------------------------------------------------------------------


def _as_seq_vocab(arr: np.ndarray, name: str) -> np.ndarray:
    """Squeeze a logits array into `[seq_len, 512]` form.

    The NVIDIA doc specifies `[seq_len, batch_size, 512]`; Arc's reference
    PyTorch model returns `(batch, seq_len, vocab)`; the hosted API returns
    `(1, seq_len, 512)` (verified live). We accept all orderings with a
    unit batch dimension.
    """
    if arr.ndim == 2 and arr.shape[-1] == VOCAB_SIZE:
        return arr
    if arr.ndim != 3 or arr.shape[-1] != VOCAB_SIZE:
        raise ForwardDecodeError(
            f"layer {name!r} has unexpected shape {arr.shape}; expected "
            f"[seq_len, batch, {VOCAB_SIZE}] or [batch, seq_len, {VOCAB_SIZE}]."
        )
    d0, d1, _ = arr.shape
    if d0 == 1 and d1 > 1:
        # [batch=1, seq_len>1, 512]
        return arr[0]
    if d1 == 1:
        # [seq_len, batch=1, 512] -- the documented hosted shape.
        return arr[:, 0, :]
    logger.warning(
        "layer %r shape %s is ambiguous; assuming [seq_len, batch, 512].",
        name, arr.shape,
    )
    return arr[:, 0, :]


def find_logits(layers: dict[str, LayerArray], preferred_name: str | None = None) -> np.ndarray:
    """Locate the final-logits tensor and return it in `[seq_len, 512]` form.

    The NPZ key may be the bare layer name (`output_layer` on self-hosted
    NIM) or `<name>.output` (`unembed.output` on the hosted API — verified
    live). We first match the preferred name exactly (with or without the
    `.output` suffix), then fall back to a vocab-dimension heuristic.
    """
    if preferred_name:
        exact = [n for n in layers if n in (preferred_name, f"{preferred_name}.output")]
        if exact:
            return _as_seq_vocab(layers[exact[0]].array, exact[0])

    vocab = [
        (n, l) for n, l in layers.items()
        if l.array.ndim in (2, 3) and l.array.shape[-1] == VOCAB_SIZE
    ]
    if len(vocab) == 1:
        name, layer = vocab[0]
        return _as_seq_vocab(layer.array, name)
    if not vocab:
        raise ForwardDecodeError(
            f"None of the returned layers {sorted(layers)} look like final "
            f"logits (no array with last dim {VOCAB_SIZE}); requested layer "
            f"was {preferred_name!r}. Request output_layers=['{preferred_name or 'unembed'}'] "
            "to get logits."
        )
    raise ForwardDecodeError(
        f"Multiple returned layers {[n for n, _ in vocab]} end in "
        f"{VOCAB_SIZE} dims; cannot unambiguously pick the final logits. "
        "Request exactly one logits layer."
    )


def _extract_output_layer_logits(layers: dict[str, LayerArray]) -> np.ndarray:
    """Backwards-compatible wrapper: extract logits preferring 'output_layer'."""
    return find_logits(layers, preferred_name="output_layer")


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (x - m) - np.log(e.sum(axis=axis, keepdims=True))


@dataclass
class ScoreResult:
    sequence_length: int
    total_log_likelihood: float
    mean_log_likelihood: float
    scored_positions: int
    per_position_log_likelihood: list[float]
    notes: str


def score_from_logits(
    logits_seq_by_vocab: np.ndarray,
    sequence: str,
    *,
    include_per_position: bool = True,
) -> ScoreResult:
    """Compute per-position log-likelihoods from a `[seq_len, 512]` logits array.

    Follows Arc's `logits_to_logprobs` exactly: for a length-N sequence we
    score positions 1..N-1 (0-based) using `logits[0..N-2]` (causal-LM shift,
    softmax over the full 512-token vocabulary, then gather the target byte
    index). Position 0 has no context and is not scored.

    Raises `ForwardDecodeError` instead of silently misaligning when the API
    returns a different seq_len than the submitted sequence (e.g. server-side
    BOS/padding).
    """
    if logits_seq_by_vocab.ndim != 2 or logits_seq_by_vocab.shape[1] != VOCAB_SIZE:
        raise ForwardDecodeError(
            f"Expected logits of shape [seq_len, {VOCAB_SIZE}], got {logits_seq_by_vocab.shape}"
        )
    n = len(sequence)
    if logits_seq_by_vocab.shape[0] != n:
        raise ForwardDecodeError(
            f"logits seq_len ({logits_seq_by_vocab.shape[0]}) does not match "
            f"sequence length ({n}). The API may have applied BOS/padding; "
            "aborting likelihood computation to avoid silent misalignment."
        )
    if n < 2:
        raise ForwardDecodeError(
            "Sequence must have length >= 2 for causal-LM likelihood (position 0 is unscored)."
        )

    # Causal shift: use logits[0..N-2] to predict tokens 1..N-1.
    logits_used = logits_seq_by_vocab[:-1]                # (N-1, 512)
    target_ids = np.fromiter(
        (NUC_TO_INDEX[c] for c in sequence[1:]),
        dtype=np.int64,
        count=n - 1,
    )
    logp = _log_softmax(logits_used, axis=-1)             # (N-1, 512)
    per_pos = logp[np.arange(n - 1), target_ids]          # (N-1,)

    per_pos_list = per_pos.tolist() if include_per_position else []
    return ScoreResult(
        sequence_length=n,
        total_log_likelihood=float(per_pos.sum()),
        mean_log_likelihood=float(per_pos.mean()),
        scored_positions=int(per_pos.size),
        per_position_log_likelihood=per_pos_list,
        notes=(
            "Log-likelihoods computed with the Arc Institute reference causal "
            "shift (logits[i] predicts the byte at position i+1 over the full "
            "512-token vocabulary; log_softmax then gather). Position 0 has no "
            "context and is not scored, so scored_positions = sequence_length - 1. "
            "per_position_log_likelihood[k] corresponds to 0-based position k+1 "
            "(1-based position k+2) of the sequence. Tokenizer: byte-level, "
            "A=65 C=67 T=84 G=71."
        ),
    )
