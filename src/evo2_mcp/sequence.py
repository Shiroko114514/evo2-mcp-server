"""DNA sequence validation and normalisation.

Verified behaviour (checked against official sources on 2026-08-24):
* NVIDIA NIM for Evo 2 — endpoints doc and hosted OpenAPI schema
  (https://docs.nvidia.com/nim/bionemo/evo2/latest/endpoints.html):
  the forward endpoint accepts a `sequence` string; the model's tokenizer is
  byte-level with a padded vocabulary of 512; the logit-index → DNA-base
  mapping is A: ASCII 65, C: 67, T: 84, G: 71, and only A/C/G/T are
  meaningful outputs.
* Arc Institute reference tokenizer (`CharLevelTokenizer`, vortex package):
  ``np.frombuffer(text.encode("utf-8"), dtype=np.uint8)`` — i.e. one token
  per UTF-8 byte. Ambiguous bytes such as ``N`` (78) are encodable but the
  pretrained model was not evaluated on them.

This module therefore:
* upper-cases lowercase input,
* strips all ASCII whitespace (covers FASTA-wrapped lines),
* rejects any character outside {A,C,G,T} with a clear error by default —
  nothing is silently dropped,
* optionally (``allow_ambiguous=True``) passes ``N`` through unchanged, with
  a documented caveat; all other IUPAC codes (R, Y, S, W, K, M, B, D, H, V)
  are always rejected.
* rejects empty sequences and warns on very long ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ALLOWED = frozenset("ACGT")
_AMBIGUOUS_ONLY = frozenset("N")
# IUPAC ambiguity codes that are never accepted (byte-encodable but
# meaningless to the pretrained model).
_REJECTED_ALWAYS = frozenset("RYSWKMBDHVU")
_WHITESPACE_RE = re.compile(r"\s+")

# Warn (via return value) above this length. The hosted API supports long
# genomic sequences but latency and response size grow linearly (each layer
# is seq_len × 512 float32 ≈ 2 KiB / bp).
LONG_SEQUENCE_WARN_THRESHOLD = 8_192


class SequenceValidationError(ValueError):
    """Raised when a DNA sequence contains unsupported characters."""


@dataclass(frozen=True)
class NormalisedSequence:
    sequence: str
    length: int
    warning: str | None


def normalise_sequence(
    seq: str,
    *,
    name: str = "sequence",
    allow_ambiguous: bool = False,
) -> NormalisedSequence:
    """Validate + upper-case + strip whitespace.

    Raises `SequenceValidationError` for unsupported characters.
    Returns a `NormalisedSequence` carrying an optional soft warning for
    very long inputs (and for ambiguous bases when allowed).
    """
    if not isinstance(seq, str):
        raise SequenceValidationError(f"{name} must be a string, got {type(seq).__name__}")

    cleaned = _WHITESPACE_RE.sub("", seq).upper()
    if not cleaned:
        raise SequenceValidationError(f"{name} is empty after stripping whitespace")

    allowed = _ALLOWED | (_AMBIGUOUS_ONLY if allow_ambiguous else frozenset())
    bad = sorted({c for c in cleaned if c not in allowed})
    if bad:
        first_idx = next(i for i, c in enumerate(cleaned) if c not in allowed)
        hint = (
            "Pass N only if the server is started with EVO2_MCP_ALLOW_AMBIGUOUS=1 "
            "(Evo2 was not evaluated on ambiguous bases; scores at N positions "
            "reflect byte-level behaviour of the pretrained model)."
            if bad == ["N"]
            else (
                "IUPAC ambiguity codes other than N are never accepted; "
                "Evo2's byte-level tokenizer has no meaningful output for them."
            )
        )
        raise SequenceValidationError(
            f"{name} contains character(s) not in {sorted(allowed)!r}: {bad!r} "
            f"(first at 0-indexed position {first_idx}). {hint}"
        )

    warnings: list[str] = []
    if len(cleaned) > LONG_SEQUENCE_WARN_THRESHOLD:
        warnings.append(
            f"{name} has {len(cleaned):,} bp. Evo2 forward output is roughly "
            f"{len(cleaned) * 512 * 4 / (1024 ** 2):.1f} MiB per layer per request "
            "(seq_len × 512 float32 logits); expect long latency — prefer "
            "mode='summary' or batch scoring over raw inline output."
        )
    if allow_ambiguous and any(c == "N" for c in cleaned):
        warnings.append(
            f"{name} contains N (ambiguous). The pretrained Evo2 model was not "
            "evaluated on ambiguous bases; likelihoods at N positions reflect "
            "the model's byte-level behaviour and should be interpreted with care."
        )

    return NormalisedSequence(
        sequence=cleaned,
        length=len(cleaned),
        warning=" ".join(warnings) if warnings else None,
    )


def preview(seq: str, head: int = 8, tail: int = 4) -> str:
    """Return a short redacted preview safe to include in logs / errors.

    Never puts the full sequence anywhere it might be persisted.
    """
    if len(seq) <= head + tail + 3:
        return f"{seq} (len={len(seq)})"
    return f"{seq[:head]}...{seq[-tail:]} (len={len(seq)})"
