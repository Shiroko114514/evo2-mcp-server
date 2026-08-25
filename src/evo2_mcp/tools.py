"""High-level Evo2 tool implementations.

These are the pure-Python entry points wrapped by MCP tools. They accept
JSON-friendly dicts, call the API client, and return JSON-friendly dicts.

Design rules:
* Never fabricate a likelihood — if the API output cannot be aligned to the
  sequence, return the raw layer statistics plus an explicit "not supported
  until verified" message.
* Never dump huge tensors into the MCP response by default; use
  mode='summary' / mode='save' and cap inline raw output.
* The WT forward pass is computed once per batch and reused for all variants.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evo2_mcp.api_client import Evo2APIError, Evo2Client
from evo2_mcp.config import Settings
from evo2_mcp.fasta import FastaError, FastaRecord, parse_fasta, read_fasta_path
from evo2_mcp.forward_output import (
    NUC_TO_INDEX,
    ForwardDecodeError,
    LayerArray,
    ScoreResult,
    decode_npz,
    find_logits,
    layer_summary,
    score_from_logits,
)
from evo2_mcp.sequence import (
    NormalisedSequence,
    SequenceValidationError,
    normalise_sequence,
    preview,
)

logger = logging.getLogger("evo2_mcp.tools")

DEFAULT_OUTPUT_LAYERS = ["output_layer"]
MAX_FASTA_TEXT_CHARS = 10_000_000
MAX_FASTA_RECORDS = 2_000

_DISCLAIMER = (
    "Evo2 is a DNA foundation model. The scores it returns are model-based "
    "sequence likelihoods and MUST NOT be interpreted as clinical "
    "pathogenicity labels. A negative delta_log_likelihood only means the "
    "mutant sequence is less likely under Evo2; it is not a diagnosis."
)


# ---------------------------------------------------------------------------
# Tool 1: evo2_forward
# ---------------------------------------------------------------------------

async def evo2_forward(
    *,
    sequence: str,
    output_layers: Iterable[str] | None = None,
    mode: str = "summary",
    save_path: str | None = None,
    settings: Settings,
    client: Evo2Client | None = None,
) -> dict[str, Any]:
    """Run the Evo2-7B forward pass on a DNA sequence.

    mode:
      * "summary" (default) - per-layer shape/dtype/min/max/mean/std stats
        (context-safe even for very long sequences).
      * "save"  - write the raw NPZ to disk (output dir) and return the path.
      * "raw"   - inline the full tensors as nested lists, only allowed while
        every requested layer stays under `settings.raw_inline_max` values.
    """
    norm = _normalise_for_tools(sequence, settings)
    layers = list(output_layers) if output_layers else list(DEFAULT_OUTPUT_LAYERS)
    if not layers:
        raise ValueError("output_layers must be a non-empty list (NVIDIA API requires >= 1).")

    npz_bytes, api_meta = await _do_forward(norm.sequence, layers, settings, client)
    decoded = decode_npz(npz_bytes)
    summaries = [layer_summary(l) for l in decoded.values()]

    result: dict[str, Any] = {
        "sequence_preview": preview(norm.sequence),
        "sequence_length": norm.length,
        "requested_output_layers": layers,
        "returned_layers": list(decoded.keys()),
        "layer_stats": summaries,
        "api": api_meta,
        "warnings": [norm.warning] if norm.warning else [],
    }

    if mode == "save" or save_path is not None:
        path = _save_npz(npz_bytes, settings, save_path)
        result["saved"] = True
        result["path"] = str(path)
        result["bytes_on_disk"] = path.stat().st_size
        return result

    if mode == "summary":
        return result

    if mode == "raw":
        total = sum(int(np.prod(l.array.shape)) for l in decoded.values())
        if total > settings.raw_inline_max:
            raise ValueError(
                f"Refusing to inline {total:,} tensor values in the MCP response "
                f"(cap: {settings.raw_inline_max:,}). Re-run with mode='save' or "
                "provide save_path to write an .npz file, or use mode='summary'."
            )
        result["arrays"] = {
            name: layer.array.astype(float).tolist() for name, layer in decoded.items()
        }
        return result

    raise ValueError(f"Unknown mode {mode!r}; expected 'summary', 'raw', or 'save'.")


# ---------------------------------------------------------------------------
# Tool 2: evo2_score
# ---------------------------------------------------------------------------

async def evo2_score(
    *,
    sequence: str,
    include_per_position: bool = False,
    settings: Settings,
    client: Evo2Client | None = None,
) -> dict[str, Any]:
    """Compute mean/total log-likelihood of `sequence` under Evo2-7B.

    Falls back to a raw-forward summary if we cannot align logits to the
    sequence (never fabricates a likelihood).
    """
    norm = _normalise_for_tools(sequence, settings)
    if norm.length < 2:
        raise SequenceValidationError(
            "sequence must be at least 2 bp long for a causal likelihood "
            "(position 0 has no context and is not scored)."
        )
    npz_bytes, api_meta = await _do_logits_forward(norm.sequence, settings, client)
    decoded = decode_npz(npz_bytes)
    return _score_result_dict(
        norm, decoded, api_meta, include_per_position, settings,
        preferred_layer=_logits_layer_name(client),
    )


# ---------------------------------------------------------------------------
# Tool 3: evo2_variant_score
# ---------------------------------------------------------------------------

async def evo2_variant_score(
    *,
    sequence: str,
    position: int,
    ref: str,
    alt: str,
    coordinate: str = "1-based",
    include_per_position: bool = False,
    settings: Settings,
    client: Evo2Client | None = None,
) -> dict[str, Any]:
    """Score a single-nucleotide variant.

    coordinate:
      * "1-based" (default, VCF-style): position `p` refers to sequence[p-1].
      * "0-based": position `p` refers to sequence[p].

    The first position (0-based 0) cannot be scored by a causal LM and is
    rejected with an explicit error.
    """
    wt = _normalise_for_tools(sequence, settings)
    zero_based = _to_zero_based(position, coordinate, wt.length)
    ref_u = _validate_allele(ref, "ref")
    alt_u = _validate_allele(alt, "alt")
    actual_ref = wt.sequence[zero_based]
    if actual_ref != ref_u:
        raise SequenceValidationError(
            f"ref allele mismatch: position {position} ({coordinate}) is "
            f"'{actual_ref}' in the sequence, but ref='{ref_u}' was requested."
        )
    if ref_u == alt_u:
        raise SequenceValidationError(
            f"ref and alt are identical ('{ref_u}'); this is not a variant."
        )

    mut_seq = wt.sequence[:zero_based] + alt_u + wt.sequence[zero_based + 1 :]
    assert len(mut_seq) == wt.length
    mut = _normalise_for_tools(mut_seq, settings)

    async def _score_one(s: NormalisedSequence) -> ScoreResult:
        npz_bytes, _ = await _do_logits_forward(s.sequence, settings, client)
        decoded = decode_npz(npz_bytes)
        logits = find_logits(decoded, preferred_name=_logits_layer_name(client))
        return score_from_logits(logits, s.sequence, include_per_position=include_per_position)

    wt_score, mut_score = await asyncio.gather(_score_one(wt), _score_one(mut))
    delta = mut_score.total_log_likelihood - wt_score.total_log_likelihood
    delta_mean = mut_score.mean_log_likelihood - wt_score.mean_log_likelihood

    return {
        "position": position,
        "coordinate": coordinate,
        "ref": ref_u,
        "alt": alt_u,
        "sequence_length": wt.length,
        "sequence_preview": preview(wt.sequence),
        "wildtype_log_likelihood": wt_score.total_log_likelihood,
        "wildtype_mean_log_likelihood": wt_score.mean_log_likelihood,
        "mutant_log_likelihood": mut_score.total_log_likelihood,
        "mutant_mean_log_likelihood": mut_score.mean_log_likelihood,
        "delta_log_likelihood": delta,
        "delta_mean_log_likelihood": delta_mean,
        "scored_positions": wt_score.scored_positions,
        "interpretation": _interpret_delta(delta),
        "warnings": [w for w in (wt.warning, mut.warning) if w],
        "wildtype_per_position_log_likelihood": (
            _bounded_per_position(wt_score.per_position_log_likelihood, settings.max_per_position)
            if include_per_position else None
        ),
        "mutant_per_position_log_likelihood": (
            _bounded_per_position(mut_score.per_position_log_likelihood, settings.max_per_position)
            if include_per_position else None
        ),
        "method_notes": wt_score.notes,
        "disclaimer": _DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Tool 4: evo2_batch_score
# ---------------------------------------------------------------------------

async def evo2_batch_score(
    *,
    sequence: str,
    variants: list[dict[str, Any]],
    coordinate: str = "1-based",
    settings: Settings,
    client: Evo2Client | None = None,
) -> dict[str, Any]:
    """Score many single-nucleotide variants against one WT sequence.

    * The WT forward pass is computed exactly once and reused.
    * Identical (position, alt) mutants are forwarded once and memoised.
    * Mutant requests are bounded by `settings.max_concurrency`; per-variant
      API failures are reported per-variant instead of failing the batch.
    * All validation happens before any API call (fail fast, no wasted
      requests).
    """
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants must be a non-empty list of {position, ref, alt} dicts")

    wt = _normalise_for_tools(sequence, settings)
    if wt.length < 2:
        raise SequenceValidationError("sequence must be at least 2 bp long")

    # --- Validate everything up-front (no API calls yet). ---
    prepared: list[dict[str, Any]] = []
    for i, v in enumerate(variants):
        try:
            pos = int(v["position"])
            ref = _validate_allele(str(v["ref"]), f"variants[{i}].ref")
            alt = _validate_allele(str(v["alt"]), f"variants[{i}].alt")
        except KeyError as e:
            raise ValueError(f"variants[{i}] is missing field {e}") from None
        zb = _to_zero_based(pos, coordinate, wt.length)
        actual = wt.sequence[zb]
        if actual != ref:
            raise SequenceValidationError(
                f"variants[{i}] ref mismatch at position {pos} ({coordinate}): "
                f"sequence has '{actual}', ref='{ref}'"
            )
        if ref == alt:
            raise SequenceValidationError(f"variants[{i}] ref == alt ('{ref}'), not a variant")
        mut_seq = wt.sequence[:zb] + alt + wt.sequence[zb + 1 :]
        prepared.append({
            "position": pos,
            "ref": ref,
            "alt": alt,
            "coordinate": coordinate,
            "_zb": zb,
            "_mut_seq": mut_seq,
        })

    async def _score_seq(s: NormalisedSequence) -> ScoreResult:
        npz_bytes, _ = await _do_logits_forward(s.sequence, settings, client)
        decoded = decode_npz(npz_bytes)
        logits = find_logits(decoded, preferred_name=_logits_layer_name(client))
        return score_from_logits(logits, s.sequence, include_per_position=False)

    # Deduplicate mutants so the same (position, alt) is forwarded exactly once.
    unique_mutants: dict[tuple[int, str], str] = {}
    for meta in prepared:
        unique_mutants.setdefault((meta["_zb"], meta["alt"]), meta["_mut_seq"])

    # --- Fire the single WT request. ---
    wt_task = asyncio.create_task(_score_seq(wt))

    sem = asyncio.Semaphore(max(1, settings.max_concurrency))

    async def _score_mutant(key: tuple[int, str]) -> tuple[tuple[int, str], Any]:
        norm = normalise_sequence(unique_mutants[key], allow_ambiguous=settings.allow_ambiguous)
        try:
            async with sem:  # bounds in-flight NVIDIA requests
                return key, await _score_seq(norm)
        except (Evo2APIError, ForwardDecodeError, SequenceValidationError) as e:
            return key, e  # reported per-variant; does not kill the batch

    wt_score = await wt_task
    mutant_map = dict(await asyncio.gather(*(_score_mutant(k) for k in unique_mutants)))

    results: list[dict[str, Any]] = []
    for meta in prepared:
        entry = {k: v for k, v in meta.items() if not k.startswith("_")}
        mres = mutant_map[(meta["_zb"], meta["alt"])]
        if isinstance(mres, (Evo2APIError, ForwardDecodeError, SequenceValidationError)):
            entry.update({"error": str(mres)})
            if isinstance(mres, Evo2APIError):
                entry["status"] = mres.status
        else:
            entry.update({
                "wildtype_log_likelihood": wt_score.total_log_likelihood,
                "mutant_log_likelihood": mres.total_log_likelihood,
                "delta_log_likelihood": mres.total_log_likelihood - wt_score.total_log_likelihood,
                "delta_mean_log_likelihood": mres.mean_log_likelihood - wt_score.mean_log_likelihood,
            })
        results.append(entry)

    return {
        "sequence_length": wt.length,
        "sequence_preview": preview(wt.sequence),
        "coordinate": coordinate,
        "num_variants": len(results),
        "wildtype_log_likelihood": wt_score.total_log_likelihood,
        "wildtype_mean_log_likelihood": wt_score.mean_log_likelihood,
        "scored_positions": wt_score.scored_positions,
        "variants": results,
        "method_notes": wt_score.notes,
        "disclaimer": _DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Tool 5: evo2_score_fasta
# ---------------------------------------------------------------------------

async def evo2_score_fasta(
    *,
    fasta_path: str | None = None,
    fasta_text: str | None = None,
    settings: Settings,
    client: Evo2Client | None = None,
) -> dict[str, Any]:
    """Score every record in a FASTA source under Evo2-7B.

    `fasta_path` is only honoured when the file's directory is listed in
    EVO2_MCP_ALLOWED_DIRS; inline `fasta_text` is always accepted (capped in
    size and record count). Per-record errors are reported inline rather than
    failing the whole request.
    """
    if bool(fasta_path) == bool(fasta_text):
        raise ValueError("Provide exactly one of fasta_path or fasta_text.")
    text = read_fasta_path(fasta_path, settings) if fasta_path else fasta_text  # type: ignore[arg-type]
    if len(text or "") > MAX_FASTA_TEXT_CHARS:
        raise FastaError(
            f"FASTA text too large ({len(text):,} chars; cap {MAX_FASTA_TEXT_CHARS:,}). "
            "Score shorter chunks or use a file inside EVO2_MCP_ALLOWED_DIRS."
        )
    records = parse_fasta(text or "")
    if len(records) > MAX_FASTA_RECORDS:
        raise FastaError(
            f"FASTA contains {len(records)} records; cap is {MAX_FASTA_RECORDS:,}. "
            "Split the input into smaller batches."
        )

    sem = asyncio.Semaphore(max(1, settings.max_concurrency))

    async def _score_record(rec: FastaRecord) -> dict[str, Any]:
        try:
            norm = normalise_sequence(
                rec.sequence, name=f"record[{rec.id}]", allow_ambiguous=settings.allow_ambiguous
            )
            if norm.length < 2:
                return {"id": rec.id, "header": rec.header, "error": "sequence < 2 bp"}
            if norm.length > settings.max_sequence_length:
                return {
                    "id": rec.id,
                    "header": rec.header,
                    "error": f"sequence longer than max_sequence_length ({settings.max_sequence_length:,})",
                }
            async with sem:
                npz_bytes, _ = await _do_logits_forward(norm.sequence, settings, client)
            decoded = decode_npz(npz_bytes)
            logits = find_logits(decoded, preferred_name=_logits_layer_name(client))
            score = score_from_logits(logits, norm.sequence, include_per_position=False)
            return {
                "id": rec.id,
                "header": rec.header,
                "sequence_length": norm.length,
                "total_log_likelihood": score.total_log_likelihood,
                "mean_log_likelihood": score.mean_log_likelihood,
                "scored_positions": score.scored_positions,
                "warning": norm.warning,
            }
        except (SequenceValidationError, ForwardDecodeError, Evo2APIError) as e:
            return {"id": rec.id, "header": rec.header, "error": str(e)}

    scored = await asyncio.gather(*(_score_record(r) for r in records))
    return {
        "num_records": len(records),
        "records": scored,
        "disclaimer": _DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_for_tools(sequence: str, settings: Settings) -> NormalisedSequence:
    norm = normalise_sequence(sequence, allow_ambiguous=settings.allow_ambiguous)
    if norm.length > settings.max_sequence_length:
        raise SequenceValidationError(
            f"sequence length {norm.length:,} exceeds the configured "
            f"max_sequence_length ({settings.max_sequence_length:,}). "
            "Raise EVO2_MCP_MAX_SEQUENCE_LENGTH if you really need longer input."
        )
    return norm


async def _do_forward(
    sequence: str,
    output_layers: list[str],
    settings: Settings,
    client: Evo2Client | None,
) -> tuple[bytes, dict[str, Any]]:
    if client is None:
        async with Evo2Client(settings) as c:
            return await c.forward(sequence=sequence, output_layers=output_layers)
    return await client.forward(sequence=sequence, output_layers=output_layers)


async def _do_logits_forward(
    sequence: str,
    settings: Settings,
    client: Evo2Client | None,
) -> tuple[bytes, dict[str, Any]]:
    """Forward requesting only the logits layer (auto-detects the layer name)."""
    if client is None:
        async with Evo2Client(settings) as c:
            return await c.forward_logits(sequence=sequence)
    return await client.forward_logits(sequence=sequence)


def _logits_layer_name(client: Evo2Client | None) -> str | None:
    """Resolved logits layer name if a client with a resolution is available."""
    return client.logits_layer if client is not None else None


def _score_result_dict(
    norm: NormalisedSequence,
    decoded: dict[str, LayerArray],
    api_meta: dict[str, Any],
    include_per_position: bool,
    settings: Settings,
    preferred_layer: str | None = None,
) -> dict[str, Any]:
    try:
        logits = find_logits(decoded, preferred_name=preferred_layer)
        score = score_from_logits(logits, norm.sequence, include_per_position=include_per_position)
    except ForwardDecodeError as e:
        # Never fabricate a likelihood — return raw stats and be explicit.
        return {
            "sequence_preview": preview(norm.sequence),
            "sequence_length": norm.length,
            "likelihood_computed": False,
            "reason": str(e),
            "layer_stats": [layer_summary(l) for l in decoded.values()],
            "api": api_meta,
            "message": (
                "Likelihood calculation is not supported because the API output "
                "format could not be verified against the expected "
                "[seq_len, 512] logits layout. Returning raw forward output "
                "statistics instead — do not interpret them as likelihoods."
            ),
        }
    out: dict[str, Any] = {
        "sequence_preview": preview(norm.sequence),
        "sequence_length": norm.length,
        "likelihood_computed": True,
        "total_log_likelihood": score.total_log_likelihood,
        "mean_log_likelihood": score.mean_log_likelihood,
        "scored_positions": score.scored_positions,
        "method_notes": score.notes,
        "api": api_meta,
        "disclaimer": _DISCLAIMER,
        "warnings": [norm.warning] if norm.warning else [],
    }
    if include_per_position:
        vals, truncated, total = _bounded_per_position(
            score.per_position_log_likelihood, settings.max_per_position
        )
        out["per_position_log_likelihood"] = vals
        out["per_position_truncated"] = truncated
        out["per_position_total"] = total
        out["per_position_notes"] = (
            "per_position_log_likelihood[k] corresponds to 0-based position k+1 "
            "(1-based position k+2) of the sequence; position 0 is unscored. "
            "Lists longer than max_per_position are returned as head+tail."
        )
    else:
        out["per_position_log_likelihood"] = None
        out["per_position_truncated"] = False
        out["per_position_total"] = score.scored_positions
    return out


def _bounded_per_position(values: list[float], cap: int) -> tuple[list[float], bool, int]:
    """Truncate a per-position list to `cap` values (head + tail) when needed."""
    n = len(values)
    if cap <= 0 or n <= cap:
        return values, False, n
    head_n = (cap + 1) // 2
    return values[:head_n] + values[n - (cap - head_n):], True, n


def _validate_allele(a: str, name: str) -> str:
    u = a.strip().upper()
    if len(u) != 1 or u not in NUC_TO_INDEX or u == "N":
        raise SequenceValidationError(
            f"{name} must be a single character in {{A,C,G,T}}, got {a!r}"
        )
    return u


def _to_zero_based(position: int, coordinate: str, seq_len: int) -> int:
    if coordinate not in {"0-based", "1-based"}:
        raise ValueError("coordinate must be '0-based' or '1-based'")
    zb = position if coordinate == "0-based" else position - 1
    if zb < 0 or zb >= seq_len:
        raise ValueError(
            f"position {position} ({coordinate}) is out of range for a sequence of length {seq_len}"
        )
    if zb == 0:
        raise ValueError(
            "Variants at the first sequence position (0-based index 0) cannot be "
            "scored by a causal LM — Evo2 does not assign a likelihood to the "
            "first token. Provide surrounding sequence context so the variant "
            "is at least at position 2."
        )
    return zb


def _interpret_delta(delta: float) -> str:
    if delta > 0:
        direction = "more likely"
    elif delta < 0:
        direction = "less likely"
    else:
        direction = "equally likely"
    return (
        f"The mutant sequence is {direction} than the wildtype under Evo2-7B "
        f"(Δ log-likelihood = {delta:+.4f}). This is a model-based sequence "
        "likelihood change only; it is NOT a clinical pathogenicity call."
    )


def _save_npz(npz_bytes: bytes, settings: Settings, save_path: str | None) -> Path:
    """Write NPZ bytes, sandboxed inside the configured output directory."""
    output_dir = settings.output_dir.expanduser().resolve()
    if save_path:
        path = Path(save_path).expanduser().resolve()
        try:
            path.relative_to(output_dir)
        except ValueError:
            raise ValueError(
                f"save_path must be inside the server's output directory "
                f"({output_dir}); got {path}. Use EVO2_MCP_OUTPUT_DIR "
                "to change where files may be written."
            ) from None
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = output_dir / f"evo2_forward_{stamp}.npz"
    path.write_bytes(npz_bytes)
    return path
