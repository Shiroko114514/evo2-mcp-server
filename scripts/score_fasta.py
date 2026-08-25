"""Score DNA sequences from FASTA files with Evo2-7B and save a CSV,
optionally also extracting per-sequence embeddings.

Design notes
------------
* Raw per-position embeddings are far too large to inline into a CSV
  (a 21 kb sequence has a 700 MB `norm` embedding). They are therefore
  saved as NPZ archives:
    - pooled (default "mean", shape (hidden_size,), float32) for every
      sequence in one `.npz` keyed by record id, and
    - optionally, the full raw embedding per record in `--embedding-raw-dir`.
  The CSV carries embedding metadata columns (layer / shape / archive path /
  key / raw path) instead of the tensors themselves.
* When embeddings are enabled, each sequence is sent ONCE with
  `output_layers=[<logits layer>, <embedding layer>]` (deduplicated if they
  are the same), so scoring and embedding extraction share a single request.
* The logits layer is auto-detected once per run (hosted API uses 'unembed').

Examples
--------
Sample run with embeddings (pooled archive + raw files):
    python scripts/score_fasta.py \
      --fasta /path/positive_sequence/cis/enhancers.fa \
             /path/positive_sequence/cis/promoters.fa \
      --ids K562_TE_629,K562_TE_3946,K562_MPT_6842 \
      --allow-ambiguous \
      --embedding-layer norm

  Every run writes a NEW timestamped folder:
      output/run_20260825_103000/
        scores.csv            # per-sequence scores (id, total/mean LL, ...)
                              # + embedding_key (matches embeddings.npz
                              #   record_ids, row-for-row; empty for
                              #   error/skipped rows or when embeddings
                              #   are disabled)
        embeddings.npz        # mean-pooled embedding matrix (n, 4096) +
                              # record_ids (source__id, aligned with rows)
        embeddings_raw/       # ONLY if --keep-raw-embeddings: per-record raw
                              # (1, seq, 4096) npz files — VERY large on disk

Full run (all records, skip >100kb, keep raw embeddings):
    python scripts/score_fasta.py \
      --fasta /path/positive_sequence/cis/enhancers.fa \
             /path/positive_sequence/cis/promoters.fa \
             /path/positive_sequence/trans/enhancers.fa \
             /path/positive_sequence/trans/promoters.fa \
      --skip-longer-than 100000 --allow-ambiguous \
      --embedding-layer norm --keep-raw-embeddings

Backward-compatible: --out / --embeddings-out / --embedding-raw-dir still
override the default run-folder layout.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from evo2_mcp.api_client import Evo2Client
from evo2_mcp.config import Settings
from evo2_mcp.forward_output import decode_npz, find_logits, score_from_logits
from evo2_mcp.sequence import normalise_sequence
from evo2_mcp.tools import evo2_score

FIELD_NAMES = [
    "id", "header", "sequence_length", "n_count", "status", "reason",
    "likelihood_computed", "total_log_likelihood", "mean_log_likelihood",
    "scored_positions", "api_elapsed_ms", "warning", "wall_seconds",
    "embedding_key",
]

# Sanity warning threshold for raw per-record embedding files (bytes).
RAW_EMBEDDING_WARN_BYTES = 200 * 1024 * 1024


def load_fasta(path: Path, ids: set[str] | None) -> list[tuple[str, str, str]]:
    """Return [(id, header, sequence)] for all records (or only `ids`)."""
    records: list[tuple[str, str, str]] = []
    cur_id, cur_header, cur = None, None, []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.rstrip("\r\n")
        if line.startswith(">"):
            if cur_id is not None:
                records.append((cur_id, cur_header, "".join(cur)))
            cur_header = line[1:].strip()
            cur_id = cur_header.split()[0].split("|")[0]
            cur = []
        elif cur_id is not None:
            cur.append(line.strip())
    if cur_id is not None:
        records.append((cur_id, cur_header, "".join(cur)))
    if ids is not None:
        records = [r for r in records if r[0] in ids]
    return records


def source_label(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def sanitize_key(key: str) -> str:
    """Make a record identifier safe as an npz archive key / filename."""
    return re.sub(r"[^0-9A-Za-z_]", "_", key)


def pool_embedding(arr: np.ndarray, method: str = "mean") -> np.ndarray:
    """Pool a (1, seq, hidden) / (seq, hidden) embedding to (hidden,) float32."""
    a = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
    if method == "mean":
        return a.mean(axis=0).astype(np.float32)
    raise ValueError(f"unknown pooling method {method!r}")


def find_embedding_array(decoded: dict, layer: str) -> tuple[str, np.ndarray]:
    """Locate the embedding array for `layer`, returning (npz_key, array)."""
    for key in (layer, f"{layer}.output"):
        if key in decoded:
            return key, decoded[key].array
    raise KeyError(
        f"requested embedding layer {layer!r} not in response (keys: {sorted(decoded)})"
    )


def _base_entry(source: str, rec_id: str, header: str, seq: str, status: str = "ok") -> dict:
    entry = dict.fromkeys(FIELD_NAMES)
    entry.update({
        "id": rec_id, "header": header,
        "sequence_length": len(seq), "n_count": seq.count("N"), "status": status,
    })
    return entry


async def score_basic(
    client: Evo2Client, settings: Settings, source: str, rec_id: str,
    header: str, seq: str, allow_ambiguous: bool, skip_longer_than: int | None,
) -> dict:
    entry = _base_entry(source, rec_id, header, seq)
    if skip_longer_than and len(seq) > skip_longer_than:
        entry.update({"status": "skipped", "reason": f"len>{skip_longer_than}"})
        return entry
    t0 = time.time()
    try:
        norm = normalise_sequence(seq, name=f"record[{rec_id}]", allow_ambiguous=allow_ambiguous)
        if norm.length < 2:
            entry.update({"status": "error", "reason": "sequence < 2 bp"})
            return entry
        result = await evo2_score(sequence=norm.sequence, settings=settings, client=client)
        entry.update({
            "likelihood_computed": result.get("likelihood_computed"),
            "total_log_likelihood": result.get("total_log_likelihood"),
            "mean_log_likelihood": result.get("mean_log_likelihood"),
            "scored_positions": result.get("scored_positions"),
            "api_elapsed_ms": result.get("api", {}).get("elapsed_ms"),
            "warning": " | ".join(result.get("warnings") or []) or None,
        })
    except Exception as e:  # noqa: BLE001 - per-record isolation
        entry.update({"status": "error", "reason": f"{type(e).__name__}: {str(e)[:300]}"})
    entry["wall_seconds"] = round(time.time() - t0, 2)
    return entry


async def score_with_embedding(
    client: Evo2Client, settings: Settings, source: str, rec_id: str,
    header: str, seq: str, allow_ambiguous: bool, skip_longer_than: int | None,
    logits_layer: str, emb_layer: str, emb_raw_dir: Path | None,
) -> tuple[dict, tuple[str, np.ndarray] | None]:
    """Score AND extract an embedding with a single forward request.

    Returns (entry, (pooled_key, pooled_vector)) — pooled vector is None on
    error/skip.
    """
    entry = _base_entry(source, rec_id, header, seq)
    if skip_longer_than and len(seq) > skip_longer_than:
        entry.update({"status": "skipped", "reason": f"len>{skip_longer_than}"})
        return entry, None
    t0 = time.time()
    try:
        norm = normalise_sequence(seq, name=f"record[{rec_id}]", allow_ambiguous=allow_ambiguous)
        if norm.length < 2:
            entry.update({"status": "error", "reason": "sequence < 2 bp"})
            return entry, None
        layers = [logits_layer] if emb_layer == logits_layer else [logits_layer, emb_layer]
        npz_bytes, meta = await client.forward(sequence=norm.sequence, output_layers=layers)
        decoded = decode_npz(npz_bytes)
        logits = find_logits(decoded, preferred_name=logits_layer)
        score = score_from_logits(logits, norm.sequence, include_per_position=False)

        if emb_layer == logits_layer:
            emb_key, emb_arr = f"{emb_layer}.output" if f"{emb_layer}.output" in decoded else emb_layer, logits
        else:
            emb_key, emb_arr = find_embedding_array(decoded, emb_layer)

        entry.update({
            "likelihood_computed": True,
            "total_log_likelihood": score.total_log_likelihood,
            "mean_log_likelihood": score.mean_log_likelihood,
            "scored_positions": score.scored_positions,
            "api_elapsed_ms": meta.get("elapsed_ms"),
            "warning": norm.warning,
        })

        pooled = pool_embedding(emb_arr, method="mean")
        pooled_key = sanitize_key(f"{source}__{rec_id}")
        entry["embedding_key"] = pooled_key  # matches record_ids in embeddings.npz

        if emb_raw_dir is not None:
            emb_raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = emb_raw_dir / f"{pooled_key}.npz"
            if emb_arr.size * emb_arr.itemsize > RAW_EMBEDDING_WARN_BYTES:
                print(f"  WARN {rec_id}: raw embedding is "
                      f"{emb_arr.size * emb_arr.itemsize / 1e6:.0f} MB — large file",
                      file=sys.stderr, flush=True)
            np.savez(raw_path, **{emb_key: emb_arr})
            # path is discoverable in the run folder; not added to the CSV
            # (kept to the plain scoring columns).

        return entry, (pooled_key, pooled)
    except Exception as e:  # noqa: BLE001 - per-record isolation
        entry.update({"status": "error", "reason": f"{type(e).__name__}: {str(e)[:300]}"})
        return entry, None
    finally:
        entry["wall_seconds"] = round(time.time() - t0, 2)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", nargs="+", required=True, type=Path)
    ap.add_argument("--ids", help="comma-separated record ids (sample run)")
    ap.add_argument("--out", type=Path, default=None,
                    help="override the default run-folder scores.csv path")
    ap.add_argument("--allow-ambiguous", action="store_true",
                    help="pass N through (server must also allow it)")
    ap.add_argument("--skip-longer-than", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=2)
    ap.add_argument("--embedding-layer", default="norm",
                    help="layer to extract embeddings from (e.g. norm, "
                         "blocks.31, embedding_layer); 'none' disables")
    ap.add_argument("--embeddings-out", type=Path, default=None,
                    help="override the default run-folder embeddings.npz path")
    ap.add_argument("--embedding-raw-dir", type=Path, default=None,
                    help="override the default run-folder embeddings_raw/ dir")
    ap.add_argument("--keep-raw-embeddings", action="store_true",
                    help="also write per-record raw (1, seq, 4096) npz files "
                         "(DEFAULT OFF — they are huge on disk)")
    args = ap.parse_args()

    ids = {i.strip() for i in args.ids.split(",")} if args.ids else None
    records: list[tuple[Path, str, str, str, str]] = []
    for path in args.fasta:
        src = source_label(path)
        records.extend((path, src, *r) for r in load_fasta(path, ids))
    if not records:
        print("no records selected; check --fasta / --ids", file=sys.stderr)
        raise SystemExit(2)

    if args.allow_ambiguous:
        os.environ["EVO2_MCP_ALLOW_AMBIGUOUS"] = "1"

    emb_layer = args.embedding_layer.lower()
    with_emb = emb_layer != "none"

    settings = Settings.from_env()
    print(f"selected {len(records)} records; concurrency={args.max_concurrency} "
          f"base_url={settings.base_url}", flush=True)

    # --- Resolve the per-run output layout (new timestamped folder per run) ---
    out_root = settings.output_dir.resolve()
    run_dir = out_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run folder: {run_dir}", flush=True)

    csv_path = args.out or (run_dir / "scores.csv")
    emb_raw_dir = args.embedding_raw_dir or (run_dir / "embeddings_raw" if args.keep_raw_embeddings else None)
    emb_out_path = args.embeddings_out or (run_dir / "embeddings.npz" if with_emb else None)

    results: list[dict] = []
    pooled: dict[str, np.ndarray] = {}

    async with Evo2Client(settings) as client:
        logits_layer = None
        if with_emb:
            # Resolve the logits layer once (hosted API: output_layer -> unembed).
            await client.forward_logits(sequence="ACGT")
            logits_layer = client.logits_layer
            print(f"logits layer: {logits_layer}; embedding layer: {emb_layer}", flush=True)

        sem = asyncio.Semaphore(max(1, args.max_concurrency))

        async def guarded(path, source, rec_id, header, seq):
            async with sem:
                if with_emb:
                    entry, pooled_vec = await score_with_embedding(
                        client, settings, source, rec_id, header, seq,
                        args.allow_ambiguous, args.skip_longer_than,
                        logits_layer, emb_layer, args.embedding_raw_dir,
                    )
                    return entry, pooled_vec
                return await score_basic(
                    client, settings, source, rec_id, header, seq,
                    args.allow_ambiguous, args.skip_longer_than,
                ), None

        done = 0
        for i in range(0, len(records), args.max_concurrency):
            batch = records[i:i + args.max_concurrency]
            outcomes = await asyncio.gather(*(guarded(*b) for b in batch))
            for entry, pooled_vec in outcomes:
                results.append(entry)
                if pooled_vec is not None and entry["status"] == "ok":
                    pooled[pooled_vec[0]] = pooled_vec[1]
            done += len(outcomes)
            ok = sum(1 for e in outcomes if e[0]["status"] == "ok")
            err = sum(1 for e in outcomes if e[0]["status"] == "error")
            print(f"  [{done}/{len(records)}] last batch: ok={ok} err={err}", flush=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELD_NAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    if with_emb and pooled:
        # Write the embedding matrix as a 2-D array with the record order
        # stored in a parallel key list (npz keys must be unique, and record
        # ids can repeat across sources).
        emb_out_path.parent.mkdir(parents=True, exist_ok=True)
        order = list(pooled.keys())
        matrix = np.stack([pooled[k] for k in order])
        np.savez(emb_out_path, embeddings=matrix, record_ids=np.array(order, dtype=object))
        print(f"embedding matrix ({matrix.shape[0]} x {matrix.shape[1]}, "
              f"float{8 * matrix.dtype.itemsize}) -> {emb_out_path}", flush=True)
    elif with_emb:
        print("no embeddings written (no successful records)", flush=True)

    n_ok = sum(1 for e in results if e["status"] == "ok")
    n_err = sum(1 for e in results if e["status"] == "error")
    n_skip = sum(1 for e in results if e["status"] == "skipped")
    for e in results:
        if e["status"] == "error":
            print(f"  ERROR {e['source']}::{e['id']}: {e['reason']}", file=sys.stderr)
    print(f"\ndone: {n_ok} ok, {n_err} error, {n_skip} skipped -> {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
