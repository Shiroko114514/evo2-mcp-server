"""Load a score_fasta.py run folder and run typical downstream analyses.

Expects the run-folder layout produced by scripts/score_fasta.py:

    output/run_YYYYMMDD_HHMMSS/
        scores.csv        # one row per sequence; embedding_key links to matrix
        embeddings.npz    # 'embeddings' (n, 4096) float32 + 'record_ids' (n,)

Shows four analyses (all optional):
  1. load + merge scores with embeddings by embedding_key
  2. unsupervised: PCA + KMeans clustering
  3. classification: enhancer vs promoter (label parsed from record key)
  4. regression: predict mean_log_likelihood from the embedding

Usage:
    python scripts/analyze_run.py output/run_20260825_105004
    python scripts/analyze_run.py output/run_20260825_105004 --k 3 --no-plot

Outputs are written next to the run folder: analysis_<runname>.*
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_run(run_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Return (score_rows, embedding_matrix, record_ids)."""
    run_dir = Path(run_dir)
    scores_path = run_dir / "scores.csv"
    emb_path = run_dir / "embeddings.npz"
    if not scores_path.is_file() or not emb_path.is_file():
        raise FileNotFoundError(f"run folder missing scores.csv or embeddings.npz: {run_dir}")

    with scores_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    d = np.load(emb_path, allow_pickle=True)
    embeddings = np.asarray(d["embeddings"], dtype=np.float32)  # (n, 4096)
    record_ids = [str(x) for x in d["record_ids"]]
    return rows, embeddings, record_ids


def merge_by_key(rows, record_ids):
    """Return (keys, row_meta, idx) where idx aligns embeddings.npz rows."""
    key_to_row = {r["embedding_key"]: r for r in rows if r.get("embedding_key")}
    if not key_to_row:
        raise SystemExit(
            "no embedding_key values in scores.csv — this run predates the "
            "embedding_key column; re-run scripts/score_fasta.py to regenerate"
        )
    keys, meta = [], []
    for key in record_ids:
        r = key_to_row.get(key)
        if r is not None:
            keys.append(key)
            meta.append(r)
        else:
            print(f"  (skip {key}: no matching scores.csv row)", file=sys.stderr)
    idx = np.array([record_ids.index(k) for k in keys], dtype=np.int64)
    return keys, meta, idx


def parse_source(key: str) -> tuple[str, str]:
    """'cis_enhancers__K562_TE_629' -> ('cis', 'enhancers')."""
    source = key.split("__")[0]
    if "_" in source:
        a, b = source.split("_", 1)
        return a, b
    return source, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="path to a run_YYYYMMDD_HHMMSS folder")
    ap.add_argument("--k", type=int, default=2, help="KMeans clusters")
    ap.add_argument("--no-plot", action="store_true", help="skip the 2-D PCA plot")
    ap.add_argument("--max-embedding-dims", type=int, default=4096,
                    help="cap embedding columns used (for quick experiments)")
    args = ap.parse_args()

    # --- 1. load + merge ----------------------------------------------------
    rows, embeddings, record_ids = load_run(args.run_dir)
    print(f"loaded {len(rows)} score rows, {embeddings.shape[0]} embeddings "
          f"(dim={embeddings.shape[1]})")
    if embeddings.shape[1] > args.max_embedding_dims:
        embeddings = embeddings[:, : args.max_embedding_dims]
        print(f"  (capped embedding dims to {args.max_embedding_dims})")

    keys, meta, idx = merge_by_key(rows, record_ids)
    X = embeddings[idx]                       # aligned with `meta`
    if len(X) < 2:
        print("need >= 2 matched sequences for analysis; aborting")
        raise SystemExit(1)
    print(f"merged {len(X)} sequences (scores.csv ∩ embeddings.npz)")

    # Normalize each embedding (unit norm) — standard for high-dim vectors.
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    out_prefix = args.run_dir.parent / f"analysis_{args.run_dir.name}"
    np.savez(out_prefix.with_suffix(".npz"), X=X, keys=np.array(keys, dtype=object))

    # --- 2. clustering ------------------------------------------------------
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    k = min(args.k, len(X))
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xn)
    print("\n[clustering] KMeans(k=%d):" % k)
    for ci in range(k):
        members = [keys[i] for i in range(len(keys)) if kmeans.labels_[i] == ci]
        print(f"  cluster {ci}: n={len(members)}  e.g. {members[:4]}")

    # --- 3. classification: enhancer vs promoter ----------------------------
    labels = [parse_source(k)[1] for k in keys]
    uniq = sorted({l for l in labels if l})
    if len(uniq) == 2 and {"enhancers", "promoters"} == set(uniq) and len(X) >= 6:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        y = np.array([1 if l == "enhancers" else 0 for l in labels])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(clf, Xn, y, cv=min(5, len(X) // 2))
        print(f"\n[classification] enhancer-vs-promoter logistic regression "
              f"(5-fold CV accuracy): {scores.mean():.3f} ± {scores.std():.3f} "
              f"(baseline {max(np.bincount(y)) / len(y):.3f})")
    else:
        print(f"\n[classification] skipped — labels present: {uniq}, "
              f"n={len(X)} (need exactly enhancers+promoters and >= 6 seqs)")

    # --- 4. regression: embedding -> mean_log_likelihood --------------------
    vals = [r.get("mean_log_likelihood") for r in meta]
    if all(v is not None for v in vals) and len(X) >= 6:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict
        from scipy.stats import pearsonr  # type: ignore[import-not-found]

        y = np.array([float(v) for v in vals])
        pred = cross_val_predict(Ridge(alpha=1.0), Xn, y, cv=min(5, len(X) // 2))
        r, _ = pearsonr(y, pred)
        print(f"\n[regression] Ridge embedding->mean_log_likelihood "
              f"cross-val Pearson r = {r:.3f}")
    else:
        print("\n[regression] skipped (need >= 6 sequences with mean_log_likelihood)")

    # --- optional plot ------------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xy = PCA(n_components=2, random_state=0).fit_transform(Xn)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            axes[0].scatter(xy[:, 0], xy[:, 1], c=kmeans.labels_, cmap="viridis", s=18)
            axes[0].set_title("PCA of Evo2 embeddings (KMeans colors)")
            axes[0].set_xlabel("PC1")
            axes[0].set_ylabel("PC2")
            if len(uniq) == 2 and {"enhancers", "promoters"} == set(uniq):
                cols = ["tab:blue" if l == "enhancers" else "tab:orange" for l in labels]
                axes[1].scatter(xy[:, 0], xy[:, 1], c=cols, s=18, alpha=0.8)
                axes[1].set_title("PCA colored by enhancer/promoter")
            else:
                axes[1].scatter(xy[:, 0], xy[:, 1], c="gray", s=18)
                axes[1].set_title("PCA")
            fig.tight_layout()
            out_png = out_prefix.with_suffix(".png")
            fig.savefig(out_png, dpi=150)
            print(f"\n[plot] saved {out_png}")
        except ImportError:
            print("\n[plot] skipped (matplotlib not installed)")


if __name__ == "__main__":
    main()
