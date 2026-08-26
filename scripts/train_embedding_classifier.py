"""Train a classifier on Evo2 mean-pooled embeddings.

Input: a run folder produced by scripts/score_fasta.py with embeddings enabled:

    output/run_YYYYMMDD_HHMMSS/
        scores.csv        # embedding_key column links to the matrix
        embeddings.npz    # 'embeddings' (n, 4096) float32, 'record_ids' (n,)

Labels are derived from the embedding key (`<source>__<id>`), where source is
e.g. `cis_enhancers` / `trans_promoters`. Two label modes are supported:
  * --label-type type      -> enhancers vs promoters (default)
  * --label-type region    -> cis vs trans

Pipeline:
  1. load embeddings + merge scores by embedding_key
  2. unit-normalise each vector (standard for high-dim embeddings)
  3. optional PCA dimensionality reduction
  4. train a classifier (logistic regression by default; --model svm/rf)
     with stratified k-fold cross-validation
  5. report accuracy, balanced accuracy, AUC (if both classes), confusion
     matrix, and (for LR) the top predictive embedding dimensions

Usage:
    python scripts/train_embedding_classifier.py output/run_20260825_105004 \
        --label-type type --model lr --pca 100 --cv 5

NOTE: embeddings are mean-pooled (one 4096-d vector per sequence). For
classification you typically need >= ~20-30 samples per class; with fewer,
expect high-variance cross-validation scores.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_source(key: str) -> tuple[str, str]:
    """'cis_enhancers__K562_TE_629' -> ('cis', 'enhancers')."""
    source = key.split("__")[0]
    if "_" in source:
        a, b = source.split("_", 1)
        return a, b
    return source, ""


def load_merged(run_dir: Path):
    run_dir = Path(run_dir)
    d = np.load(run_dir / "embeddings.npz", allow_pickle=True)
    X = np.asarray(d["embeddings"], dtype=np.float32)
    ids = [str(x) for x in d["record_ids"]]

    rows = list(csv.DictReader(open(run_dir / "scores.csv", encoding="utf-8")))
    key_to_row = {r["embedding_key"]: r for r in rows if r.get("embedding_key")}

    keep, keep_idx, meta = [], [], []
    for i, k in enumerate(ids):
        if k in key_to_row:
            keep.append(k)
            keep_idx.append(i)
            meta.append(key_to_row[k])
        else:
            print(f"  (skip {k}: no scores.csv row)", flush=True)
    return X[np.array(keep_idx)], ids, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--label-type", choices=["type", "region"], default="type",
                    help="type = enhancers vs promoters; region = cis vs trans")
    ap.add_argument("--model", choices=["lr", "svm", "rf"], default="lr")
    ap.add_argument("--pca", type=int, default=None,
                    help="reduce to this many PCA components first (None = raw)")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_raw, ids, meta = load_merged(args.run_dir)
    n = len(X_raw)
    print(f"loaded {n} embeddings, dim={X_raw.shape[1]}")
    if n < 6:
        raise SystemExit("too few samples (<6); embedding subset run needed first")

    # --- labels from keys -------------------------------------------------
    labels = []
    for k in ids:
        region, typ = parse_source(k)
        labels.append(typ if args.label_type == "type" else region)
    uniq = sorted({l for l in labels if l})
    if len(uniq) != 2:
        raise SystemExit(f"need exactly 2 classes for {args.label_type}, found {uniq}")
    y = np.array([0 if l == uniq[0] else 1 for l in labels])
    print(f"classes: {uniq[0]}={int((y == 0).sum())}, {uniq[1]}={int((y == 1).sum())}")

    # --- normalise + optional PCA ------------------------------------------
    X = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-12)
    if args.pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(args.pca, n - 1), random_state=args.seed)
        X = pca.fit_transform(X)
        print(f"PCA -> {X.shape[1]} components "
              f"(explained var {pca.explained_variance_ratio_.sum():.2%})")

    # --- classifier --------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 roc_auc_score, confusion_matrix)

    if args.model == "lr":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
    elif args.model == "svm":
        clf = SVC(kernel="linear", probability=True, random_state=args.seed)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=args.seed)

    cv = StratifiedKFold(n_splits=min(args.cv, max(2, min(np.bincount(y)))),
                         shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    print(f"\n[{args.model} on {args.label_type}] CV accuracy={acc:.3f} "
          f"balanced-acc={bacc:.3f} (baseline={max(np.bincount(y)) / n:.3f})")
    print("confusion matrix (rows=truth, cols=pred):")
    print(confusion_matrix(y, y_pred))
    try:
        y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        print(f"AUC={roc_auc_score(y, y_prob):.3f}")
    except Exception:
        pass

    # --- most predictive dimensions (LR only, raw space) -------------------
    if args.model == "lr" and args.pca is None:
        clf.fit(X, y)
        coef = np.abs(clf.coef_[0])
        top = np.argsort(coef)[-10:][::-1]
        print("\ntop predictive embedding dims (index: |coef|):")
        for i in top:
            print(f"  dim {i}: {coef[i]:.4f}")


if __name__ == "__main__":
    main()
