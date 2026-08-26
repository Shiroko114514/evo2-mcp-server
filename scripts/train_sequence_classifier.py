"""Train a classifier on DNA sequence features (k-mer spectrum, GC, length,
optionally mean_log_likelihood from an Evo2 scores.csv).

All sequence features are computed LOCALLY from FASTA — no API calls, no
network. This is the fallback path when Evo2 embeddings are unavailable.

Features (per sequence):
  * k-mer frequencies for k = 1..K (default K=4) -> 4+16+64+256 = 340 dims
    (windows containing N are skipped; frequencies normalised by valid
    window count)
  * GC content
  * log10(length)
  * optional: mean_log_likelihood merged from an Evo2 scores.csv
    (--mean-ll-csv) — the only feature that needs a prior Evo2 scoring run

Pipeline: StandardScaler -> classifier (lr/svm/rf) -> stratified k-fold CV
-> accuracy / balanced-accuracy / AUC / confusion matrix / top features.

Usage:
    python scripts/train_sequence_classifier.py \
      --fasta cis/enhancers.fa --label enhancer \
      --fasta cis/promoters.fa --label promoter \
      --fasta trans/enhancers.fa --label enhancer \
      --fasta trans/promoters.fa --label promoter \
      --k-max 4 --model lr --cv 5

    # cis vs trans (labels are the region):
    python scripts/train_sequence_classifier.py \
      --fasta cis/enhancers.fa --label cis --fasta cis/promoters.fa --label cis \
      --fasta trans/enhancers.fa --label trans --fasta trans/promoters.fa --label trans \
      --mean-ll-csv output/run_.../scores.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

_TRANS = bytes.maketrans(b"ACGTN", b"\x00\x01\x02\x03\x04")


def kmer_counts(seq: str, k: int) -> tuple[np.ndarray, int]:
    """Vectorised k-mer counting (windows containing N are marked invalid).

    Returns (counts[4**k], number of valid windows).
    """
    a = np.frombuffer(seq.encode("ascii").translate(_TRANS), dtype=np.uint8)
    n = a.size
    if n < k:
        return np.zeros(4**k, dtype=np.int64), 0
    a = a.astype(np.int64)
    has_n = a >= 4
    nwin = n - k + 1
    acc = a[:nwin].copy()
    for j in range(1, k):
        acc = acc * 4 + a[j : j + nwin]
    bad = np.zeros(nwin, dtype=bool)
    for j in range(k):
        bad |= has_n[j : j + nwin]
    acc[bad] = -1
    counts = np.bincount(acc[acc >= 0], minlength=4**k)
    return counts, int((acc >= 0).sum())


_KMERS_CACHE: dict[int, list[str]] = {}


def kmer_names(k: int) -> list[str]:
    if k not in _KMERS_CACHE:
        _KMERS_CACHE[k] = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
    return _KMERS_CACHE[k]


def sequence_features(seq: str, k_max: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    names: list[str] = []
    for k in range(1, k_max + 1):
        counts, denom = kmer_counts(seq, k)
        freqs = counts.astype(np.float64) / max(denom, 1)
        parts.append(freqs)
        names.extend(f"k{k}_{m}" for m in kmer_names(k))
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    parts.append(np.array([gc, np.log10(max(len(seq), 1))], dtype=np.float64))
    names.extend(["gc", "log10_len"])
    return np.concatenate(parts), names


def load_fasta(path: Path) -> list[tuple[str, str]]:
    """Return [(id, sequence)] with id = first '|'-split token of header."""
    records: list[tuple[str, str]] = []
    cur_id, cur = None, []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.rstrip("\r\n")
        if line.startswith(">"):
            if cur_id is not None:
                records.append((cur_id, "".join(cur)))
            cur_id = line[1:].split("|")[0].split()[0]
            cur = []
        elif cur_id is not None:
            cur.append(line.strip())
    if cur_id is not None:
        records.append((cur_id, "".join(cur)))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", action="append", required=True, nargs="+", type=Path,
                    help="FASTA file(s); pair with --label")
    ap.add_argument("--label", action="append", required=True,
                    help="class label for the corresponding --fasta group")
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--mean-ll-csv", type=Path, default=None,
                    help="optional Evo2 scores.csv to merge mean_log_likelihood")
    ap.add_argument("--model", choices=["lr", "svm", "rf"], default="lr")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq-per-group", type=int, default=None,
                    help="cap sequences per label (fast smoke test)")
    args = ap.parse_args()

    if len(args.fasta) != len(args.label):
        raise SystemExit("--fasta and --label must appear the same number of times")

    # --- load sequences -----------------------------------------------------
    seqs: list[tuple[str, str]] = []  # (id, seq)
    labels: list[str] = []
    for files, lab in zip(args.fasta, args.label):
        for path in files:
            recs = load_fasta(path)
            if args.max_seq_per_group:
                recs = recs[: args.max_seq_per_group]
            seqs.extend(recs)
            labels.extend([lab] * len(recs))
            print(f"  {path}: {len(recs)} records -> label {lab!r}", flush=True)
    print(f"total {len(seqs)} sequences", flush=True)

    # --- merge Evo2 mean_ll if provided -------------------------------------
    mean_ll = {}
    if args.mean_ll_csv:
        for row in csv.DictReader(open(args.mean_ll_csv, encoding="utf-8")):
            v = row.get("mean_log_likelihood")
            if v not in (None, ""):
                mean_ll[row["id"]] = float(v)
        print(f"merged mean_log_likelihood for {len(mean_ll)} ids", flush=True)

    # --- features ------------------------------------------------------------
    print(f"computing k-mer features (k=1..{args.k_max}) ...", flush=True)
    t0 = time.time()
    X_parts, meta_rows, names = [], [], None
    for i, (rid, seq) in enumerate(seqs):
        x, names = sequence_features(seq.upper(), args.k_max)
        if rid in mean_ll:
            x = np.concatenate([x, [mean_ll[rid]]])
        else:
            x = np.concatenate([x, [0.0]])
        X_parts.append(x)
        meta_rows.append((rid, len(seq)))
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(seqs)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.vstack(X_parts).astype(np.float64)
    feature_names = names + (["mean_ll"] if args.mean_ll_csv else [])
    print(f"feature matrix: {X.shape} in {time.time()-t0:.0f}s", flush=True)

    # --- labels ---------------------------------------------------------------
    uniq = sorted(set(labels))
    if len(uniq) != 2:
        raise SystemExit(f"need exactly 2 classes, got {uniq}")
    y = np.array([0 if l == uniq[0] else 1 for l in labels])
    print(f"classes: {uniq[0]}={int((y==0).sum())}, {uniq[1]}={int((y==1).sum())}")

    # --- scale + classify ------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

    Xs = StandardScaler().fit_transform(X)
    if args.model == "lr":
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=args.seed)
    elif args.model == "svm":
        clf = SVC(kernel="linear", probability=True, random_state=args.seed)
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=args.seed)

    n_per_class = [int((y == 0).sum()), int((y == 1).sum())]
    cv = StratifiedKFold(n_splits=min(args.cv, max(2, min(n_per_class))),
                         shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(clf, Xs, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    print(f"\n[{args.model}] {uniq[0]} vs {uniq[1]}: "
          f"CV accuracy={acc:.3f} balanced-acc={bacc:.3f} "
          f"(baseline={max(n_per_class)/len(y):.3f})")
    print("confusion matrix (rows=truth, cols=pred):")
    print(confusion_matrix(y, y_pred))
    try:
        y_prob = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")[:, 1]
        print(f"AUC={roc_auc_score(y, y_prob):.3f}")
    except Exception:
        pass

    # --- feature importance -----------------------------------------------------
    clf.fit(Xs, y)
    if args.model in ("lr", "svm"):
        coef = np.abs(clf.coef_[0])
        top = np.argsort(coef)[-15:][::-1]
        print("\ntop predictive features:")
        for i in top:
            print(f"  {feature_names[i]:<10} |coef|={coef[i]:.3f}")
    elif args.model == "rf":
        imp = clf.feature_importances_
        top = np.argsort(imp)[-15:][::-1]
        print("\ntop predictive features:")
        for i in top:
            print(f"  {feature_names[i]:<10} importance={imp[i]:.4f}")


if __name__ == "__main__":
    sys.exit(main())
