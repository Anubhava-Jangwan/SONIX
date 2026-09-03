#!/usr/bin/env python3
"""
split_indic_holdout.py  --  SONIX / SIH26104

Carve a LEAK-FREE holdout out of an embedding root, splitting by RECORDING,
not by row.

WHY THIS EXISTS
IndicVoices filenames are <recording_id>_chunk_<n>. Chunks 1..N are slices of
the SAME recording, same speaker, same room, same microphone. A random row-level
split puts chunk 1 in train and chunk 2 in the holdout, and the model scores the
holdout near-perfectly because it has already memorised that exact recording.
The result looks excellent and means nothing.

Splitting by recording id keeps every chunk of a recording on one side.

    python split_indic_holdout.py ^
        --src outputs\\embeddings_indicvoices ^
        --out-train outputs\\embeddings_indicvoices_tr ^
        --out-holdout outputs\\embeddings_indicvoices_ho ^
        --holdout-frac 0.12

THEN retrain WITHOUT the holdout, and score the holdout with every model:

    python src\\train.py --emb-root D:\\embeddings ^
        --extra-emb-root D:\\embeddings_g711 ^
        --extra-emb-root D:\\embeddings_rawboost ^
        --extra-emb-root outputs\\embeddings_rirmusan_bonafide ^
        --extra-emb-root outputs\\embeddings_rirmusan_spoof ^
        --extra-emb-root outputs\\embeddings_indicvoices_tr ^
        --out outputs\\models\\head_full_ho.pt

    python src\\eval.py --split train --emb-root outputs\\embeddings_indicvoices_ho ^
        --model-ckpt outputs\\models\\head_full_ho.pt --out-scores outputs\\scores_ho_full
    python src\\eval.py --split train --emb-root outputs\\embeddings_indicvoices_ho ^
        --model-ckpt outputs\\models\\head.pt --out-scores outputs\\scores_ho_baseline
    python src\\eval.py --split train --emb-root outputs\\embeddings_indicvoices_ho ^
        --model-ckpt outputs\\models\\head_robust_v2.pt --out-scores outputs\\scores_ho_robust

    python flag_rate.py --split train ^
        --scores-dir outputs\\scores_ho_baseline ^
        --scores-dir outputs\\scores_ho_robust ^
        --scores-dir outputs\\scores_ho_full

baseline and robust_v2 never saw ANY IndicVoices, so their holdout numbers are
clean either way. head_full_ho is the one this script exists to make honest.
"""

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SHARD_SIZE = 100
# <recording_id>_chunk_<n>  ->  <recording_id>
CHUNK_RE = re.compile(r"^(.*?)_chunk_\d+$", re.IGNORECASE)


def recording_id(stem: str) -> str:
    m = CHUNK_RE.match(stem)
    if m:
        return m.group(1)
    # no _chunk_ suffix: treat the whole stem as its own recording
    return stem


def load_root(src: Path):
    d = src / "train"
    if not d.is_dir():
        sys.exit(f"FATAL: {d} does not exist")
    shards = sorted(s for s in d.glob("shard_*.npy") if ".labels" not in s.name)
    if not shards:
        sys.exit(f"FATAL: no shards in {d}")

    X, Y, F = [], [], []
    for s in shards:
        lab = s.with_suffix(".labels.npy")
        fl = Path(str(s)[:-4] + ".files.txt")
        if not lab.exists():
            sys.exit(f"FATAL: {s.name} has no labels sidecar")
        emb = np.load(s)
        y = np.load(lab)
        if len(emb) != len(y):
            sys.exit(f"FATAL: {s.name} rows {len(emb)} != labels {len(y)}")
        if fl.exists():
            names = fl.read_text(encoding="utf-8", errors="replace").split()
        else:
            names = [f"{s.stem}_row{i}" for i in range(len(emb))]
        if len(names) != len(emb):
            sys.exit(f"FATAL: {fl.name} lists {len(names)} names for {len(emb)} rows")
        X.append(emb.astype(np.float32))
        Y.append(y.astype(np.int8))
        F.extend(names)
    return np.concatenate(X, 0), np.concatenate(Y, 0), F


def write_root(out: Path, X, Y, F, tag):
    d = out / "train"
    if d.exists() and any(d.iterdir()):
        sys.exit(f"FATAL: {d} exists and is not empty. Delete it first -- refusing "
                 f"to merge into a directory that may hold an earlier split.")
    d.mkdir(parents=True, exist_ok=True)
    n = len(X)
    for i, start in enumerate(range(0, n, SHARD_SIZE)):
        sl = slice(start, min(start + SHARD_SIZE, n))
        stem = d / f"shard_{i:05d}"
        np.save(stem.with_suffix(".npy"), X[sl])
        with open(str(stem) + ".labels.npy", "wb") as fh:
            np.save(fh, Y[sl])
        Path(str(stem) + ".files.txt").write_text("\n".join(F[sl]) + "\n",
                                                  encoding="utf-8")
    print(f"  {tag:8s} {n:7d} rows -> {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="embedding root holding train/")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-holdout", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.12)
    ap.add_argument("--seed", default="sonix", help="salt for the deterministic split")
    args = ap.parse_args()

    X, Y, F = load_root(Path(args.src))
    print(f"source rows        : {len(X)}")
    print(f"labels             : {dict(Counter(Y.tolist()))}")

    rec = [recording_id(Path(f).stem) for f in F]
    uniq = sorted(set(rec))
    print(f"distinct recordings: {len(uniq)}")
    if len(uniq) == len(rec):
        print("  ! every row is its own recording -- no _chunk_ grouping found.")
        print("  ! the split is still valid, but check the filenames are what you expect.")
    else:
        print(f"  mean chunks per recording: {len(rec)/len(uniq):.1f}")

    # deterministic per-recording assignment -- same input always gives the
    # same split, and no chunk of a recording can straddle the boundary
    def to_holdout(r):
        h = hashlib.sha1(f"{args.seed}:{r}".encode()).digest()
        return (int.from_bytes(h[:4], "big") / 2**32) < args.holdout_frac

    ho_recs = {r for r in uniq if to_holdout(r)}
    mask = np.array([r in ho_recs for r in rec], dtype=bool)

    print(f"holdout recordings : {len(ho_recs)} of {len(uniq)} "
          f"({100*len(ho_recs)/len(uniq):.1f}%)")
    print(f"holdout rows       : {int(mask.sum())} of {len(mask)} "
          f"({100*mask.mean():.1f}%)")

    if mask.sum() == 0 or mask.sum() == len(mask):
        sys.exit("FATAL: split produced an empty side. Adjust --holdout-frac.")

    # prove no recording appears on both sides
    tr_recs = {r for r, m in zip(rec, mask) if not m}
    overlap = tr_recs & ho_recs
    if overlap:
        sys.exit(f"FATAL: {len(overlap)} recordings on both sides "
                 f"(e.g. {sorted(overlap)[:3]}). The split leaked -- refusing to write.")
    print("no recording appears on both sides -- split is leak-free")
    print()

    F = np.array(F, dtype=object)
    write_root(Path(args.out_train), X[~mask], Y[~mask], list(F[~mask]), "train")
    write_root(Path(args.out_holdout), X[mask], Y[mask], list(F[mask]), "holdout")
    print()
    print("Retrain WITHOUT the holdout, then score the holdout. See the header "
          "of this file for the exact commands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
