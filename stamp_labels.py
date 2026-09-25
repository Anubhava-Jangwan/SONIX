#!/usr/bin/env python3
"""
stamp_labels.py  --  SONIX / SIH26104

Fix the labels on embeddings extracted with `extract_embeddings.py --audio-dir`.

WHY THIS EXISTS
`--audio-dir` mode has no protocol file, so it writes a PLACEHOLDER label of 0
(bonafide) for every clip. If you extract a folder of CLONED audio that way and
train on it directly, every fake is labelled real -- a silent, catastrophic label
bug. This stamps the correct label across a split's shards.

That is also why we keep real/ and fake/ in SEPARATE folders and extract them
separately: the folder determines the label, so there is nothing to match up by
hand and nothing to get wrong.

USAGE
    # cloned / AI audio -> spoof (1)
    python stamp_labels.py --emb-dir outputs/embeddings_fake/train --label 1

    # genuine audio -> bonafide (0)   (already 0, but run it to be explicit)
    python stamp_labels.py --emb-dir outputs/embeddings_real/train --label 0

    # look without changing anything
    python stamp_labels.py --emb-dir outputs/embeddings_fake/train --check

Idempotent and safe to re-run. Row counts are verified against the embedding
shards, so labels can never drift out of sync.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Stamp labels on extracted embeddings.")
    ap.add_argument("--emb-dir", required=True,
                    help="a SPLIT directory, e.g. outputs/embeddings_fake/train")
    ap.add_argument("--label", type=int, choices=[0, 1],
                    help="0 = bonafide (real), 1 = spoof (cloned)")
    ap.add_argument("--check", action="store_true",
                    help="report current labels and exit without writing")
    args = ap.parse_args()

    d = Path(args.emb_dir)
    if not d.is_dir():
        sys.exit(f"FATAL: not a directory: {d.resolve()}")

    shards = sorted(s for s in d.glob("shard_*.npy") if ".labels" not in s.name)
    if not shards:
        sys.exit(f"FATAL: no embedding shards in {d.resolve()}\n"
                 f"Expected files like shard_00000.npy - did extraction finish?")

    if not args.check and args.label is None:
        sys.exit("FATAL: pass --label 0 or --label 1 (or use --check).")

    total = 0
    counts = {0: 0, 1: 0}
    changed = 0

    for s in shards:
        lab_p = s.with_suffix(".labels.npy")
        n = len(np.load(s, mmap_mode="r"))
        total += n

        if lab_p.exists():
            cur = np.load(lab_p)
            if len(cur) != n:
                sys.exit(f"FATAL: {s.name} has {n} rows but "
                         f"{lab_p.name} has {len(cur)} labels. Corrupt shard - "
                         f"delete it and re-extract.")
            for v in (0, 1):
                counts[v] += int((cur == v).sum())
        else:
            if args.check:
                print(f"  ! {lab_p.name} missing")
                continue

        if args.check:
            continue

        new = np.full(n, args.label, dtype=np.int8)
        if (not lab_p.exists()) or (not np.array_equal(np.load(lab_p), new)):
            # np.save() APPENDS ".npy" to any path that does not already end
            # in it, so np.save(Path(".../x.labels.npy.tmp"), ...) wrote
            # x.labels.npy.tmp.npy and the rename below then failed with
            # FileNotFoundError. Passing an open handle keeps the name we chose.
            # This only ever fired when a label actually needed changing -- i.e.
            # exactly when stamping spoof folders, the case the tool exists for.
            tmp = lab_p.with_suffix(".npy.tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, new)
            tmp.replace(lab_p)
            changed += 1

    name = {0: "bonafide/real", 1: "spoof/cloned"}
    if args.check:
        print(f"[check] {d}  shards={len(shards)}  rows={total}")
        print(f"        current labels: bonafide(0)={counts[0]}  spoof(1)={counts[1]}")
        if counts[1] == 0 and counts[0] == total:
            print("        NOTE: all zeros - this is the --audio-dir placeholder. "
                  "If this folder is CLONED audio you must stamp --label 1.")
        return 0

    print(f"[stamp] {d}")
    print(f"        shards={len(shards)}  rows={total}  rewritten={changed}")
    print(f"        every row is now label {args.label} = {name[args.label]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
