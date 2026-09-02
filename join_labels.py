#!/usr/bin/env python3
"""
join_labels.py  --  SONIX / SIH26104

Attach REAL labels to embeddings that were extracted with `--audio-dir`.

WHY THIS EXISTS
`extract_embeddings.py --audio-dir` has no protocol, so it writes a PLACEHOLDER
label of 0 (bonafide) for every clip and saves the filenames alongside in
shard_XXXXX.files.txt. Scoring that as-is makes every clip count as "real": the
EER comes out meaningless or nan, and it looks like the model failed when
actually the labels were never there.

This reads the dataset's own key file, matches it to those saved filenames, and
rewrites the .labels.npy sidecars in the exact row order of each shard.

Use it for ASVspoof 2021 DF (needs the DF CM key / trial_metadata) and for
In-the-Wild (meta.csv).

    python join_labels.py --emb-dir outputs/embeddings_df21/eval \
                          --key-file path/to/trial_metadata.txt

    python join_labels.py --emb-dir outputs/embeddings/itw \
                          --key-file data/in_the_wild/meta.csv

    # look first, change nothing
    python join_labels.py --emb-dir ... --key-file ... --dry-run

The key file's layout is detected automatically: it finds the column holding
bonafide/spoof and the column whose values match our filenames. Handles
whitespace- or comma-separated files and the 'bona-fide' spelling variant.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SPOOF_WORDS = {"spoof", "fake", "1"}
BONA_WORDS = {"bonafide", "bona-fide", "bona_fide", "genuine", "real", "0"}


def split_row(line):
    if "," in line and len(line.split(",")) >= len(line.split()):
        return [c.strip() for c in line.split(",")]
    return line.split()


def load_key(key_path, wanted):
    """Return {stem: label}. Auto-detects the id column and the label column."""
    rows = []
    for line in Path(key_path).read_text(errors="replace").splitlines():
        if line.strip():
            rows.append(split_row(line))
    if not rows:
        sys.exit(f"FATAL: key file is empty: {key_path}")

    ncol = max(len(r) for r in rows)

    # label column = the one whose values are mostly bonafide/spoof words
    label_col, best = None, 0.0
    for c in range(ncol):
        vals = [r[c].lower() for r in rows if len(r) > c]
        if not vals:
            continue
        hit = sum(1 for v in vals if v in SPOOF_WORDS or v in BONA_WORDS)
        frac = hit / len(vals)
        if frac > best:
            best, label_col = frac, c
    if label_col is None or best < 0.8:
        sys.exit("FATAL: could not find a bonafide/spoof column in the key file.\n"
                 "Open it and check it is the CM key, not the ASV key.")

    # id column = the one matching most of our filenames
    id_col, best_id = None, 0
    for c in range(ncol):
        vals = {Path(r[c]).stem for r in rows if len(r) > c}
        hit = len(vals & wanted)
        if hit > best_id:
            best_id, id_col = hit, c
    if id_col is None or best_id == 0:
        sys.exit("FATAL: no column in the key file matches our audio filenames.\n"
                 "Is this the key for the right dataset/subset?")

    out = {}
    for r in rows:
        if len(r) <= max(id_col, label_col):
            continue
        v = r[label_col].lower()
        if v in SPOOF_WORDS:
            out[Path(r[id_col]).stem] = 1
        elif v in BONA_WORDS:
            out[Path(r[id_col]).stem] = 0
    print(f"key file: id column {id_col}, label column {label_col}, "
          f"{len(out)} labelled entries")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Join real labels onto embeddings.")
    ap.add_argument("--emb-dir", required=True,
                    help="split dir, e.g. outputs/embeddings_df21/eval")
    ap.add_argument("--key-file", required=True,
                    help="the dataset's key/metadata file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = Path(args.emb_dir)
    shards = sorted(s for s in d.glob("shard_*.npy") if ".labels" not in s.name)
    if not shards:
        sys.exit(f"FATAL: no embedding shards in {d.resolve()}")

    # gather every filename we have, in shard row order
    per_shard = []
    wanted = set()
    for s in shards:
        fp = s.with_suffix(".files.txt")
        if not fp.exists():
            sys.exit(f"FATAL: {s.name} has no {fp.name}. Without the filename\n"
                     f"sidecar the labels cannot be matched. Re-extract this shard.")
        names = [Path(x.strip()).stem for x in
                 fp.read_text().splitlines() if x.strip()]
        n = len(np.load(s, mmap_mode="r"))
        if len(names) != n:
            sys.exit(f"FATAL: {s.name} has {n} rows but {fp.name} lists "
                     f"{len(names)} files. Corrupt shard - re-extract it.")
        per_shard.append((s, names))
        wanted.update(names)

    print(f"{len(shards)} shards, {len(wanted)} unique clips in {d}")
    key = load_key(args.key_file, wanted)

    missing = wanted - set(key)
    if missing:
        print(f"\n! {len(missing)} clips have no entry in the key file, e.g. "
              f"{sorted(missing)[:3]}")
        if len(missing) == len(wanted):
            sys.exit("FATAL: nothing matched. Wrong key file for this dataset.")

    total = {0: 0, 1: 0}
    unmatched = 0
    for s, names in per_shard:
        labels = np.zeros(len(names), dtype=np.int8)
        for i, nm in enumerate(names):
            if nm in key:
                labels[i] = key[nm]
            else:
                labels[i] = -1
                unmatched += 1
        for v in (0, 1):
            total[v] += int((labels == v).sum())
        if not args.dry_run:
            lab_p = s.with_suffix(".labels.npy")
            tmp = lab_p.with_suffix(".npy.tmp")
            np.save(tmp, labels)
            tmp.replace(lab_p)

    print(f"\n{'[dry run] would write' if args.dry_run else 'wrote'} labels:")
    print(f"  bonafide (0): {total[0]}")
    print(f"  spoof    (1): {total[1]}")
    if unmatched:
        print(f"  UNMATCHED   : {unmatched}  (marked -1 - drop these before scoring)")
    if total[0] == 0 or total[1] == 0:
        print("\n! WARNING: only one class present. EER cannot be computed.")
    else:
        print("\nBoth classes present - ready for eval.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
