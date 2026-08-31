#!/usr/bin/env python3
"""
merge_df21.py  --  SONIX / SIH26104

Build ONE clean, correctly-labelled ASVspoof 2021 DF eval directory out of the
separately-extracted embedding parts, so eval.py can score it.

WHY THIS EXISTS
    extract_embeddings.py was run on several machines with --audio-dir. That code
    path deliberately writes label 0 as a PLACEHOLDER for every clip (see its
    _build_manifest_dir docstring) and records the real trial IDs in the
    shard_*.files.txt sidecars. The true labels live in the official DF21 CM key
    (keys/DF/CM/trial_metadata.txt) and are joined back by trial ID here.

    Each part was written with its own --out root, so every part restarts shard
    numbering at shard_00000. They CANNOT be merged by copying -- the names
    collide. This script renumbers.

WHAT IT GUARANTEES (it aborts rather than guess)
    * every embedding's trial ID is found in the CM key            -> else abort
    * no trial ID appears in two different parts                   -> else abort
    * byte-identical duplicate parts are DETECTED, reported, and used once
    * embedding rows and label rows stay aligned, per shard
    * it NEVER modifies the source parts and NEVER invents a label
    * it writes COVERAGE.txt recording exactly what the output does and does not
      contain, so the resulting EER can never be quoted as full-set DF21 by
      accident

USAGE
    python src\\merge_df21.py ^
        --parts outputs\\embeddings\\eval_full outputs\\embeddings\\eval_part01 ^
                outputs\\embeddings\\eval_part02 outputs\\embeddings\\eval_part03 ^
        --key D:\\DF-keys-full\\keys\\DF\\CM\\trial_metadata.txt ^
        --out outputs\\embeddings_df21 ^
        --subset eval

    Add --dry-run first to see the report without writing anything.

Then score it:
    python src\\eval.py --emb-root outputs\\embeddings_df21 --split eval ^
                        --model-ckpt outputs\\models\\head_aug.pt
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Each part directory is expected to contain a <part>/eval/ subfolder of shards,
# which is what extract_embeddings.py produces for "--split eval".
INNER = "eval"


# ---------------------------------------------------------------------------
# Reading the parts
# ---------------------------------------------------------------------------
def scan_part(part_dir: Path):
    """Return (shard_stems, ids) for one part. ids is the flat list of trial IDs
    in on-disk order, which is exactly the row order of the embeddings."""
    d = part_dir / INNER
    if not d.is_dir():
        sys.exit(f"FATAL: {d} is not a directory.")

    stems = sorted(s for s in d.glob("shard_*.npy") if ".labels" not in s.name)
    if not stems:
        sys.exit(f"FATAL: no embedding shards in {d}")

    ids = []
    for s in stems:
        files_p = s.with_suffix(".files.txt")
        if not files_p.exists():
            sys.exit(f"FATAL: {s.name} has no .files.txt sidecar in {d}.\n"
                     f"       Without trial IDs the labels cannot be joined. "
                     f"That part must be re-extracted.")
        names = files_p.read_text().split()
        ids.extend(names)
    return stems, ids


def load_key(key_path: Path):
    """trial_metadata.txt -> {trial_id: (label01, subset)}.  1 = spoof."""
    if not key_path.exists():
        sys.exit(f"FATAL: CM key not found: {key_path}")
    table = {}
    with open(key_path) as fh:
        for ln, line in enumerate(fh, 1):
            p = line.split()
            if len(p) < 8:
                continue
            trial, label, subset = p[1], p[5], p[7]
            if label not in ("bonafide", "spoof"):
                sys.exit(f"FATAL: unexpected label {label!r} on key line {ln}. "
                         f"Is this really the DF CM key?")
            table[trial] = (1 if label == "spoof" else 0, subset)
    if not table:
        sys.exit(f"FATAL: parsed 0 rows from {key_path}")
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge DF21 embedding parts and join true labels from the CM key.")
    ap.add_argument("--parts", nargs="+", required=True,
                    help="part directories, each containing an eval/ subfolder")
    ap.add_argument("--key", required=True, help="path to trial_metadata.txt")
    ap.add_argument("--out", required=True,
                    help="output root; shards go to <out>/eval/")
    ap.add_argument("--subset", default="eval",
                    choices=["eval", "progress", "hidden", "all"],
                    help="which CM-key partition to emit (default: eval, the "
                         "partition the published DF21 numbers are quoted on)")
    ap.add_argument("--shard-size", type=int, default=2000,
                    help="rows per output shard (default 2000)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report and verify, but write nothing")
    args = ap.parse_args()

    print("=" * 68)
    print("SONIX  --  DF21 merge + label join")
    print("=" * 68)

    # ---- 1. scan every part -------------------------------------------------
    parts = {}
    for p in args.parts:
        pd = Path(p)
        stems, ids = scan_part(pd)
        parts[pd] = (stems, ids)
        print(f"  {pd.name:14s} {len(stems):5d} shards  {len(ids):7d} rows  "
              f"{ids[0]} .. {ids[-1]}")
    print()

    # ---- 2. find duplicate parts (identical ID sets) ------------------------
    keep, dropped = [], []
    seen_sets = {}
    for pd, (stems, ids) in parts.items():
        sig = (len(ids), ids[0], ids[-1], hash(tuple(ids)))
        if sig in seen_sets:
            dropped.append((pd, seen_sets[sig]))
        else:
            seen_sets[sig] = pd
            keep.append(pd)

    if dropped:
        print("  DUPLICATE PARTS DETECTED (identical trial-ID lists):")
        for dup, orig in dropped:
            print(f"    - {dup.name} is a duplicate of {orig.name}; using "
                  f"{orig.name} once, ignoring {dup.name}")
        print()

    # ---- 3. overlap between the parts we keep -------------------------------
    all_ids = []
    for pd in keep:
        all_ids.extend(parts[pd][1])
    dupe_ids = [i for i, c in Counter(all_ids).items() if c > 1]
    if dupe_ids:
        sys.exit(f"FATAL: {len(dupe_ids)} trial IDs appear in more than one "
                 f"part (e.g. {dupe_ids[:5]}). The parts overlap -- that would "
                 f"double-count clips in the EER. Resolve before merging.")
    print(f"  kept {len(keep)} parts, {len(all_ids)} unique trial IDs, no overlap")

    # ---- 4. join the key ----------------------------------------------------
    key = load_key(Path(args.key))
    print(f"  CM key: {len(key)} trials")

    missing = [i for i in all_ids if i not in key]
    if missing:
        sys.exit(f"FATAL: {len(missing)} embedding trial IDs are absent from the "
                 f"CM key (e.g. {missing[:5]}). Wrong key file, or these "
                 f"embeddings are not DF21. Refusing to guess a label.")
    print("  every embedding trial ID found in the key")

    want = None if args.subset == "all" else args.subset
    sel = [i for i in all_ids if want is None or key[i][1] == want]
    lab_counts = Counter(key[i][0] for i in sel)
    sub_counts = Counter(key[i][1] for i in all_ids)

    print()
    print(f"  partitions present across all parts: {dict(sub_counts)}")
    print(f"  emitting subset={args.subset}: {len(sel)} rows "
          f"(spoof={lab_counts[1]}, bonafide={lab_counts[0]})")
    if lab_counts[0] == 0 or lab_counts[1] == 0:
        sys.exit("FATAL: only one class in the selected subset -- EER is "
                 "undefined. Check --subset.")
    covered = len(all_ids)
    print(f"  coverage of the full DF21 key: {covered}/{len(key)} = "
          f"{100*covered/len(key):.1f}%")
    print()

    if args.dry_run:
        print("dry run -- nothing written.")
        return 0

    # ---- 5. write the merged shards ----------------------------------------
    out_dir = Path(args.out) / INNER
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"FATAL: {out_dir} already exists and is not empty. Delete it "
                 f"first -- refusing to merge into a directory that may hold "
                 f"the earlier collided copy.")
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(sel)
    buf_x, buf_y, buf_f, out_idx, written = [], [], [], 0, 0

    def flush():
        nonlocal buf_x, buf_y, buf_f, out_idx, written
        if not buf_x:
            return
        stem = out_dir / f"shard_{out_idx:05d}"
        np.save(stem.with_suffix(".npy"), np.concatenate(buf_x, 0))
        np.save(Path(str(stem) + ".labels.npy"),
                np.asarray(buf_y, dtype=np.int8))
        Path(str(stem) + ".files.txt").write_text("\n".join(buf_f) + "\n")
        written += len(buf_y)
        out_idx += 1
        buf_x, buf_y, buf_f = [], [], []

    for pd in keep:
        stems, _ = parts[pd]
        for s in stems:
            emb = np.load(s)
            names = s.with_suffix(".files.txt").read_text().split()
            if len(emb) != len(names):
                sys.exit(f"FATAL: {s} has {len(emb)} rows but "
                         f"{len(names)} filenames. Row alignment is broken.")
            mask = np.array([n in wanted for n in names], dtype=bool)
            if not mask.any():
                continue
            buf_x.append(emb[mask])
            buf_y.extend(key[n][0] for n, m in zip(names, mask) if m)
            buf_f.extend(n for n, m in zip(names, mask) if m)
            if sum(len(b) for b in buf_x) >= args.shard_size:
                flush()
        print(f"  merged {pd.name}")
    flush()

    if written != len(sel):
        sys.exit(f"FATAL: wrote {written} rows but selected {len(sel)}. "
                 f"Do not use this output.")

    # ---- 6. the honesty file ------------------------------------------------
    lo, hi = min(all_ids), max(all_ids)
    cov = Path(args.out) / "COVERAGE.txt"
    cov.write_text(
        "SONIX -- ASVspoof 2021 DF merged eval embeddings\n"
        "=" * 60 + "\n"
        f"rows written          : {written}\n"
        f"  spoof               : {lab_counts[1]}\n"
        f"  bonafide            : {lab_counts[0]}\n"
        f"CM-key partition      : {args.subset}\n"
        f"trial ID range        : {lo} .. {hi}\n"
        f"trials covered        : {covered} of {len(key)} in the full DF21 key "
        f"({100*covered/len(key):.1f}%)\n"
        f"source parts used     : {', '.join(p.name for p in keep)}\n"
        f"duplicate parts ignored: "
        f"{', '.join(d.name for d, _ in dropped) if dropped else 'none'}\n"
        "\n"
        "labels were joined by trial ID from the official DF CM key\n"
        f"  {args.key}\n"
        "the placeholder zeros written by extract_embeddings.py --audio-dir\n"
        "were discarded, not trusted.\n"
        "\n"
        "!! ANY EER COMPUTED FROM THIS DIRECTORY COVERS THE TRIAL RANGE ABOVE\n"
        "!! ONLY. IF COVERAGE IS BELOW 100% IT IS **NOT** THE FULL ASVspoof 2021\n"
        "!! DF EER AND MUST NOT BE QUOTED AS ONE. Report it with the coverage.\n"
    )

    print()
    print("=" * 68)
    print(f"  wrote {written} rows in {out_idx} shards -> {out_dir}")
    print(f"  coverage note -> {cov}")
    print("=" * 68)
    print("Next:")
    print(f"  python src\\eval.py --emb-root {args.out} --split eval "
          f"--model-ckpt outputs\\models\\head_aug.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
