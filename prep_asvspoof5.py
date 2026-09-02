#!/usr/bin/env python3
"""
prep_asvspoof5.py  --  SONIX / SIH26104

Split an extracted ASVspoof 5 partition into bonafide/ and spoof/ folders so the
FOLDER CARRIES THE LABEL, the same pattern make_augment.py uses.

WHY THIS EXISTS
`extract_embeddings.py --audio-dir` writes a PLACEHOLDER label of 0 for every
file (its line 89). ASVspoof 5 dev is ~142k utterances of BOTH classes, so a
single flat folder cannot be stamped with one label afterwards. Split first,
extract each folder separately, stamp each with its own label. Nothing to match
up by hand, nothing to get wrong.

DISK COST: ZERO. Files are HARDLINKED, not copied -- the same bytes appear under
two names on the same NTFS volume. Use --copy only if the audio and the output
are on different drives (hardlinks cannot cross volumes).

PROTOCOL FORMAT (ASVspoof5.dev.track_1.tsv, space-separated)
    col 1  SPEAKER_ID          col 6  CODEC_SEED
    col 2  FLAC_FILE_NAME  <-- col 7  ATTACK_TAG
    col 3  SPEAKER_GENDER      col 8  ATTACK_LABEL
    col 4  CODEC               col 9  KEY  <-- bonafide | spoof
    col 5  CODEC_Q             col 10 TMP
This script does NOT hardcode column 9: it finds the field that is literally
"bonafide" or "spoof". If the layout ever shifts, it still gets the label right.

USAGE (PowerShell, from D:\\SONIX)

    python prep_asvspoof5.py ^
        --protocol  "D:\\ASVspoof5_protocols\\ASVspoof5.dev.track_1.tsv" ^
        --audio-dir "D:\\asvspoof5\\flac_D" ^
        --out-dir   "D:\\asvspoof5_split\\dev"

THEN

    python src\\extract_embeddings.py --split train ^
        --audio-dir "D:\\asvspoof5_split\\dev\\bonafide" ^
        --out outputs\\embeddings_as5dev_bonafide --batch 8
    python src\\extract_embeddings.py --split train ^
        --audio-dir "D:\\asvspoof5_split\\dev\\spoof" ^
        --out outputs\\embeddings_as5dev_spoof --batch 8

    python stamp_labels.py --emb-dir outputs\\embeddings_as5dev_bonafide\\train --label 0
    python stamp_labels.py --emb-dir outputs\\embeddings_as5dev_spoof\\train    --label 1
    python stamp_labels.py --emb-dir outputs\\embeddings_as5dev_spoof\\train    --check

NOTE ON "--split train"
That is just the SUBFOLDER NAME extract_embeddings.py writes into, and the name
train.py looks for under an --extra-emb-root. It does not mean this data is the
ASVspoof 5 train partition. Using the dev partition for training is fine and
leaks nothing, as long as you keep the EVAL partition purely for testing.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

LABELS = {"bonafide", "spoof"}


def load_protocol(path):
    """-> {flac_stem: 'bonafide'|'spoof'}. Label found by value, not by index."""
    table = {}
    malformed = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                malformed += 1
                continue
            name = parts[1]
            label = next((p for p in parts if p in LABELS), None)
            if label is None:
                malformed += 1
                continue
            table[Path(name).stem] = label
    if malformed:
        print(f"  ! {malformed} protocol lines had no bonafide/spoof field; skipped")
    if not table:
        sys.exit(f"FATAL: no usable rows in {path}.\n"
                 f"Is this a track_1 CM protocol? track_2 files are ASV trials "
                 f"and do not carry a CM key.")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True,
                    help="ASVspoof5.dev.track_1.tsv (or the train .tsv)")
    ap.add_argument("--audio-dir", required=True,
                    help="folder holding the extracted .flac (searched recursively)")
    ap.add_argument("--out-dir", required=True,
                    help="<out>/bonafide and <out>/spoof are created")
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of hardlink (needed only across volumes; "
                         "costs full disk space)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N files")
    args = ap.parse_args()

    proto = load_protocol(args.protocol)
    print(f"protocol rows      : {len(proto)}")
    print(f"  bonafide         : {sum(1 for v in proto.values() if v == 'bonafide')}")
    print(f"  spoof            : {sum(1 for v in proto.values() if v == 'spoof')}")

    src_root = Path(args.audio_dir)
    if not src_root.exists():
        sys.exit(f"FATAL: --audio-dir not found: {src_root}")

    print("scanning audio ...", flush=True)
    files = sorted(src_root.rglob("*.flac"))
    if not files:
        sys.exit(f"FATAL: no .flac found under {src_root}. Did the tar extract?")
    if args.limit:
        files = files[: args.limit]
    print(f"audio files found  : {len(files)}")

    out_root = Path(args.out_dir)
    for sub in ("bonafide", "spoof"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    linked = Counter()
    skipped_existing = 0
    unlabelled = []
    errors = []

    for i, f in enumerate(files, 1):
        label = proto.get(f.stem)
        if label is None:
            unlabelled.append(f.name)
            continue
        dst = out_root / label / f.name
        if dst.exists():
            skipped_existing += 1
            continue
        try:
            if args.copy:
                import shutil
                shutil.copy2(f, dst)
            else:
                os.link(f, dst)
            linked[label] += 1
        except OSError as exc:
            errors.append((f.name, str(exc)))
            if len(errors) == 1 and not args.copy:
                print(f"\n  ! hardlink failed: {exc}")
                print(f"  ! if the audio and --out-dir are on DIFFERENT drives, "
                      f"rerun with --copy\n")
        if i % 20000 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    print()
    print(f"linked bonafide    : {linked['bonafide']}")
    print(f"linked spoof       : {linked['spoof']}")
    if skipped_existing:
        print(f"already present    : {skipped_existing}")
    if unlabelled:
        print(f"\n!! {len(unlabelled)} audio files are NOT in the protocol and were "
              f"SKIPPED")
        print(f"   e.g. {unlabelled[:5]}")
        print("   An unlabelled clip must never be written -- that is exactly the "
              "bug this script exists to prevent.")
    if errors:
        print(f"\n!! {len(errors)} link/copy errors, e.g. {errors[:3]}")

    missing = len(proto) - (linked['bonafide'] + linked['spoof'] + skipped_existing)
    if missing > 0:
        print(f"\nNote: {missing} protocol entries had no matching audio file. "
              f"Expected if you have only some of the archives -- the split is "
              f"still correct for what you do have.")

    print()
    print(f"  {out_root / 'bonafide'}   -> extract, then stamp_labels.py --label 0")
    print(f"  {out_root / 'spoof'}      -> extract, then stamp_labels.py --label 1")
    if errors:
        print("\nNON-ZERO ERRORS -- resolve before extracting embeddings.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
