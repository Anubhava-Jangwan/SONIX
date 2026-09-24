#!/usr/bin/env python3
"""
make_librisevoc_lists.py  --  SONIX / SIH26104

Seeded, per-vocoder file lists for LibriSeVoc, for extract_embeddings(_v2).py
--file-list.

WHY NOT --limit
    --limit keeps the first N files by sorted name. LibriSpeech names start with
    the speaker id and every vocoder folder uses the same names as gt/, so
    --limit 4200 would give all six vocoders the SAME 4,200 utterances from the
    lowest-numbered speakers, and ~9,000 gt utterances would have no fake twin.
    Two-thirds of the speakers would then only ever appear as real -- a
    "this speaker = real" shortcut of exactly the kind v4 exists to remove.

WHAT THIS DOES
    1. Reads the names in <root>/gt and in each vocoder folder beside it.
       Vocoder files are <utterance>_gen.wav, gt files <utterance>.wav; the
       _gen suffix is ignored for matching, and each list keeps the real name.
    2. Keeps the utterances present in gt AND in every vocoder (the common set).
    3. Shuffles the common set with a fixed seed and lays it end to end
       (perm, perm, perm, ...). Vocoder k (sorted by name) takes slice k of
       length --per-vocoder. A slice never repeats a name, because it is shorter
       than one permutation.
    Result: each vocoder gets a different sample, and every utterance is covered
    roughly evenly across vocoders (6 x 4,200 = 25,200 over 13,201 names means
    about 2 each).

OUTPUT
    <out>/<vocoder>.txt   one filename per line, as it appears in that folder
    <out>/summary.txt     counts, coverage, seed -- commit this with the lists

USAGE (Windows cmd)
    .venv\\Scripts\\python.exe make_librisevoc_lists.py --root F:\\librisevoc\\wav ^
        --per-vocoder 4200 --seed 0 --out docs\\v4_splits\\librisevoc

    then, per vocoder:
    .venv\\Scripts\\python.exe src\\extract_embeddings_v2.py --split train ^
        --audio-dir F:\\librisevoc\\wav\\melgan ^
        --file-list docs\\v4_splits\\librisevoc\\melgan.txt ^
        --layers 5,6,24 --batch 8 --workers 6 ^
        --out D:\\SONIX\\outputs\\embeddings_v2_librisevoc_melgan
"""

import argparse
import collections
import random
import sys
from pathlib import Path

AUDIO = (".wav", ".flac")


GEN = "_gen"   # LibriSeVoc vocoder files are <utterance>_gen.wav; gt is <utterance>.wav


def utt(stem: str) -> str:
    """Utterance id shared by gt and every vocoder: the stem without _gen."""
    return stem[:-len(GEN)] if stem.endswith(GEN) else stem


def names_in(folder: Path) -> dict:
    """utterance id -> filename, for audio files directly inside folder (no
    recursion, matching --audio-dir). Vocoder names carry a _gen suffix that gt
    names do not; both map to the same utterance id."""
    out = {}
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in AUDIO:
            key = utt(f.stem)
            if key in out:
                sys.exit(f"FATAL: {folder.name}/ has two files for utterance "
                         f"{key}: {out[key]} and {f.name}")
            out[key] = f.name
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True,
                    help="folder holding gt/ and the vocoder folders")
    ap.add_argument("--real", default="gt", help="name of the bonafide folder")
    ap.add_argument("--per-vocoder", type=int, default=4200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    real_dir = root / args.real
    if not real_dir.is_dir():
        sys.exit(f"FATAL: no {args.real}/ folder under {root.resolve()}")

    vocoders = sorted(d.name for d in root.iterdir()
                      if d.is_dir() and d.name != args.real)
    if not vocoders:
        sys.exit(f"FATAL: no vocoder folders beside {args.real}/ in {root.resolve()}")

    gt = names_in(real_dir)
    if not gt:
        sys.exit(f"FATAL: no .wav/.flac directly inside {real_dir.resolve()} "
                 f"(nested folders are not read by --audio-dir either)")
    per = {v: names_in(root / v) for v in vocoders}

    for v, m in per.items():
        if not m:
            sys.exit(f"FATAL: no .wav/.flac directly inside {(root / v).resolve()}")
        overlap = len(set(m) & set(gt))
        if overlap == 0:
            sys.exit(f"FATAL: {v}/ shares no names with {args.real}/. "
                     f"e.g. {args.real}: {sorted(gt)[:2]}  {v}: {sorted(m)[:2]}. "
                     f"The names must match to pair real and fake.")

    common = set(gt)
    for m in per.values():
        common &= set(m)
    common = sorted(common)
    n = args.per_vocoder
    if n > len(common):
        sys.exit(f"FATAL: --per-vocoder {n} exceeds the {len(common)} names "
                 f"common to every folder")

    rng = random.Random(args.seed)
    perm = common[:]
    rng.shuffle(perm)
    need = n * len(vocoders)
    tape = perm * (need // len(perm) + 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cover = collections.Counter()
    for k, v in enumerate(vocoders):
        pick = tape[k * n:(k + 1) * n]
        assert len(set(pick)) == len(pick), "a slice repeated a name"
        cover.update(pick)
        (out / f"{v}.txt").write_text(
            "\n".join(per[v][stem] for stem in sorted(pick)) + "\n",
            encoding="utf-8")

    hist = collections.Counter(cover[stem] for stem in common)
    lines = [
        f"root: {root}",
        f"seed: {args.seed}   per_vocoder: {n}   vocoders: {', '.join(vocoders)}",
        f"{args.real}/ names: {len(gt)}   common to all folders: {len(common)}",
    ]
    for v in vocoders:
        lines.append(f"  {v:22s} {len(per[v]):6d} files, {n} selected")
    lines.append("coverage of common names (times used across vocoders):")
    for times in sorted(hist):
        lines.append(f"  {times}x: {hist[times]}")
    lines.append(f"{args.real}/ is extracted in full (no list); only vocoders use lists.")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {len(vocoders)} lists + summary.txt to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
