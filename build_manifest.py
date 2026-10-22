#!/usr/bin/env python3
"""Build the v4 manifest for a real/ + fake/ audio root, from filenames alone.

    <root>/real/<recording_id>__<lang>__<channel>__real.wav      label 0
    <root>/fake/<recording_id>__<lang>__<channel>__<tool>.wav    label 1

recording_id is SHARED between a clip and its clone. That is what makes the
holdout recording-level (V4_BUILD_BRIEF §2.3) and what pairs real to fake.

There is deliberately no metadata CSV to maintain: a sidecar drifts from disk
within a week, the filename cannot. Rename the file, the manifest follows.

    python build_manifest.py --root sonix_real/sonix_real
    python build_manifest.py --root sonix_real/sonix_real \
        --speakers-out sonix_real/speakers.txt
    python build_manifest.py --root sonix_real/sonix_real \
        --check-disjoint data/cv_train_speakers.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ("path", "recording_id", "corpus", "language",
          "generator_family", "channel", "label", "split")

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

# Closed vocabularies. A typo that invents a new language or channel silently
# splits a group in every per-group metric downstream, so fail loudly instead.
LANGS = {"hi", "bn", "ta", "te", "mr", "kn", "ml", "ur",
         "gu", "pa", "or", "sa", "as", "en"}
CHANNELS = {"phone", "laptop", "studio", "broadcast", "clean", "codec"}


def parse_stem(stem: str):
    """-> (recording_id, lang, channel, generator) or raise ValueError."""
    parts = stem.split("__")
    if len(parts) != 4:
        raise ValueError(f"expected 4 '__'-separated fields, got {len(parts)}")
    rid, lang, chan, gen = (p.strip().lower() for p in parts)
    if not rid:
        raise ValueError("empty recording_id")
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r} (allowed: {sorted(LANGS)})")
    if chan not in CHANNELS:
        raise ValueError(f"unknown channel {chan!r} (allowed: {sorted(CHANNELS)})")
    if not gen:
        raise ValueError("empty generator")
    return rid, lang, chan, gen


def speaker_of(recording_id: str) -> str:
    """Trailing take number is not part of the speaker identity."""
    head, _, tail = recording_id.rpartition("_")
    return head if head and tail.isdigit() else recording_id


def scan(root: Path, corpus: str, split: str, legacy_ok: bool):
    rows, errors, legacy = [], [], 0
    for sub, label in (("real", 0), ("fake", 1)):
        d = root / sub
        if not d.is_dir():
            sys.exit(f"FATAL: missing {d}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in AUDIO_EXT:
                continue
            try:
                rid, lang, chan, gen = parse_stem(p.stem.lower())
            except ValueError as e:
                if not legacy_ok:
                    errors.append(f"{sub}/{p.name}: {e}")
                    continue
                # Provenance was never recorded for these. Do not guess it.
                rid, lang, chan, gen = p.stem.lower(), "unknown", "unknown", "unknown"
                legacy += 1
            else:
                if label == 0 and gen != "real":
                    errors.append(f"real/{p.name}: generator must be 'real', got {gen!r}")
                    continue
                if label == 1 and gen == "real":
                    errors.append(f"fake/{p.name}: generator 'real' in fake/")
                    continue
            rows.append({
                "path": p.as_posix(),
                "recording_id": rid,
                "corpus": corpus,
                "language": lang,
                "generator_family": "bonafide" if label == 0 else gen,
                "channel": chan,
                "label": label,
                "split": split,
            })
    return rows, errors, legacy


def report(rows):
    """Pair coverage and per-group counts -- the numbers that say whether the
    set is big enough to make a claim with."""
    by_rid = defaultdict(set)
    for r in rows:
        by_rid[r["recording_id"]].add(r["label"])
    paired = sorted(k for k, v in by_rid.items() if v == {0, 1})
    real_only = sorted(k for k, v in by_rid.items() if v == {0})
    fake_only = sorted(k for k, v in by_rid.items() if v == {1})

    print(f"clips            {len(rows)}")
    print(f"matched pairs    {len(paired)}")
    print(f"speakers         {len({speaker_of(r['recording_id']) for r in rows})}")
    if real_only:
        print(f"  ! real with no clone  ({len(real_only)}): {', '.join(real_only[:8])}")
    if fake_only:
        print(f"  ! clone with no real  ({len(fake_only)}): {', '.join(fake_only[:8])}")

    for col in ("language", "generator_family", "channel"):
        counts = Counter(r[col] for r in rows)
        print(f"\n{col}")
        for k, n in counts.most_common():
            print(f"  {k:<16} {n}")
    return len(paired)


def check_durations(rows, min_s=2.0):
    """Header-only read. A truncated clip shorter than the 4 s window gets
    repeat-padded at scoring time, which distorts the score rather than failing."""
    try:
        import soundfile as sf
    except ImportError:
        print("\n(soundfile not installed -- skipped duration check)")
        return
    short = []
    for r in rows:
        try:
            info = sf.info(r["path"])
            if info.frames / info.samplerate < min_s:
                short.append((r["path"], info.frames / info.samplerate))
        except Exception as e:
            short.append((r["path"], f"UNREADABLE: {e}"))
    if short:
        print(f"\n! {len(short)} clip(s) under {min_s}s or unreadable:")
        for p, d in short[:10]:
            print(f"  {d if isinstance(d, str) else f'{d:.2f}s':<12} {p}")


def speakers_from_manifest(manifest_csv) -> list[str]:
    """Speaker list straight off a written manifest -- for gate checks that run
    after the build, without rescanning the audio tree."""
    with open(manifest_csv, newline="", encoding="utf-8") as fh:
        return sorted({speaker_of(r["recording_id"]) for r in csv.DictReader(fh)})


def assert_disjoint(speakers, other_path) -> list[str]:
    """-> the overlapping speakers, empty if clean. Caller decides how to fail.

    Import this from verify_v4.py rather than shelling out, so the gate cannot
    be skipped by a missing subprocess or a swallowed exit code.
    """
    other = {l.split("	")[0].strip().lower()
             for l in Path(other_path).read_text(encoding="utf-8").splitlines()
             if l.strip()}
    return sorted({s.lower() for s in speakers} & other)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder containing real/ and fake/")
    ap.add_argument("--corpus", default="sonix_real")
    ap.add_argument("--split", default="test",
                    help="sonix_real is held out; there is no reason to change this")
    ap.add_argument("--out", default="")
    ap.add_argument("--speakers-out", default="",
                    help="write the speaker list, for the disjointness handoff")
    ap.add_argument("--check-disjoint", default="",
                    help="speaker list that must NOT intersect this set "
                         "(e.g. the training-clone speakers)")
    ap.add_argument("--legacy-ok", action="store_true",
                    help="accept filenames that predate the convention; their "
                         "metadata is written as 'unknown', never guessed")
    args = ap.parse_args()

    root = Path(args.root)
    rows, errors, legacy = scan(root, args.corpus, args.split, args.legacy_ok)
    if errors:
        print(f"FATAL: {len(errors)} filename(s) rejected:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    if not rows:
        sys.exit(f"FATAL: no audio under {root.resolve()}")
    if legacy:
        print(f"! {legacy} legacy filename(s): language/channel/generator = 'unknown'.\n"
              f"  These cannot train the generator-ID head.\n")

    n_pairs = report(rows)
    check_durations(rows)

    speakers = sorted({speaker_of(r["recording_id"]) for r in rows})
    if args.check_disjoint:
        clash = assert_disjoint(speakers, args.check_disjoint)
        if clash:
            print(f"\nFATAL: {len(clash)} speaker(s) overlap {args.check_disjoint}: "
                  f"{', '.join(clash)}\n  {args.corpus} is no longer held out.",
                  file=sys.stderr)
            sys.exit(2)
        print(f"\ndisjoint from {args.check_disjoint}: OK ({len(speakers)} speakers checked)")
    if args.speakers_out:
        Path(args.speakers_out).write_text("\n".join(speakers) + "\n")
        print(f"\nwrote {args.speakers_out}  ({len(speakers)} speakers)")

    out = Path(args.out or root / "manifest.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows, {n_pairs} pairs)")
    if n_pairs < 200:
        print(f"  target is 200+ matched pairs; {200 - n_pairs} short.")


if __name__ == "__main__":
    main()
