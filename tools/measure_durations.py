#!/usr/bin/env python3
"""
measure_durations.py  --  SONIX / SIH26104

Clip durations for train.py --durations (the padding-shortcut check and the
--min-dur 4.0 filter). Reads audio HEADERS only, so it is CPU-light and safe to
run next to a GPU extraction.

Same file rule as extract_embeddings.py --audio-dir: .wav/.flac directly inside
each folder, no recursion.

    python tools/measure_durations.py --audio-dir <folder> [--audio-dir <folder> ...] \
        --out docs/v4_splits/durations_<corpus>.csv

Output columns: file, duration   (seconds). Unreadable files are listed at the
end and left out -- never given a made-up duration.
"""
import argparse
import csv
import sys
import wave
from pathlib import Path

AUDIO = (".wav", ".flac")


def seconds(p: Path):
    try:
        import soundfile as sf
        i = sf.info(str(p))
        return i.frames / float(i.samplerate)
    except ImportError:
        if p.suffix.lower() != ".wav":
            raise
        with wave.open(str(p)) as w:
            return w.getnframes() / float(w.getframerate())


def main() -> int:
    ap = argparse.ArgumentParser(description="clip durations for train.py --durations")
    ap.add_argument("--audio-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, bad = [], []
    for d in args.audio_dir:
        d = Path(d)
        if not d.is_dir():
            sys.exit(f"FATAL: {d} is not a folder")
        files = sorted(f for f in d.iterdir()
                       if f.is_file() and f.suffix.lower() in AUDIO)
        print(f"  {d}: {len(files)} files", flush=True)
        for k, f in enumerate(files, 1):
            try:
                rows.append((f.name, round(seconds(f), 3)))
            except Exception as e:           # noqa: BLE001 -- report, never guess
                bad.append((f.name, repr(e)[:120]))
            if k % 5000 == 0:
                print(f"    {k}/{len(files)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "duration"])
        w.writerows(rows)

    short = sum(1 for _, s in rows if s < 4.0)
    print(f"wrote {len(rows)} rows -> {out}   under 4.0 s: {short} "
          f"({100 * short / max(len(rows), 1):.1f}%)   unreadable: {len(bad)}")
    for name, err in bad[:10]:
        print(f"  UNREADABLE {name}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
