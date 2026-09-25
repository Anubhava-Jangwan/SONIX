#!/usr/bin/env python3
"""
make_mp3_match.py  --  SONIX / SIH26104        v4, Common Voice companion

Common Voice audio is MP3 at the source and it is all REAL. Added alone, "MP3
artifacts" would mean "real" in Malayalam, Urdu and Odia. This gives the FAKES
of those languages the same MP3 round trip, at the parameters Common Voice
actually uses (prep_cv_local.py writes them to mp3_params.txt), so MP3 sits on
both sides of the label.

Only files whose name says one of --langs are processed:
    indicsynth_Malayalam_00012 -> ml     mlaad_ur_<model>_000031 -> ur
Output: 16 kHz mono wav, SAME stem, full length (the extractor crops later).

At training the clean copies of those languages are dropped with
    --drop-langs embeddings_v2_indicsynth=ml,ur,or   (and _cond, and mlaad)
and the MP3 roots come in as embeddings_v2_indicsynth_mp3 / embeddings_v2_mlaad_mp3.

USAGE (Windows cmd, repo root, on the box with the fake audio)
    python make_mp3_match.py --audio-dir F:\\indicsynth --langs ml,ur,or ^
        --bitrate 64k --encode-sr 48000 --out D:\\SONIX\\mp3match\\indicsynth
    python make_mp3_match.py --audio-dir D:\\SONIX\\data\\mlaad_indic --langs ml,ur ^
        --bitrate 64k --encode-sr 48000 --out D:\\SONIX\\mp3match\\mlaad

Needs ffmpeg with libmp3lame (every normal build has it). Resumable.
"""

import argparse
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AUDIO = (".wav", ".flac")
PREFIXES = {"aug", "indicsynth", "mlaad", "mms", "tts", "cv"}
LANG = {
    "ml": "ml", "mal": "ml", "malayalam": "ml",
    "ur": "ur", "urd": "ur", "urdu": "ur",
    "or": "or", "ory": "or", "ori": "or", "odia": "or", "oriya": "or",
    "sa": "sa", "san": "sa", "sanskrit": "sa",
    "hi": "hi", "hin": "hi", "hindi": "hi", "bn": "bn", "ben": "bn",
    "bengali": "bn", "ta": "ta", "tam": "ta", "tamil": "ta", "te": "te",
    "tel": "te", "telugu": "te", "mr": "mr", "mar": "mr", "marathi": "mr",
    "gu": "gu", "guj": "gu", "gujarati": "gu", "kn": "kn", "kan": "kn",
    "kannada": "kn", "pa": "pa", "pan": "pa", "punjabi": "pa",
}


def lang_of(stem):
    toks = stem.split("_")
    i = 0
    while i < len(toks) - 1 and toks[i].lower() in PREFIXES:
        i += 1
    return LANG.get(toks[i].lower())


def roundtrip(src, dst, bitrate, encode_sr):
    base = ["ffmpeg", "-v", "error", "-nostdin"]
    mp3 = subprocess.run(base + ["-i", str(src), "-ac", "1", "-ar", str(encode_sr),
                                 "-c:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3",
                                 "pipe:1"], capture_output=True, check=True).stdout
    part = str(dst) + ".part"
    subprocess.run(base + ["-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
                           "-c:a", "pcm_s16le", "-f", "wav", part],
                   input=mp3, capture_output=True, check=True)
    Path(part).replace(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="MP3-match fakes to Common Voice")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--langs", required=True, help="e.g. ml,ur,or")
    ap.add_argument("--bitrate", required=True, help="from mp3_params.txt, e.g. 64k")
    ap.add_argument("--encode-sr", type=int, required=True,
                    help="from mp3_params.txt, e.g. 48000 or 32000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    want = {LANG.get(l.strip().lower(), l.strip().lower()) for l in args.langs.split(",")}
    src = Path(args.audio_dir)
    files = sorted(f for f in src.iterdir()
                   if f.is_file() and f.suffix.lower() in AUDIO)
    picked = [f for f in files if lang_of(f.stem) in want]
    unknown = [f.name for f in files if lang_of(f.stem) is None]
    if unknown:
        sys.exit(f"FATAL: {len(unknown)} names carry no known language "
                 f"(e.g. {unknown[:3]}). Nothing guessed.")
    by = Counter(lang_of(f.stem) for f in picked)
    print(f"{len(picked)} of {len(files)} files in {sorted(want)}: {dict(by)}")
    if not picked:
        sys.exit("FATAL: nothing matched --langs")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def work(f):
        dst = out / (f.stem + ".wav")
        if dst.exists():
            return "skipped"
        try:
            roundtrip(f, dst, args.bitrate, args.encode_sr)
            return "ok"
        except subprocess.CalledProcessError as e:
            return "error: " + (e.stderr or b"").decode("utf-8", "replace")[:120]

    st = Counter()
    with ThreadPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(work, picked), 1):
            st[r.split(":")[0]] += 1
            if k % 1000 == 0:
                print(f"  {k}/{len(picked)} {dict(st)}", flush=True)
    (out / "mp3_match.txt").write_text(
        f"source {src.resolve()}\nlangs {sorted(want)}\nbitrate {args.bitrate}\n"
        f"encode_sr {args.encode_sr}\ncounts {dict(by)}\n")
    print(f"done {dict(st)} -> {out}")
    if st.get("error"):
        print("Re-run the same command to retry failed files.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
