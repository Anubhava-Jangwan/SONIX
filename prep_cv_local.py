#!/usr/bin/env python3
"""
prep_cv_local.py  --  SONIX / SIH26104        v4, Common Voice gap languages

Turns a Common Voice download (Mozilla Data Collective tarballs, extracted)
into v4-ready 16 kHz wavs for the languages that have FAKES but no REAL audio:
    Malayalam 2,400 fake (IndicSynth + MLAAD)   Urdu 2,400   Odia 1,400
Without real speech in those languages the model can learn "Malayalam = fake".

WHAT IT DOES, PER LANGUAGE
  1. reads validated.tsv (client_id, path, sentence, ...)
  2. speaker split by sha256(client_id): 80% train / 20% eval -- the SAME rule
     as prep_common_voice.py, so a speaker never moves between runs or machines
  3. caps clips per speaker (--per-speaker, default 20) so a few prolific
     volunteers cannot dominate, then takes --per-lang clips, seeded
  4. decodes MP3 -> 16 kHz mono, trims leading/trailing silence (Common Voice
     clips open with 0.5-1 s of room tone that no other corpus has), drops
     anything under 1.0 s after trimming
  5. writes <out>/train/cv_<lang>_<clip>.wav and <out>/eval/..., plus
       cv_manifest.csv       filename, split, recording_id, language,
                             duration_s, speaker_id   (train.py --lang-table)
       cv_train_speakers.txt / cv_eval_speakers.txt   (clone references must
                             NEVER use an eval speaker, or sonix_real speakers)
       mp3_params.txt        the source MP3 codec/bitrate/rate actually seen --
                             make_mp3_match.py must use the same for the fakes

USAGE (Windows cmd, repo root)
    .venv\\Scripts\\python.exe prep_cv_local.py --cv-root data\\common_voice\\raw ^
        --langs ml=2400,ur=2400,or=1400 --out data\\cv_v4

    --cv-root is searched recursively for <lang>/validated.tsv, so pointing it
    at the folder you extracted the tarballs into is enough.

Needs ffmpeg (and ffprobe) on PATH. Resumable: finished wavs are skipped.
"""

import argparse
import csv
import hashlib
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SR = 16000
TRAIN_PCT = 80
MIN_SEC = 1.0


def speaker_split(client_id, train_pct=TRAIN_PCT):
    """Identical to prep_common_voice.py -- do not change one without the other."""
    h = int(hashlib.sha256(client_id.encode()).hexdigest()[:8], 16)
    return "train" if h % 100 < train_pct else "eval"


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
                          "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def trim(x, frame=400, hop=160, below_peak_db=40.0, floor_db=-60.0, pad_s=0.1):
    """Cut leading/trailing frames quieter than max(peak-40 dB, -60 dBFS)."""
    if x.size < frame:
        return x
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    e = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    thr = max(e.max() - below_peak_db, floor_db)
    on = np.flatnonzero(e > thr)
    if on.size == 0:
        return x[:0]
    pad = int(pad_s * SR)
    a = max(0, on[0] * hop - pad)
    b = min(x.size, on[-1] * hop + frame + pad)
    return x[a:b]


def probe(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                              "-show_entries", "stream=codec_name,sample_rate,bit_rate",
                              "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out
    except Exception:
        return "unknown"


def find_lang_dirs(root, langs):
    found = {}
    for tsv in Path(root).rglob("validated.tsv"):
        lang = tsv.parent.name
        if lang in langs:
            if lang in found:
                sys.exit(f"FATAL: two copies of {lang}: {found[lang]} and {tsv.parent}. "
                         f"Keep one Common Voice version.")
            found[lang] = tsv.parent
    missing = [l for l in langs if l not in found]
    if missing:
        sys.exit(f"FATAL: no <lang>/validated.tsv under {root} for {missing}. "
                 f"Extract those tarballs there (the folder name must be the "
                 f"Common Voice code: ml, ur, or, ...).")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="Common Voice gap languages -> v4 wavs")
    ap.add_argument("--cv-root", required=True)
    ap.add_argument("--langs", default="ml=2400,ur=2400,or=1400",
                    help="lang=clips for the TRAIN half; eval takes a quarter of that")
    ap.add_argument("--per-speaker", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="data/cv_v4")
    args = ap.parse_args()

    want = {}
    for tok in args.langs.split(","):
        lang, n = tok.split("=")
        want[lang.strip()] = int(n)
    dirs = find_lang_dirs(args.cv_root, set(want))
    out = Path(args.out)
    for s in ("train", "eval"):
        (out / s).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    jobs, spk = [], {"train": set(), "eval": set()}
    report = []
    for lang, n_train in want.items():
        d = dirs[lang]
        rows = list(csv.DictReader(open(d / "validated.tsv", encoding="utf-8"),
                                   delimiter="\t", quoting=csv.QUOTE_NONE))
        by_half = {"train": defaultdict(list), "eval": defaultdict(list)}
        for r in rows:
            by_half[speaker_split(r["client_id"])][r["client_id"]].append(r["path"])
        for half, n in (("train", n_train), ("eval", max(1, n_train // 4))):
            pool = []
            for cid in sorted(by_half[half]):
                clips = sorted(by_half[half][cid])
                rng.shuffle(clips)
                pool += [(cid, c) for c in clips[: args.per_speaker]]
            rng.shuffle(pool)
            pick = pool[:n]
            n_spk = len({c for c, _ in pick})
            report.append(f"{lang} {half}: {len(pick)} clips from {n_spk} speakers "
                          f"(asked {n}; {len(by_half[half])} speakers available)")
            if len(pick) < n:
                report[-1] += "  <- SHORT: not enough speakers at this cap"
            for cid, clip in pick:
                spk[half].add(cid)
                stem = Path(clip).stem
                jobs.append((lang, half, cid, d / "clips" / clip,
                             out / half / f"cv_{lang}_{stem}.wav"))
    print("\n".join(report))

    def work(j):
        lang, half, cid, src, dst = j
        if dst.exists():
            import soundfile as sf
            return j, sf.info(str(dst)).duration, "skipped"
        try:
            y = trim(decode(src))
            if y.size < MIN_SEC * SR:
                return j, y.size / SR, "too-short"
            import soundfile as sf
            part = str(dst) + ".part"
            sf.write(part, y, SR, subtype="PCM_16", format="WAV")
            Path(part).replace(dst)
            return j, y.size / SR, "ok"
        except Exception as e:                  # noqa: BLE001
            return j, 0.0, f"error: {e}"

    stats, manifest = Counter(), []
    with ThreadPoolExecutor(args.workers) as ex:
        for k, (j, dur, st) in enumerate(ex.map(work, jobs), 1):
            stats[st.split(":")[0]] += 1
            lang, half, cid, src, dst = j
            if st in ("ok", "skipped"):
                manifest.append({"filename": dst.name, "split": half,
                                 "recording_id": dst.stem, "language": lang,
                                 "duration_s": f"{dur:.3f}", "speaker_id": cid})
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)} {dict(stats)}", flush=True)

    with open(out / "cv_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["filename", "split", "recording_id", "language",
                                "duration_s", "speaker_id"])
        w.writeheader()
        w.writerows(sorted(manifest, key=lambda r: r["filename"]))
    for half in ("train", "eval"):
        (out / f"cv_{half}_speakers.txt").write_text(
            "\n".join(sorted(spk[half])) + "\n", encoding="utf-8")
    params = Counter(probe(j[3]) for j in jobs[:: max(1, len(jobs) // 50)])
    (out / "mp3_params.txt").write_text(
        "codec,sample_rate,bit_rate  (count in a sample of the sources)\n" +
        "\n".join(f"{k}  x{v}" for k, v in params.most_common()) + "\n")
    short = sum(float(r["duration_s"]) < 4.0 for r in manifest)
    print(f"done: {dict(stats)}  manifest rows {len(manifest)}  "
          f"under 4 s: {short}")
    print(f"source MP3 params (use for make_mp3_match.py): {params.most_common(3)}")
    if stats.get("error"):
        print("Some clips failed to decode; re-run the same command to retry them.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
