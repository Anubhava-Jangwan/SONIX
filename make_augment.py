#!/usr/bin/env python3
"""
make_augment.py  --  SONIX / SIH26104

Build a real-world-sounding copy of the ASVspoof-2019 LA train set, using
room impulse responses (RIRS_NOISES) and additive noise (MUSAN).

WHAT THIS DOES
    For each clean training clip, produce ONE augmented copy that sounds like
    it was recorded on a real device in a real place:

        reverb   : convolve with a random room impulse response
        + noise  : mix in a random MUSAN clip at a random SNR
                   (noise / music / babble, chosen at random)

    The label NEVER changes. A bonafide clip recorded in a noisy room is still
    bonafide; a spoof clip is still spoof.

THE LABEL TRAP THIS AVOIDS
    extract_embeddings.py --audio-dir writes a PLACEHOLDER label of 0 for every
    file (see its line 89). The train split contains BOTH classes, so a single
    flat output folder cannot be stamped with one label afterwards.

    So this script splits its output by label, using the official train
    protocol:

        <out>/bonafide/*.wav      -> extract, then stamp_labels.py --label 0
        <out>/spoof/*.wav         -> extract, then stamp_labels.py --label 1

    The folder carries the label. Same pattern the team already uses for the
    real/ and fake/ recordings. Nothing to match up by hand.

    Both folders are FLAT, because extract_embeddings.py --audio-dir globs
    "*.wav" non-recursively and would silently find nothing in a nested tree.

WHAT NEVER GETS EMBEDDED
    The RIR files and the MUSAN files themselves. They are ingredients, not
    training data. Embedding them produces vectors of a room echo or of traffic
    noise, labelled "genuine human voice". Do not do it.

USAGE (PowerShell, from the repo root D:\\SONIX)

    # 1. smoke test on 20 clips -- ALWAYS do this first, then listen to them
    python make_augment.py ^
        --audio-dir "data\\asvspoof19_la\\ASVspoof2019_LA_train\\flac" ^
        --protocol  "data\\asvspoof19_la\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt" ^
        --rir-root  "D:\\RIRS_NOISES" ^
        --musan-root "D:\\musan" ^
        --out-dir   "outputs\\audio_aug\\train_smoke" ^
        --limit 20

    # 2. the full run (CPU only, no GPU needed)
    python make_augment.py ^
        --audio-dir "data\\asvspoof19_la\\ASVspoof2019_LA_train\\flac" ^
        --protocol  "data\\asvspoof19_la\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt" ^
        --rir-root  "D:\\RIRS_NOISES" ^
        --musan-root "D:\\musan" ^
        --out-dir   "outputs\\audio_aug\\train" ^
        --workers 8

    # 3. embed each label folder SEPARATELY (this is the GPU step)
    python src\\extract_embeddings.py --split train ^
        --audio-dir "outputs\\audio_aug\\train\\bonafide" ^
        --out outputs\\embeddings_aug_bonafide --batch 8
    python src\\extract_embeddings.py --split train ^
        --audio-dir "outputs\\audio_aug\\train\\spoof" ^
        --out outputs\\embeddings_aug_spoof --batch 8

    # 4. stamp the correct label on each
    python stamp_labels.py --emb-dir outputs\\embeddings_aug_bonafide\\train --label 0
    python stamp_labels.py --emb-dir outputs\\embeddings_aug_spoof\\train    --label 1

    # 5. VERIFY before training -- expect bonafide(0)=0 spoof(1)=N on the spoof dir
    python stamp_labels.py --emb-dir outputs\\embeddings_aug_spoof\\train --check

    # 6. retrain
    python src\\train.py --emb-root outputs\\embeddings ^
        --extra-emb-root outputs\\embeddings_aug_bonafide ^
        --extra-emb-root outputs\\embeddings_aug_spoof ^
        --out outputs\\models\\head_robust.pt

    (If --extra-emb-root only accepts ONE value, see the note at the bottom.)

TRAIN ONLY. Never augment eval, DF21, or In-the-Wild.
"""

import argparse
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

SR = 16000
AUDIO_EXTS = {".wav", ".flac"}

# Kaldi-style SNR ranges, in dB. Lower = noisier = harder.
SNR_RANGES = {
    "noise":  (0.0, 15.0),
    "music":  (5.0, 15.0),
    "babble": (13.0, 20.0),
}
BABBLE_SPEAKERS = (3, 7)


# ---------------------------------------------------------------------------
# audio io
# ---------------------------------------------------------------------------

def load_mono_16k(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR and len(x) > 1:
        n_out = int(round(len(x) * SR / sr))
        x = np.interp(
            np.linspace(0.0, len(x) - 1, n_out),
            np.arange(len(x)),
            x.astype(np.float64),
        ).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def rms(x):
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def fit_length(n, target):
    """Tile-or-crop a noise clip to exactly `target` samples, from a random offset."""
    if n.size == 0:
        return np.zeros(target, dtype=np.float32)
    if n.size < target:
        reps = int(np.ceil(target / n.size))
        n = np.tile(n, reps)
    start = random.randrange(0, max(1, n.size - target + 1))
    return n[start:start + target]


# ---------------------------------------------------------------------------
# the two augmentations
# ---------------------------------------------------------------------------

def prepare_rir(h):
    """
    Trim to the direct-path peak so convolution adds room, not delay.
    Unit-energy so levels stay sane. Tail capped at 1 s.
    """
    if h.size == 0:
        return None
    peak = int(np.argmax(np.abs(h)))
    h = h[max(0, peak - 32): max(0, peak - 32) + SR]
    e = float(np.sqrt(np.sum(h.astype(np.float64) ** 2)))
    if e < 1e-8 or not np.isfinite(e):
        return None
    return (h / e).astype(np.float32)


def apply_reverb(x, h):
    """Convolve, trim to original length, restore the original loudness."""
    if x.size == 0:
        return x
    before = rms(x)
    y = fftconvolve(x, h, mode="full")[: len(x)]
    after = rms(y)
    if after > 1e-8 and before > 1e-8:
        y = y * (before / after)
    return y.astype(np.float32)


def add_noise(x, noise, snr_db):
    """
    Mix `noise` under `x` so that speech is `snr_db` decibels louder.
    Returns the mixture at the original speech loudness.
    """
    if x.size == 0:
        return x
    s_rms = rms(x)
    n = fit_length(noise, len(x))
    n_rms = rms(n)
    if s_rms < 1e-8 or n_rms < 1e-8:
        return x
    # scale noise so that 20*log10(s_rms / scaled_n_rms) == snr_db
    target_n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    y = x + n * (target_n_rms / n_rms)
    out = rms(y)
    if out > 1e-8:
        y = y * (s_rms / out)
    return y.astype(np.float32)


def make_babble(speech_files, target_len, rng):
    """Sum several unrelated speakers into an unintelligible crowd murmur."""
    k = rng.randint(*BABBLE_SPEAKERS)
    acc = np.zeros(target_len, dtype=np.float32)
    used = 0
    for _ in range(k):
        p = speech_files[rng.randrange(len(speech_files))]
        try:
            s = load_mono_16k(p)
        except Exception:
            continue
        if s.size == 0:
            continue
        acc += fit_length(s, target_len)
        used += 1
    return acc if used else None


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

_CACHE = {}


def _cached_rir(p):
    h = _CACHE.get(("rir", p))
    if h is None:
        h = prepare_rir(load_mono_16k(p))
        _CACHE[("rir", p)] = h
    return h


def process_one(job):
    src, dst, rir_path, kind, noise_path, snr, seed, speech_pool = job
    try:
        rng = random.Random(seed)
        random.seed(seed)  # fit_length uses the module rng

        x = load_mono_16k(src)
        if x.size == 0:
            return (src, "empty-source")

        if rir_path is not None:
            h = _cached_rir(rir_path)
            if h is not None:
                x = apply_reverb(x, h)

        if kind == "babble":
            n = make_babble(speech_pool, len(x), rng)
        elif noise_path is not None:
            n = load_mono_16k(noise_path)
        else:
            n = None

        if n is not None:
            x = add_noise(x, n, snr)

        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0.99:
            x = x * (0.99 / peak)

        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, x, SR, subtype="PCM_16")
        return (src, "ok")
    except Exception as exc:  # noqa: BLE001
        return (src, f"error: {exc}")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_protocol(path):
    """
    ASVspoof2019.LA.cm.train.trn.txt
      col0 = speaker, col1 = filename, col4 = bonafide|spoof
    Returns {filename_stem: 0|1}.  1 = spoof.
    """
    table = {}
    bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            parts = line.split()
            if len(parts) < 5:
                bad += 1
                continue
            name, label = parts[1], parts[4]
            if label not in ("bonafide", "spoof"):
                bad += 1
                continue
            table[name] = 1 if label == "spoof" else 0
    if bad:
        print(f"  ! {bad} malformed protocol lines skipped")
    if not table:
        sys.exit(f"FATAL: no usable rows in protocol {path}")
    return table


def collect(root, patterns, what):
    root = Path(root)
    if not root.exists():
        sys.exit(f"FATAL: {what} root not found: {root}")
    out = []
    for pat in patterns:
        out.extend(str(p) for p in root.glob(pat))
    out = sorted(set(out))
    if not out:
        sys.exit(f"FATAL: no {what} files matched {patterns} under {root}")
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True, help="folder of clean train clips")
    ap.add_argument("--protocol", required=True,
                    help="ASVspoof2019.LA.cm.train.trn.txt -- decides bonafide/ vs spoof/")
    ap.add_argument("--rir-root", default=None, help="unpacked RIRS_NOISES folder")
    ap.add_argument("--musan-root", default=None, help="unpacked musan folder")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rir-glob", default="simulated_rirs/**/*.wav")
    ap.add_argument("--p-reverb", type=float, default=0.8,
                    help="fraction of clips that get reverb (default 0.8)")
    ap.add_argument("--p-noise", type=float, default=0.8,
                    help="fraction of clips that get additive noise (default 0.8)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.rir_root and not args.musan_root:
        sys.exit("FATAL: give at least one of --rir-root / --musan-root.")

    src_root = Path(args.audio_dir)
    if not src_root.exists():
        sys.exit(f"FATAL: --audio-dir not found: {src_root}")
    files = sorted(p for p in src_root.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"FATAL: no .wav/.flac directly inside {src_root}")

    proto = load_protocol(args.protocol)

    rirs = collect(args.rir_root, [args.rir_glob], "RIR") if args.rir_root else []
    noise_files = music_files = speech_files = []
    if args.musan_root:
        noise_files = collect(args.musan_root, ["noise/**/*.wav"], "MUSAN noise")
        music_files = collect(args.musan_root, ["music/**/*.wav"], "MUSAN music")
        speech_files = collect(args.musan_root, ["speech/**/*.wav"], "MUSAN speech")

    print(f"source clips   : {len(files)}")
    print(f"RIRs           : {len(rirs)}")
    print(f"MUSAN noise    : {len(noise_files)}")
    print(f"MUSAN music    : {len(music_files)}")
    print(f"MUSAN speech   : {len(speech_files)}  (babble only -- never used as bonafide)")
    print(f"output         : {args.out_dir}\\{{bonafide,spoof}}")

    rng = random.Random(args.seed)
    out_root = Path(args.out_dir)
    kinds = [k for k, pool in (("noise", noise_files), ("music", music_files),
                               ("babble", speech_files)) if pool]

    jobs, skipped, unlabelled = [], 0, []
    label_counts = Counter()
    for i, f in enumerate(files):
        lab = proto.get(f.stem)
        if lab is None:
            unlabelled.append(f.name)
            continue
        label_counts[lab] += 1
        sub = "spoof" if lab == 1 else "bonafide"
        dst = out_root / sub / (f.stem + ".wav")

        rir = rirs[rng.randrange(len(rirs))] if (rirs and rng.random() < args.p_reverb) else None
        kind = noise_path = None
        snr = 0.0
        if kinds and rng.random() < args.p_noise:
            kind = kinds[rng.randrange(len(kinds))]
            snr = rng.uniform(*SNR_RANGES[kind])
            if kind == "noise":
                noise_path = noise_files[rng.randrange(len(noise_files))]
            elif kind == "music":
                noise_path = music_files[rng.randrange(len(music_files))]

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append((str(f), str(dst), rir, kind, noise_path, snr,
                     args.seed + i, speech_files))

    if unlabelled:
        print(f"\n!! {len(unlabelled)} clips are NOT in the protocol and were SKIPPED")
        print(f"   e.g. {unlabelled[:5]}")
        print("   A clip with no label must never be written -- that is the bug "
              "this script exists to prevent.")
    print(f"\nlabelled: bonafide={label_counts[0]}  spoof={label_counts[1]}")
    if skipped:
        print(f"skipping {skipped} already-done files (--overwrite to redo)")
    if not jobs:
        print("nothing to do.")
        return

    done = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_one, j) for j in jobs]
        for fut in as_completed(futs):
            src, status = fut.result()
            done += 1
            if status != "ok":
                errors += 1
                if errors <= 20:
                    print(f"  ! {Path(src).name}: {status}")
            if done % 500 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    print(f"\ndone. wrote {done - errors} files, {errors} errors.")
    print(f"  {out_root / 'bonafide'}   -> extract, then stamp_labels.py --label 0")
    print(f"  {out_root / 'spoof'}      -> extract, then stamp_labels.py --label 1")
    if errors:
        print("NON-ZERO ERRORS -- resolve before extracting embeddings.")


if __name__ == "__main__":
    main()
