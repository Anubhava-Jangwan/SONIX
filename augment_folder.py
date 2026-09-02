"""Channel-augment ANY folder of wavs. No protocol, no dataset assumptions.

WHY THIS EXISTS
make_augment.py and make_codec.py both require an ASVspoof protocol file, so
neither can be pointed at a folder of Indic spoofs. This one takes a folder in
and writes a folder out.

WHY IT MATTERS MORE THAN IT LOOKS
Our Indic spoofs (MLAAD, IndicSynth, MMS-TTS) are clean studio-quality synthesis.
Our Indic bonafide (IndicVoices) is varied real-world recording -- phones, rooms,
background noise. Train on them as-is and the head learns "clean Hindi = fake"
rather than "synthetic Hindi = fake". That is the SAME shortcut we already
diagnosed once (the model had learned "clean background = fake" from ASVspoof's
-75 dBFS digital silence), rebuilt in a new language. The Bachchan clip would
fail again for a brand-new reason.

Augmenting the spoofs so both classes span the same channel conditions is what
stops that.

Everything is synthesised in numpy -- no RIRS_NOISES download, no MUSAN, no
ffmpeg required. If you DO have RIR/MUSAN unpacked, point at them and real
impulse responses and real noise are used instead.

    python augment_folder.py --in data/mlaad_indic --out data/mlaad_indic_aug
    python augment_folder.py --in data/indicsynth  --out data/indicsynth_aug
    python augment_folder.py --in data/indic_spoof --out data/indic_spoof_aug

Then extract the AUGMENTED folder as well as the clean one, and stamp both
--label 1.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3")
TARGET_SR = 16000


# ---------------------------------------------------------------- G.711
def mulaw_roundtrip(x):
    """G.711 mu-law encode + decode: the 8-bit companding a phone line applies.
    Pure numpy -- the same transform ffmpeg's pcm_mulaw performs."""
    mu = 255.0
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) * 127.5)                  # 8-bit quantise
    y = q / 127.5 - 1.0
    return (np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu).astype(np.float32)


# ---------------------------------------------------------------- reverb
def synthetic_rir(rng, sr=TARGET_SR):
    """Exponentially-decaying noise burst -- a crude but genuine room response."""
    rt60 = rng.uniform(0.15, 0.65)                   # small room to big hall
    n = int(sr * rt60)
    ir = rng.standard_normal(n).astype(np.float32)
    ir *= np.exp(-np.arange(n, dtype=np.float32) * (6.9 / max(n, 1)))
    ir[0] += 1.0                                     # keep the direct path
    return ir / (np.linalg.norm(ir) + 1e-9)


def reverb(x, ir):
    y = np.convolve(x, ir, mode="full")[: len(x)].astype(np.float32)
    p = float(np.max(np.abs(y))) or 1.0
    return (y / p * (float(np.max(np.abs(x))) or 1.0)).astype(np.float32)


# ----------------------------------------------------------------- noise
def pink(n, rng):
    """1/f noise -- closer to room tone than white noise is."""
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    k = np.arange(len(f))
    k[0] = 1
    f = f / np.sqrt(k)
    y = np.fft.irfft(f, n).astype(np.float32)
    return y / (np.std(y) + 1e-9)


def speech_rms(x, frame=400):
    """RMS of the loudest 10% of frames -- the speech level, not the average."""
    nf = max(1, x.size // frame)
    r = np.sqrt(np.mean(np.square(x[: nf * frame].reshape(nf, frame)), axis=1))
    k = max(1, nf // 10)
    return float(np.sort(r)[-k:].mean())


def add_noise(x, noise, snr_db):
    s = speech_rms(x)
    if s <= 0:
        return x
    if noise.size < x.size:
        noise = np.tile(noise, int(np.ceil(x.size / max(noise.size, 1))))
    noise = noise[: x.size]
    nr = float(np.sqrt(np.mean(np.square(noise)))) or 1.0
    target = s / (10.0 ** (snr_db / 20.0))
    y = x + noise * (target / nr)
    p = float(np.max(np.abs(y)))
    return (y / p * 0.95).astype(np.float32) if p > 0.95 else y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--p-reverb", type=float, default=0.8)
    ap.add_argument("--p-noise", type=float, default=0.8)
    ap.add_argument("--p-codec", type=float, default=0.5)
    ap.add_argument("--snr-min", type=float, default=8.0)
    ap.add_argument("--snr-max", type=float, default=30.0)
    ap.add_argument("--musan-root", default=None,
                    help="optional: unpacked MUSAN, for real noise instead of pink")
    ap.add_argument("--rir-root", default=None,
                    help="optional: unpacked RIRS_NOISES, for real impulse responses")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        sys.exit("pip install -U soundfile")

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"no audio in {src}")
    if args.limit:
        files = files[: args.limit]
    dst.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pyrng = random.Random(args.seed)

    musan = []
    if args.musan_root:
        musan = sorted(Path(args.musan_root).rglob("*.wav"))
        print(f"MUSAN: {len(musan)} noise files")
    rirs = []
    if args.rir_root:
        rirs = sorted(Path(args.rir_root).rglob("*.wav"))
        print(f"RIR: {len(rirs)} impulse responses")

    print(f"{len(files)} clips: {src} -> {dst}")
    n_rev = n_noi = n_cod = done = 0

    for i, f in enumerate(files):
        try:
            x, sr = sf.read(str(f), dtype="float32", always_2d=False)
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if x.size < sr * 0.3:
            continue

        if pyrng.random() < args.p_reverb:
            if rirs:
                try:
                    ir, _ = sf.read(str(pyrng.choice(rirs)), dtype="float32", always_2d=False)
                    ir = ir.mean(axis=1) if ir.ndim > 1 else ir
                    ir = ir / (np.linalg.norm(ir) + 1e-9)
                except Exception:
                    ir = synthetic_rir(rng, sr)
            else:
                ir = synthetic_rir(rng, sr)
            x = reverb(x, ir); n_rev += 1

        if pyrng.random() < args.p_noise:
            snr = pyrng.uniform(args.snr_min, args.snr_max)
            if musan:
                try:
                    nz, _ = sf.read(str(pyrng.choice(musan)), dtype="float32", always_2d=False)
                    nz = nz.mean(axis=1) if nz.ndim > 1 else nz
                except Exception:
                    nz = pink(x.size, rng)
            else:
                nz = pink(x.size, rng)
            x = add_noise(x, nz, snr); n_noi += 1

        if pyrng.random() < args.p_codec:
            x = mulaw_roundtrip(x); n_cod += 1

        sf.write(str(dst / f"aug_{f.stem}.wav"), x, sr)
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(files)}")

    print(f"\nwrote {done} clips to {dst}")
    print(f"  reverb applied : {n_rev}")
    print(f"  noise added    : {n_noi}  (SNR {args.snr_min:.0f}-{args.snr_max:.0f} dB)")
    print(f"  G.711 codec    : {n_cod}")
    if not musan and not rirs:
        print("\n  (synthetic reverb + pink noise -- pass --rir-root/--musan-root "
              "for the real ones if you have them unpacked)")
    print("\nNEXT: extract this folder TOO, and stamp it --label 1 like the clean one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
