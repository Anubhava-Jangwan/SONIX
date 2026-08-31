#!/usr/bin/env python3
"""
add_noise.py -- SONIX / SIH26104

TEST THE SHORTCUT HYPOTHESIS.

Measured on our own 20 clips: every genuine recording with a NOISY background
(SNR < 38 dB) is scored real; every genuine recording with a CLEAN background
(SNR > 39 dB) is scored fake. Synthetic audio has an essentially silent
background (SNR 95-140 dB). So the model appears to have learned
"clean background = fake" instead of learning synthesis artefacts.

If that is what is happening, then adding a little background noise to a
FAILING genuine clip should flip it from "fake" back to "real" -- without
changing the voice at all.

    python add_noise.py --in-dir sonix_real/sonix_real/real --out-dir data/noised --snr 30

Then score the output folder and compare. If the scores collapse, the shortcut
is confirmed and the fix is to break that correlation in training.
"""
import argparse, os, wave
import numpy as np


def read_wav(p):
    w = wave.open(p)
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    a = np.frombuffer(w.readframes(n), dtype={1: np.int8, 2: np.int16, 4: np.int32}[sw]).astype(np.float64)
    w.close()
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a / (2 ** (8 * sw - 1)), sr


def write_wav(p, a, sr):
    a = np.clip(a, -1, 1)
    w = wave.open(p, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((a * 32767).astype(np.int16).tobytes())
    w.close()


def speech_rms(a, sr):
    """RMS of the loudest 10% of frames -- i.e. the speech, not the silence."""
    N = 1024
    f = np.array([np.mean(a[i:i + N] ** 2) for i in range(0, len(a) - N, N)])
    if not len(f):
        return float(np.sqrt(np.mean(a ** 2) + 1e-12))
    top = np.sort(f)[-max(1, len(f) // 10):]
    return float(np.sqrt(top.mean() + 1e-12))


def pink(n, rng):
    """Pink-ish noise: closer to real room tone than white noise."""
    w = rng.standard_normal(n)
    S = np.fft.rfft(w)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    S /= np.sqrt(f)
    out = np.fft.irfft(S, n)
    return out / (np.std(out) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--snr", type=float, default=30.0,
                    help="target speech-to-noise ratio in dB (30 = clearly audible room tone)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    files = sorted(f for f in os.listdir(args.in_dir) if f.lower().endswith(".wav"))
    if not files:
        raise SystemExit(f"no .wav in {args.in_dir}")

    print(f"adding pink noise at {args.snr:.0f} dB SNR -> {args.out_dir}\n")
    for fn in files:
        a, sr = read_wav(os.path.join(args.in_dir, fn))
        s = speech_rms(a, sr)
        target_noise = s / (10 ** (args.snr / 20.0))
        nz = pink(len(a), rng) * target_noise
        out = a + nz
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out /= peak
        write_wav(os.path.join(args.out_dir, fn), out, sr)
        print(f"  {fn:<16} speech_rms {20*np.log10(s+1e-12):7.1f} dB  "
              f"noise added at {20*np.log10(target_noise+1e-12):7.1f} dB")
    print(f"\n{len(files)} files written. Now score {args.out_dir} and compare.")


if __name__ == "__main__":
    main()
