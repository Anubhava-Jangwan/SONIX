#!/usr/bin/env python3
"""
make_rawboost.py  --  SONIX / SIH26104

Make a RawBoost-augmented copy of an ASVspoof split, so the detector stops being
fussy about WHICH microphone / channel / room a real voice was recorded in.

WHY THIS EXISTS
Our baseline was trained only on ASVspoof-2019 bonafide audio, which is clean
studio recording. On ordinary phone/laptop recordings it wrongly flags genuine
speech as fake, because it has never heard that channel. RawBoost simulates the
nuisance variation of real recording conditions -- different mics, different
channels, impulsive clicks, coloured background noise -- WITHOUT changing whether
the speech is real or fake. So the labels stay valid and the model learns to
ignore the channel instead of keying on it.

This is TRAINING augmentation (unlike make_codec.py, which we also use for
evaluation). Apply it to train (and optionally dev); NEVER to eval / In-the-Wild.

Reference: Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method
for Text-Independent Speaker Verification and Anti-Spoofing", ICASSP 2022.

ALGORITHMS (--algo)
    1 = linear + non-linear convolutive noise   (channel / mic colouration)
    2 = impulsive signal-dependent additive noise (clicks, crackle)
    3 = stationary signal-independent additive noise (coloured background hiss)
    4 = series 1 -> 2 -> 3
    5 = series 1 -> 2          (default; strongest reported setting on LA)
    6 = series 1 -> 3
    7 = series 2 -> 3
    8 = parallel 1 and 2, averaged

USAGE
    # 1) augmented copy of the training split (whole split)
    python make_rawboost.py --split train --out data/asvspoof19_la_rawboost --limit 0

    # 2) extract embeddings from it (normal script, unchanged)
    python src/extract_embeddings.py --split train --batch 8 \
                         --data-root data/asvspoof19_la_rawboost \
                         --out outputs/embeddings_rawboost

    # 3) train the robust head on clean + codec + rawboost
    python src/train.py --emb-root outputs/embeddings \
                        --extra-emb-root outputs/embeddings_rawboost \
                        --out outputs/models/head_robust.pt

Mirrors the ASVspoof folder layout and writes a matching protocol, so
extract_embeddings.py runs on it unchanged. Resumable: existing outputs are
skipped. One bad file is logged and skipped, never fatal.

Needs: numpy, scipy, soundfile.  No ffmpeg required.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROTOCOL_FILE = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
FLAC_DIR = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}

# ---- RawBoost default hyper-parameters (as published) ----------------------
N_F = 5                 # number of non-linear terms
NBANDS = 5              # notch filters per random FIR
MINF, MAXF = 20, 8000
MINBW, MAXBW = 100, 1000
MINCOEFF, MAXCOEFF = 10, 100
MING, MAXG = 0, 0
MIN_BIAS_LNL, MAX_BIAS_LNL = 5, 20
P_IMPULSE = 10          # max % of samples hit by impulsive noise
G_SD = 2                # impulsive noise gain
SNRMIN, SNRMAX = 10, 40


def randRange(x1, x2, integer):
    """Return a PLAIN Python scalar. NumPy 2.x refuses to convert a size-1
    array to a scalar, so never return an array here."""
    y = float(np.random.uniform(low=x1, high=x2))
    return int(y) if integer else y


def normWav(x, always):
    m = np.amax(abs(x))
    if m == 0:
        return x
    if always or m > 1:
        x = x / m
    return x


def genNotchCoeffs(nBands, minF, maxF, minBW, maxBW,
                   minCoeff, maxCoeff, minG, maxG, fs):
    from scipy import signal
    b = 1
    for _ in range(nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)
        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs), b)
    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x, b):
    from scipy import signal
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2):int(y.shape[0] - N / 2)]
    return y


def LnL_convolutive_noise(x, fs):
    """Linear + non-linear convolutive noise: simulates mic / channel colour."""
    y = np.zeros_like(x, dtype=np.float64)
    minG, maxG = MING, MAXG
    for i in range(N_F):
        if i == 1:
            minG = minG - MIN_BIAS_LNL
            maxG = maxG - MAX_BIAS_LNL
        b = genNotchCoeffs(NBANDS, MINF, MAXF, MINBW, MAXBW,
                           MINCOEFF, MAXCOEFF, minG, maxG, fs)
        yf = filterFIR(np.power(x, (i + 1)), b)
        if len(yf) != len(y):            # guard against off-by-one
            yf = yf[:len(y)] if len(yf) > len(y) else np.pad(
                yf, (0, len(y) - len(yf)))
        y = y + yf
    y = y - np.mean(y)
    return normWav(y, 0)


def ISD_additive_noise(x):
    """Impulsive, signal-dependent additive noise: clicks and crackle."""
    beta = randRange(0, P_IMPULSE, 0)
    y = x.copy().astype(np.float64)
    n = int(x.shape[0] * (beta / 100))
    if n <= 0:
        return normWav(y, 0)
    p = np.random.permutation(x.shape[0])[:n]
    f_r = np.multiply(((2 * np.random.rand(p.shape[0])) - 1),
                      ((2 * np.random.rand(p.shape[0])) - 1))
    y[p] = x[p] + G_SD * x[p] * f_r
    return normWav(y, 0)


def SSI_additive_noise(x, fs):
    """Stationary, signal-independent coloured additive noise: background hiss."""
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(NBANDS, MINF, MAXF, MINBW, MAXBW,
                       MINCOEFF, MAXCOEFF, MING, MAXG, fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = randRange(SNRMIN, SNRMAX, 0)
    nrm = np.linalg.norm(noise, 2)
    if nrm == 0:
        return x
    noise = noise / nrm * np.linalg.norm(x, 2) / (10 ** (0.05 * SNR))
    return x + noise


def rawboost(x, fs, algo=5):
    """Apply one RawBoost configuration. Input/output: float64 1-D waveform."""
    x = np.asarray(x, dtype=np.float64)
    if algo == 1:
        y = LnL_convolutive_noise(x, fs)
    elif algo == 2:
        y = ISD_additive_noise(x)
    elif algo == 3:
        y = SSI_additive_noise(x, fs)
    elif algo == 4:
        y = SSI_additive_noise(ISD_additive_noise(LnL_convolutive_noise(x, fs)), fs)
    elif algo == 5:
        y = ISD_additive_noise(LnL_convolutive_noise(x, fs))
    elif algo == 6:
        y = SSI_additive_noise(LnL_convolutive_noise(x, fs), fs)
    elif algo == 7:
        y = SSI_additive_noise(ISD_additive_noise(x), fs)
    elif algo == 8:
        y1 = LnL_convolutive_noise(x, fs)
        y2 = ISD_additive_noise(x)
        y = normWav(y1 + y2, 0)
    else:
        y = x
    return normWav(y, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="RawBoost-augment an ASVspoof split.")
    ap.add_argument("--split", default="train", choices=["train", "dev"],
                    help="NEVER run this on eval / In-the-Wild -- training aug only")
    ap.add_argument("--data-root", default="data/asvspoof19_la")
    ap.add_argument("--out", default="data/asvspoof19_la_rawboost",
                    help="output data-root for the augmented copy")
    ap.add_argument("--algo", type=int, default=5, choices=range(1, 9),
                    help="RawBoost algorithm (default 5 = LnL then ISD, series)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only convert the first N clips (0 = whole split)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    np.random.seed(args.seed)

    try:
        import soundfile as sf
    except Exception:
        sys.exit("FATAL: soundfile not installed.  pip install soundfile")
    try:
        import scipy  # noqa: F401
    except Exception:
        sys.exit("FATAL: scipy not installed.  pip install scipy")

    src_root = Path(args.data_root)
    proto = src_root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILE[args.split]
    if not proto.exists():
        sys.exit(f"FATAL: protocol not found: {proto}")
    src_flac = src_root / FLAC_DIR[args.split] / "flac"

    out_root = Path(args.out)
    out_flac = out_root / FLAC_DIR[args.split] / "flac"
    out_proto_dir = out_root / "ASVspoof2019_LA_cm_protocols"
    out_flac.mkdir(parents=True, exist_ok=True)
    out_proto_dir.mkdir(parents=True, exist_ok=True)

    lines = [ln for ln in proto.read_text().splitlines() if ln.strip()]
    lines.sort(key=lambda ln: ln.split()[1] if len(ln.split()) >= 2 else ln)
    if args.limit:
        lines = lines[: args.limit]

    # matching protocol so extract_embeddings.py sees exactly these files
    (out_proto_dir / PROTOCOL_FILE[args.split]).write_text("\n".join(lines) + "\n")

    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **k):
            return x

    done = skipped = failed = 0
    for ln in tqdm(lines, desc=f"rawboost{args.algo}:{args.split}", unit="file"):
        parts = ln.split()
        if len(parts) < 5:
            continue
        name = parts[1]
        src = src_flac / f"{name}.flac"
        dst = out_flac / f"{name}.flac"
        if dst.exists():
            skipped += 1
            continue
        if not src.exists():
            print(f"\n  ! missing source {src}")
            failed += 1
            continue
        try:
            wav, fs = sf.read(str(src), dtype="float64", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            y = rawboost(wav, fs, algo=args.algo)
            sf.write(str(dst), y.astype(np.float32), fs, format="FLAC")
            done += 1
        except Exception as exc:
            print(f"\n  ! failed on {name}: {exc}")
            failed += 1

    print(f"\n[rawboost{args.algo}/{args.split}] made={done}  already={skipped}  "
          f"failed={failed}  total={len(lines)}")
    print(f"augmented data-root: {out_root.resolve()}")
    print(f"next: python src/extract_embeddings.py --split {args.split} "
          f"--data-root {args.out} --out outputs/embeddings_rawboost --batch 8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
