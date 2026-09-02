#!/usr/bin/env python3
"""
check_rawboost.py -- verify RawBoost output is degraded but NOT destroyed.

    python check_rawboost.py

Compares each augmented file against its original and reports duration match,
loudness, correlation with the original, and clipping/NaN problems.

WHAT GOOD LOOKS LIKE
  duration identical, no NaN, not silent, and correlation roughly 0.1-0.9.
  Very high correlation (>0.98) = augmentation barely did anything.
  Near-zero correlation + very low loudness = the signal was destroyed.
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

SRC = Path("data/asvspoof19_la/ASVspoof2019_LA_train/flac")
AUG = Path("data/asvspoof19_la_rawboost/ASVspoof2019_LA_train/flac")


def dbfs(x):
    r = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    return 20.0 * np.log10(r + 1e-12)


def main():
    files = sorted(AUG.glob("*.flac"))
    if not files:
        sys.exit(f"no augmented files in {AUG.resolve()}")
    print(f"{'file':<20} {'dur':>6} {'orig dB':>8} {'aug dB':>8} {'corr':>7} {'%@FS':>7}  notes")
    bad = 0
    for f in files:
        o = SRC / f.name
        if not o.exists():
            print(f"{f.stem:<20} original missing")
            continue
        a, sa = sf.read(str(f), dtype="float64")
        b, sb = sf.read(str(o), dtype="float64")
        notes = []
        if len(a) != len(b):
            notes.append(f"LENGTH MISMATCH {len(a)}vs{len(b)}")
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if not np.all(np.isfinite(a)):
            notes.append("NaN/Inf")
        if dbfs(a) < -60:
            notes.append("NEARLY SILENT")
        pct_fs = 100.0 * float(np.mean(np.abs(a) >= 0.999))
        if pct_fs > 0.05:
            notes.append(f"REAL CLIPPING {pct_fs:.2f}% of samples at full scale")
        sd = a.std() * b.std()
        corr = float(np.mean((a - a.mean()) * (b - b.mean())) / sd) if sd > 0 else 0.0
        if corr > 0.98:
            notes.append("barely changed")
        if abs(corr) < 0.02:
            notes.append("UNRELATED to original")
        if notes:
            bad += 1
        print(f"{f.stem:<20} {n/sa:6.2f} {dbfs(b):8.1f} {dbfs(a):8.1f} {corr:7.3f} "
              f"{pct_fs:7.3f}  {'; '.join(notes) if notes else 'ok'}")
    print(f"\n{len(files)} files, {bad} flagged.")
    print("If most rows say 'ok' with correlation ~0.1-0.9, RawBoost is working.")


if __name__ == "__main__":
    main()
