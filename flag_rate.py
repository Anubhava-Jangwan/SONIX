#!/usr/bin/env python3
"""
flag_rate.py  --  SONIX / SIH26104

False-alarm rate on a BONAFIDE-ONLY set.

WHY THIS EXISTS
metrics.py computes EER, which needs both classes. A genuine-speech-only set
(IndicVoices, our own real recordings) has one class, so EER is undefined --
but the question we actually care about is still answerable and is arguably
more important:

    Of clips we KNOW are real human speech, what fraction does the model call fake?

That is the Modi / Bachchan failure mode, measured directly.

USAGE
    python flag_rate.py --scores-dir outputs\\scores_indicvoices_baseline --split train

    # compare several models at once
    python flag_rate.py --split train ^
        --scores-dir outputs\\scores_indicvoices_baseline ^
        --scores-dir outputs\\scores_indicvoices_robust ^
        --scores-dir outputs\\scores_indicvoices_full

Reports the flag rate at the demo's amber/red bands and at the EER thresholds
measured on DF21, so the numbers line up with what the product would actually do.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Thresholds worth reporting. The DF21 EER thresholds are each model's own
# operating point on the cross-dataset benchmark.
BANDS = [
    ("amber 0.10", 0.10),
    ("0.50", 0.50),
    ("red 0.90", 0.90),
    ("0.99", 0.99),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", action="append", required=True,
                    help="a directory holding <split>_scores.npy (repeatable)")
    ap.add_argument("--split", default="train",
                    help="filename prefix eval.py used (default: train)")
    ap.add_argument("--expect", choices=["bonafide", "spoof"], default="bonafide",
                    help="what this set actually is (default: bonafide)")
    args = ap.parse_args()

    print()
    print(f"  Set is entirely {args.expect.upper()}.", end=" ")
    if args.expect == "bonafide":
        print("Every clip scored above a threshold is a FALSE ALARM.")
    else:
        print("Every clip scored below a threshold is a MISS.")
    print()

    hdr = f"{'model / scores dir':38s} {'n':>7s} " + " ".join(f"{b:>11s}" for b, _ in BANDS)
    print(hdr)
    print("-" * len(hdr))

    for d in args.scores_dir:
        p = Path(d) / f"{args.split}_scores.npy"
        if not p.exists():
            print(f"{d:38s}  MISSING {p}")
            continue
        s = np.load(p).astype(np.float64)

        lp = Path(d) / f"{args.split}_labels.npy"
        if lp.exists():
            lab = np.load(lp)
            uniq = set(np.unique(lab).tolist())
            want = {0} if args.expect == "bonafide" else {1}
            if uniq != want:
                print(f"  ! {d}: labels are {uniq}, expected {want}. "
                      f"This set is not purely {args.expect} -- read the numbers "
                      f"below with that in mind.")

        cells = []
        for _, t in BANDS:
            rate = (s > t).mean() * 100 if args.expect == "bonafide" else (s < t).mean() * 100
            cells.append(f"{rate:10.2f}%")
        print(f"{Path(d).name:38s} {len(s):7d} " + " ".join(cells))

    print()
    print("  Lower is better in every column.")
    print("  These are RAW rates, not EER. They depend on each model's calibration,")
    print("  so read them alongside the DF21 EER rather than instead of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
