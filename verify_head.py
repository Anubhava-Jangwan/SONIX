"""Prove that the current head.pt is the same model our headline number came from.

Navya's branch replaced outputs/models/head.pt and head_aug.pt with her rebuilt
copies. Git did it silently because those paths were still gitignored, and git
overwrites ignored files during a merge without a conflict. The rebuilds are
probably the same weights re-saved -- but "probably" is not good enough for the
1.4937 % that goes on a slide.

This re-scores the eval split with whatever head.pt is on disk now and compares,
score by score, against the scores the ORIGINAL head produced.

    python verify_head.py
    python verify_head.py --ckpt outputs/models/head_aug.pt --ref outputs/scores/... 
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def eer_from(scores, labels):
    """EER and its threshold. Same definition metrics.py uses."""
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    # sweep the threshold across every distinct score
    fn = np.cumsum(y == 1) / n_pos                 # spoof scored below thr
    tn = np.cumsum(y == 0) / n_neg
    fp = 1.0 - tn
    i = int(np.nanargmin(np.abs(fn - fp)))
    return float((fn[i] + fp[i]) / 2.0), float(s[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/models/head.pt")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--emb-root", default="outputs/embeddings")
    ap.add_argument("--ref-dir", default="outputs/scores",
                    help="scores produced by the ORIGINAL head")
    ap.add_argument("--out-dir", default="outputs/scores_verify")
    args = ap.parse_args()

    ref = Path(args.ref_dir) / f"{args.split}_scores.npy"
    if not ref.exists():
        print(f"no reference scores at {ref} -- nothing to compare against")
        return 1

    print(f"re-scoring {args.split} with {args.ckpt} ...")
    cmd = [sys.executable, "src/eval.py", "--split", args.split,
           "--emb-root", args.emb_root, "--model-ckpt", args.ckpt,
           "--out-scores", args.out_dir]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("eval.py failed -- see the output above")
        return 1

    new = np.load(Path(args.out_dir) / f"{args.split}_scores.npy")
    old = np.load(ref)
    lab = np.load(Path(args.out_dir) / f"{args.split}_labels.npy")

    print()
    print("=" * 68)
    if new.shape != old.shape:
        print(f"SHAPE MISMATCH: new {new.shape} vs reference {old.shape}")
        print("These were scored on different trial sets -- not comparable.")
        return 1

    diff = np.abs(new - old)
    eer_new, thr_new = eer_from(new, lab)
    eer_old, thr_old = eer_from(old, lab)

    print(f"trials compared     : {new.size:,}")
    print(f"max score difference: {diff.max():.3e}")
    print(f"mean difference     : {diff.mean():.3e}")
    print()
    print(f"EER, original head  : {eer_old * 100:.4f} %   @ {thr_old:.6f}")
    print(f"EER, current head   : {eer_new * 100:.4f} %   @ {thr_new:.6f}")
    print("=" * 68)

    if diff.max() < 1e-5:
        print("IDENTICAL -- the rebuilt checkpoint is the same model. Your "
              "headline number stands, no slide changes needed.")
        return 0
    if abs(eer_new - eer_old) < 5e-5:
        print("Scores differ slightly but the EER is unchanged to four decimals. "
              "Safe to keep quoting the number; note the rebuild in your log.")
        return 0
    print("DIFFERENT MODEL. The number on your results slide was measured with a "
          "checkpoint you no longer have. Quote the CURRENT EER above instead, "
          "and tell Yukti to update the results table.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
