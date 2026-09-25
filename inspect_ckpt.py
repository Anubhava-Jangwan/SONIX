"""Show what is actually inside a SONIX head checkpoint.

Use it on anything a teammate sends before trusting it: it proves the file is a
real checkpoint (not an unzipped directory, not a truncated download), prints
the dev EER the training run recorded, and shows the standardiser stats so two
heads can be compared like for like.

    python inspect_ckpt.py                          # every head in outputs/models
    python inspect_ckpt.py path\\to\\head_robust_v2.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

INTERESTING = ("dev_eer", "eer", "threshold", "epoch", "best_epoch", "temperature",
               "train_sources", "emb_roots", "n_train", "n_dev", "pos_weight",
               "args", "created", "timestamp")


def describe(path: Path):
    print("=" * 72)
    print(path)
    print("=" * 72)

    if path.is_dir():
        print("  NOT A CHECKPOINT -- this is a DIRECTORY.")
        print("  A .zip was extracted into a folder instead of unzipped to a file.")
        print(f"  contents: {[p.name for p in path.iterdir()][:8]}")
        return False
    if not path.exists():
        print("  missing")
        return False

    size = path.stat().st_size
    print(f"  size      : {size:,} bytes")
    if size < 100_000:
        print("  WARNING: far smaller than the ~1.06 MB a 262,657-param head "
              "should be. Likely truncated.")

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  UNREADABLE: {exc}")
        return False

    if not isinstance(ck, dict):
        print(f"  unexpected type: {type(ck)}")
        return False

    missing = [k for k in ("config", "state_dict", "mu", "sd") if k not in ck]
    if missing:
        print(f"  NOT A SONIX HEAD -- missing {missing}")
        print(f"  keys present: {sorted(ck.keys())}")
        return False

    cfg = ck["config"]
    n_params = sum(int(np.prod(v.shape)) for v in ck["state_dict"].values())
    print(f"  config    : {cfg}")
    print(f"  parameters: {n_params:,}")

    mu, sd = np.asarray(ck["mu"], np.float64), np.asarray(ck["sd"], np.float64)
    print(f"  mu        : dim {mu.shape[0]}  mean {mu.mean():+.4f}  "
          f"min {mu.min():+.4f}  max {mu.max():+.4f}")
    print(f"  sd        : dim {sd.shape[0]}  mean {sd.mean():.4f}   "
          f"zeros {int((sd == 0).sum())}")

    shown = False
    for k in INTERESTING:
        if k in ck:
            v = ck[k]
            if isinstance(v, float):
                extra = f"   ({v * 100:.4f} %)" if "eer" in k.lower() and v <= 1 else ""
                print(f"  {k:<10}: {v:.6f}{extra}")
            else:
                print(f"  {k:<10}: {v}")
            shown = True
    extras = sorted(set(ck) - {"config", "state_dict", "mu", "sd"} - set(INTERESTING))
    if extras:
        print(f"  other keys: {extras}")
    if not shown and not extras:
        print("  no metrics stored in this checkpoint -- the training run did not "
              "save its dev EER, so it has to come from the training log.")
    return True


def main():
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args]
    else:
        root = Path("outputs/models")
        paths = sorted(root.glob("*.pt")) + sorted(p for p in root.iterdir()
                                                   if p.is_dir()) if root.exists() else []
        if not paths:
            print("no checkpoints under outputs/models/")
            return 1
    ok = all(describe(p) for p in paths)
    print()
    print("all checkpoints readable" if ok else "one or more checkpoints are unusable")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
