"""Write an UNTRAINED head so the live path can be exercised before head.pt lands.

WHY THIS EXISTS. head.pt is not in git and cannot be - .gitignore excludes *.pt
and outputs/, and no .pt object exists in any branch. It has to be copied from
Yugal's machine. Until it arrives there is no way to find out whether the live
path actually works end to end, or whether a wav2vec2 forward pass every 0.5 s
keeps up on this GPU. Discovering that at midnight before the demo is the risk
this file removes.

WHAT IT IS NOT. The weights are random. Its scores are NOISE and mean nothing
about any voice. The checkpoint is stamped synthetic=True, the server reads that
and reports scoring_synthetic, and the capture page shows a loud banner. Never
show a number from this to anyone, and never quote one.

WHAT IT IS GOOD FOR: proving capture -> window -> front-end -> head -> band ->
spectrogram is connected, and measuring real per-window latency on this machine.

    python -m realtime.make_dev_head
    python -m realtime.server --ckpt outputs/models/head_dev.pt --ws-port 8000 --mode webrtc

Delete it the moment the real checkpoint arrives.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Must match src/train.py build_head() and its argparse defaults, or the latency
# measured here will not reflect the real head.
IN_DIM, HIDDEN, DROPOUT = 1024, 256, 0.3
DEFAULT_OUT = "outputs/models/head_dev.pt"


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    args = ap.parse_args()

    out = Path(args.out)
    if out.name in ("head.pt", "head_robust.pt", "head_aug.pt"):
        raise SystemExit(
            f"refusing to write to {out.name} - that name is reserved for a real "
            "trained checkpoint. Use head_dev.pt."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    head = nn.Sequential(
        nn.Linear(IN_DIM, args.hidden),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(args.hidden, 1),
    )

    torch.save({
        "state_dict": head.state_dict(),
        # Identity standardiser: no training data was seen, so there are no
        # real per-dimension statistics to store.
        "mu": np.zeros(IN_DIM, np.float32),
        "sd": np.ones(IN_DIM, np.float32),
        "config": {"in_dim": IN_DIM, "hidden": args.hidden,
                   "dropout": DROPOUT, "standardized": False},
        "dev_eer": None,             # never trained, so there is no EER
        "front_end": "facebook/wav2vec2-xls-r-300m",
        "synthetic": True,           # the flag every consumer keys off
        "note": "UNTRAINED random weights. Scores are meaningless. Not for demo.",
    }, out)

    print(f"wrote {out.resolve()}")
    print("UNTRAINED - scores from this checkpoint are noise. Plumbing/latency only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
