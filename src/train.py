#!/usr/bin/env python3
"""
train.py  --  SONIX / SIH26104

Train the small MLP head on the cached wav2vec2 embeddings. The front-end is
already frozen and its outputs are on disk, so this trains in MINUTES. If it is
taking hours you are re-extracting features somewhere -- stop and find it.

Model (exactly as specified in the brief):
    Linear(1024, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)
    Binary cross-entropy, Adam, lr 1e-3, ~20 epochs.
    Dev is used ONLY for early stopping. Eval is never touched here.

Input:   embeddings/train/shard_*.npy   embeddings/dev/shard_*.npy   (+ .labels.npy)
Output:  models/head.pt   (state_dict + input standardiser + config, all in one)

USAGE
    python train.py --emb-root embeddings --out models/head.pt
    python train.py --emb-root embeddings --out models/head.pt --epochs 30 --seed 0

Note on standardisation: we z-score the 1024-dim vectors using statistics from the
TRAIN split only, and store those stats in the checkpoint so eval.py and
score_file.py apply the identical transform. It is a no-op on the architecture and
consistently helps the head converge. Turn it off with --no-standardize.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# EER -- local cross-check copy. Yukti's metrics.py is the source of truth for
# the reported number; this exists so training can early-stop on dev EER and so
# our number and hers can be compared. labels: 1 = spoof (positive class).
# ---------------------------------------------------------------------------
def eer(labels, scores) -> float:
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)        # positive class = spoof = 1
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)      # == brief's fpr[idx] to <0.01%


def load_split(emb_root: str, split: str):
    """Concatenate all shards for a split -> (X float32 (N,1024), y int64 (N,))."""
    d = Path(emb_root) / split
    shards = sorted(d.glob("shard_*.npy"))
    shards = [s for s in shards if ".labels" not in s.name]
    if not shards:
        sys.exit(f"FATAL: no shards in {d}. Run extract_embeddings.py --split "
                 f"{split} first.")
    Xs, ys = [], []
    for s in shards:
        lab = s.with_suffix(".labels.npy")
        if not lab.exists():
            sys.exit(f"FATAL: {s} has no matching {lab.name}. Re-extract this "
                     f"shard -- embeddings and labels must stay paired.")
        emb = np.load(s)
        y = np.load(lab)
        if len(emb) != len(y):
            sys.exit(f"FATAL: {s.name} rows {len(emb)} != labels {len(y)}. "
                     f"Corrupt shard; delete it and re-extract.")
        Xs.append(emb.astype(np.float32))
        ys.append(y.astype(np.int64))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    print(f"[{split}] loaded {len(X)} vectors from {len(shards)} shards  "
          f"(spoof={int((y == 1).sum())}, bonafide={int((y == 0).sum())})")
    return X, y


def build_head(in_dim, hidden, dropout):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def main(argv=None) -> int:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser(description="Train the SONIX MLP head.")
    ap.add_argument("--emb-root", default="outputs/embeddings")
    ap.add_argument("--extra-emb-root", default=None, action="append",
                    metavar="ROOT",
                    help="extra embeddings root whose train (and dev, if present) "
                         "is concatenated for augmentation. REPEATABLE - pass it "
                         "once per root, e.g. --extra-emb-root outputs/embeddings_g711 "
                         "--extra-emb-root outputs/embeddings_rawboost "
                         "--extra-emb-root outputs/embeddings_real")
    ap.add_argument("--out", default="outputs/models/head.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=6,
                    help="early-stop if dev EER doesn't improve for N epochs")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    Xtr, ytr = load_split(args.emb_root, "train")
    Xdv, ydv = load_split(args.emb_root, "dev")

    # ---- codec augmentation: also train on codec'd copies -----------------
    # Point --extra-emb-root at a second embeddings root (e.g. the G.711 one).
    # Its train (and dev, if present) are concatenated so the head sees both
    # clean and phone-compressed audio. This is how you make the AUGMENTED model.
    if args.extra_emb_root:
        for _root in args.extra_emb_root:
            Xe, ye = load_split(_root, "train")
            Xtr = np.concatenate([Xtr, Xe], 0)
            ytr = np.concatenate([ytr, ye], 0)
            try:
                Xde, yde = load_split(_root, "dev")
                Xdv = np.concatenate([Xdv, Xde], 0)
                ydv = np.concatenate([ydv, yde], 0)
            except SystemExit:
                print(f"[aug] {_root}: no dev split found; leaving dev as-is "
                      f"for early stopping")
        print(f"[aug] augmented train set: {len(Xtr)} vectors from "
              f"{1 + len(args.extra_emb_root)} roots  "
              f"(spoof={int((ytr == 1).sum())}, bonafide={int((ytr == 0).sum())})")

    # ---- input standardiser (train stats only) ---------------------------
    if args.no_standardize:
        mu = np.zeros(Xtr.shape[1], np.float32)
        sd = np.ones(Xtr.shape[1], np.float32)
    else:
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xdv = (Xdv - mu) / sd

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()
    Xdv_t = torch.from_numpy(Xdv).float().to(device)

    model = build_head(Xtr.shape[1], args.hidden, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"head parameters: {n_params:,}")

    # class imbalance (~9x spoof): weight the positive class so bonafide isn't
    # drowned out. pos_weight = n_neg / n_pos.
    n_pos = max(int((ytr == 1).sum()), 1)
    n_neg = max(int((ytr == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = len(Xtr_t)
    best_eer, best_state, best_epoch, since = float("inf"), None, -1, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device).unsqueeze(1)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        train_loss = running / n

        model.eval()
        with torch.no_grad():
            dev_logits = model(Xdv_t).squeeze(1).cpu().numpy()
        dev_eer = eer(ydv, dev_logits)
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"dev_EER={dev_eer * 100:.2f}%")

        if dev_eer < best_eer - 1e-5:
            best_eer, best_epoch, since = dev_eer, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop: no dev improvement for {args.patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBEST dev EER = {best_eer * 100:.2f}%  (epoch {best_epoch})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mu": mu, "sd": sd,
        "config": {"in_dim": int(Xtr.shape[1]), "hidden": args.hidden,
                   "dropout": args.dropout, "standardized": not args.no_standardize},
        "dev_eer": best_eer,
        "front_end": "facebook/wav2vec2-xls-r-300m",
    }, out)
    print(f"saved checkpoint -> {out.resolve()}")
    print("Next: python eval.py --split eval --model-ckpt", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
