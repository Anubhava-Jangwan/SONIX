"""Score every labelled clip with every head, alongside the acoustic features
that predict failure -- so a wrong verdict points at a CAUSE, not just a number.

We already established the baseline's failure mode: it learned "clean
background = fake". Real recordings split perfectly at ~38 dB SNR -- below it
they passed, above it they were called fake, while the synthetic clips sat at
96-140 dB. That is a shortcut, not detection.

This runs the same test across all four heads and reports, per model:
  - false-alarm rate on genuine clips, detection rate on cloned clips
  - EER on this clip set
  - the correlation between a REAL clip's SNR and its score

That last column is the one that matters. If it is still strongly positive for
a head, that head is still reading background noise instead of the voice, and
no threshold change will fix it.

    python bench_clips.py
    python bench_clips.py --real data/noised_real --fake sonix_real/sonix_real/fake
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "demo"))

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".m4a")


# ---------------------------------------------------------------- features
def load_mono16k(path):
    import soundfile as sf
    w, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != 16000:
        n = int(round(w.size * 16000 / sr))
        w = np.interp(np.linspace(0, w.size - 1, n), np.arange(w.size), w).astype(np.float32)
    return w


def dbfs(x):
    r = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(max(r, 1e-12))


def features(path):
    """Duration, level, and the SNR/noise-floor pair that predicted failure."""
    w = load_mono16k(path)
    n = w.size
    if n == 0:
        return None

    # 25 ms frames -> loudest 10 % is speech, quietest 10 % is the noise floor
    f = 400
    nf = max(1, n // f)
    frames = w[:nf * f].reshape(nf, f)
    rms = np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-12
    order = np.sort(rms)
    k = max(1, nf // 10)
    speech = float(order[-k:].mean())
    floor = float(order[:k].mean())

    return {
        "dur": n / 16000.0,
        "dbfs": dbfs(w),
        "floor_db": 20.0 * np.log10(max(floor, 1e-12)),
        "snr_db": 20.0 * np.log10(speech / max(floor, 1e-12)),
    }


# ------------------------------------------------------------------ scoring
def score_clip(path, ckpt):
    import score_file as S
    S.set_vad(enabled=True)
    v = np.asarray(S.score_file(str(path), ckpt_path=ckpt), dtype=float)
    return v[np.isfinite(v)]


def eer(scores, labels):
    order = np.argsort(scores)
    y = np.asarray(labels)[order]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    fn = np.cumsum(y == 1) / n_pos
    fp = 1.0 - np.cumsum(y == 0) / n_neg
    i = int(np.nanargmin(np.abs(fn - fp)))
    return float((fn[i] + fp[i]) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="sonix_real/sonix_real/real")
    ap.add_argument("--fake", default="sonix_real/sonix_real/fake")
    ap.add_argument("--amber", type=float, default=0.10)
    ap.add_argument("--models", default=None, help="comma-separated registry keys")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from realtime import models as reg

    heads = [(m["key"], m["label"], m["path"]) for m in reg.catalogue() if m["exists"]]
    if args.models:
        want = set(args.models.split(","))
        heads = [h for h in heads if h[0] in want]
    if not heads:
        print("no checkpoints found under outputs/models/")
        return 1

    clips = []
    for folder, truth in ((args.real, 0), (args.fake, 1)):
        p = Path(folder)
        if not p.is_dir():
            print(f"missing folder: {folder}")
            continue
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in AUDIO_EXT:
                clips.append((f, truth))
    if not clips:
        print("no audio found")
        return 1

    print(f"{len(clips)} clips x {len(heads)} heads\n")

    feats, med = {}, {}
    for f, _ in clips:
        feats[f] = features(f)
    for key, label, ckpt in heads:
        print(f"  scoring with {label} ...")
        for f, _ in clips:
            v = score_clip(f, ckpt)
            med[(f, key)] = float(np.median(v)) if v.size else float("nan")

    # ---------------------------------------------------------- per clip
    print()
    hdr = f"{'clip':<16}{'truth':<7}{'SNR dB':>8}{'floor':>8}{'dur':>7}  "
    hdr += "".join(f"{k:>11}" for k, _, _ in heads)
    print(hdr)
    print("-" * len(hdr))
    for f, truth in clips:
        ft = feats[f] or {}
        row = (f"{f.name[:15]:<16}{'real' if truth == 0 else 'fake':<7}"
               f"{ft.get('snr_db', float('nan')):>8.1f}"
               f"{ft.get('floor_db', float('nan')):>8.1f}"
               f"{ft.get('dur', float('nan')):>7.1f}  ")
        for key, _, _ in heads:
            s = med[(f, key)]
            flag = "*" if (truth == 0 and s >= args.amber) or (truth == 1 and s < args.amber) else " "
            row += f"{s:>10.3f}{flag}"
        print(row)
    print("\n* = wrong side of the amber threshold "
          f"({args.amber:.2f}); scores are the clip's MEDIAN window.")

    # ---------------------------------------------------------- per model
    print()
    print(f"{'model':<28}{'false alarm':>13}{'detection':>11}{'EER':>9}"
          f"{'SNR corr':>11}")
    print("-" * 72)
    for key, label, _ in heads:
        s = np.array([med[(f, key)] for f, _ in clips])
        y = np.array([t for _, t in clips])
        ok = np.isfinite(s)
        fa = float(np.mean(s[ok & (y == 0)] >= args.amber))
        det = float(np.mean(s[ok & (y == 1)] >= args.amber))

        real_snr = np.array([feats[f]["snr_db"] for f, t in clips
                             if t == 0 and feats[f]])
        real_sc = np.array([med[(f, key)] for f, t in clips if t == 0 and feats[f]])
        m = np.isfinite(real_snr) & np.isfinite(real_sc)
        corr = (float(np.corrcoef(real_snr[m], real_sc[m])[0, 1])
                if m.sum() > 2 and real_sc[m].std() > 1e-9 else float("nan"))

        print(f"{label:<28}{fa:>12.0%}{det:>11.0%}"
              f"{eer(s[ok], y[ok]) * 100:>8.1f}%{corr:>11.2f}")

    print()
    print("Reading the last column -- correlation between a GENUINE clip's SNR")
    print("and how fake the model thinks it is:")
    print("  > +0.6   the shortcut is intact: cleaner recording -> 'more fake'")
    print("  ~  0     score is independent of background noise (what we want)")
    print("  < -0.6   inverted shortcut; also wrong, just in the other direction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
