#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tier_a.py -- behavioural confirmation that head.pt decides on BACKGROUND LEVEL, not voice.

Single process: the frozen wav2vec2-XLS-R front-end loads ONCE (on GPU if available),
then every variant is scored. Uses the demo's own pipeline (demo/score_file.py) with the
VAD gate switched OFF, so the numbers are raw model behaviour.

    python docs/handoff/scripts/tier_a.py --repo-root .            # auto GPU/CPU
    python docs/handoff/scripts/tier_a.py --repo-root . --device cuda

Outputs:
    stdout                     human-readable tables (also redirect to a .log)
    docs/handoff/scripts/tier_a_result.json   machine-readable, send this back

Sub-tests
    A1  SNR dose-response   one clip swept clean -> 40..10 dB SNR; is score monotone in SNR?
    A2  pure-noise input    pink noise / digital silence, ZERO speech; is there still a boundary?
    A3  score vs SNR bank   correlate head.pt score with measured SNR over all 22 clips,
                            with score-vs-duration and score-vs-loudness as controls
    A4  background-only     strip speech frames, score the leftover room tone

Expected if the shortcut hypothesis is correct
    A1  score falls monotonically as SNR falls        Spearman(score, SNR)  ~ +0.8..+1.0
    A2  quiet/silent -> ~1 (fake), loud noise -> ~0   Spearman(score, level) ~ -0.8..-1.0
    A3  score tracks SNR, NOT duration or loudness    |r(score,SNR)| high; |r(score,dur)| low
    A4  genuine-bg and fake-bg both score similar / high, i.e. voice is not needed for a verdict
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

T0 = time.time()


def el():
    return f"[{time.time() - T0:6.0f}s]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--ckpt", default="outputs/models/head.pt")
    args = ap.parse_args()

    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    root = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(root, "demo"))
    sys.path.insert(0, root)

    import torch
    import score_file as S
    import add_noise as AN

    if args.device:
        S.configure(device=args.device)
    S.set_vad(enabled=False)
    CKPT = args.ckpt
    VAD_DB = -45.0

    # -- helpers --------------------------------------------------------------
    def rms_dbfs(w):
        r = float(np.sqrt(np.mean(np.square(w)) + 1e-12))
        return 20.0 * np.log10(r + 1e-12)

    def spearman(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        if rx.std() == 0 or ry.std() == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])

    def pearson(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def load(p):
        return S._load_audio(p)

    def score_wav(wav, stride=1):
        wins = list(S._windows(wav))
        if stride > 1:
            wins = wins[::stride]
        if not wins:
            return float("nan"), 0, 0
        sc = []
        for i in range(0, len(wins), 8):
            sc.extend(S._score_windows(wins[i:i + 8], CKPT))
        sc = np.asarray(sc)
        dbs = np.array([rms_dbfs(w) for w in wins])
        m = dbs > VAD_DB
        smean = float(sc[m].mean()) if m.any() else float(sc.mean())
        return smean, int(m.sum()), len(wins)

    def add_pink(a, snr_db, seed):
        rng = np.random.default_rng(seed)
        s = AN.speech_rms(a, 16000)
        tgt = s / (10 ** (snr_db / 20.0))
        out = a + AN.pink(len(a), rng) * tgt
        p = np.max(np.abs(out))
        return (out / p if p > 1.0 else out).astype(np.float32)

    def noise_floor_db(a):
        N = 1024
        f = np.array([np.sqrt(np.mean(a[i:i + N] ** 2) + 1e-12)
                      for i in range(0, len(a) - N, N)])
        if not len(f):
            return rms_dbfs(a)
        q = np.sort(f)[:max(1, len(f) // 10)]
        return 20.0 * np.log10(q.mean() + 1e-12)

    def snr_db(a):
        return 20.0 * np.log10(AN.speech_rms(a, 16000) + 1e-12) - noise_floor_db(a)

    # -- report device ------------------------------------------------------
    S._load_frontend()
    dev = S._FE.get("device")
    print(f"{el()}  front-end device = {dev}   torch {torch.__version__}   "
          f"cuda_available={torch.cuda.is_available()}")
    print(f"{el()}  checkpoint = {CKPT}   dev_eer(stored) = {S.dev_eer(CKPT)}")

    R = {"meta": {"device": str(dev), "torch": torch.__version__,
                  "cuda": torch.cuda.is_available(), "ckpt": CKPT}}

    P1 = os.path.join(root, "data", "probe", "clean")
    P2 = os.path.join(root, "data", "probe2", "clean")

    # ================================================================= A1
    print("\n" + "=" * 72 + "\nA1  SNR DOSE-RESPONSE   (higher score = more 'fake')\n" + "=" * 72)
    a1 = {
        "real_03 (clean laptop-mic genuine)": os.path.join(P1, "real_03.wav"),
        "fake_03 (downloaded clone)":         os.path.join(P1, "fake_03.wav"),
        "output.wav (AI Modi clone)":         os.path.join(P2, "output_fake.wav"),
    }
    grid = [None, 40, 35, 30, 25, 20, 15, 10]
    R["A1"] = {}
    for name, p in a1.items():
        if not os.path.isfile(p):
            print(f"  SKIP {name}: {p} missing (run build_probe*.py first)")
            continue
        base = load(p)
        row = {}
        print(f"\n{name}   {el()}")
        for g in grid:
            wav = base if g is None else add_pink(base, g, abs(hash((name, g))) % (2**32))
            sm, nsp, ntot = score_wav(wav)
            tag = "clean" if g is None else f"{g}dB"
            row[tag] = round(sm, 4)
            print(f"   {tag:>6}  score={sm:6.3f}   ({nsp}/{ntot} speech-win)")
        xs = [60.0 if k == "clean" else float(k[:-2]) for k in row]
        sp = spearman(xs, [row[k] for k in row])
        print(f"   --> Spearman(score, SNR) = {sp:+.3f}")
        R["A1"][name] = {"row": row, "spearman_score_vs_snr": round(sp, 3)}

    # ================================================================= A2
    print("\n" + "=" * 72 + "\nA2  PURE-NOISE INPUT  --  NO SPEECH AT ALL\n" + "=" * 72)
    R["A2"] = {}
    rng = np.random.default_rng(20260902)
    dur = 16000 * 8
    print(f"\n   {'input':>22}  {'RMS dBFS':>9}  {'score':>7}")
    sil = np.full(dur, 1e-5, np.float32)
    sm, _, _ = score_wav(sil)
    print(f"   {'digital silence':>22}  {rms_dbfs(sil):9.1f}  {sm:7.3f}")
    R["A2"]["silence"] = round(sm, 4)
    for db in [-60, -55, -50, -45, -40, -35, -30, -25, -20, -15]:
        n = AN.pink(dur, rng).astype(np.float32)
        n = n / (np.sqrt(np.mean(n ** 2)) + 1e-12) * (10 ** (db / 20.0))
        sm, _, _ = score_wav(n.astype(np.float32))
        print(f"   {'pink noise':>22}  {db:9.1f}  {sm:7.3f}")
        R["A2"][f"pink_{db}dBFS"] = round(sm, 4)
    lv = sorted((k for k in R["A2"] if k.startswith("pink_")),
                key=lambda k: int(k.split("_")[1].replace("dBFS", "")))
    sp = spearman([int(k.split("_")[1].replace("dBFS", "")) for k in lv], [R["A2"][k] for k in lv])
    print(f"\n   Spearman(score, noise level dBFS) = {sp:+.3f}  (-1 => louder bg scored 'real')")
    R["A2"]["spearman_score_vs_level"] = round(sp, 3)

    # ================================================================= A3
    print("\n" + "=" * 72 + "\nA3  SCORE vs SNR ACROSS THE BANK (first 24 s, stride 2)\n" + "=" * 72)
    bank = []
    for i in range(1, 11):
        bank.append(("real", f"real_{i:02d}",
                     os.path.join(root, "sonix_real", "sonix_real", "real", f"real_{i:02d}.wav")))
    for i in range(1, 11):
        bank.append(("fake", f"fake_{i:02d}",
                     os.path.join(root, "sonix_real", "sonix_real", "fake", f"fake_{i:02d}.wav")))
    mkb = os.path.join(P2, "mkb_real.wav")
    out_fake = os.path.join(P2, "output_fake.wav")
    if os.path.isfile(mkb):
        bank.append(("real", "mann_ki_baat", mkb))
    if os.path.isfile(out_fake):
        bank.append(("fake", "output_wav", out_fake))

    R["A3"] = {"rows": []}
    print(f"\n   {'clip':>14} {'lbl':>4}  {'dur_s':>6}  {'speechdB':>8}  {'SNR_dB':>7}  {'score':>7}")
    for lbl, nm, p in bank:
        if not os.path.isfile(p):
            print(f"   {nm:>14}  MISSING {p}")
            continue
        a = load(p)
        a24 = a[:16000 * 24]
        sm, nsp, ntot = score_wav(a24, stride=2)
        d = len(a) / 16000.0
        sdb = float(20.0 * np.log10(AN.speech_rms(a24, 16000) + 1e-12))
        snr = float(snr_db(a24))
        print(f"   {nm:>14} {lbl:>4}  {d:6.1f}  {sdb:8.1f}  {snr:7.1f}  {sm:7.3f}   {el()}")
        R["A3"]["rows"].append(dict(clip=nm, label=lbl, dur_s=round(d, 1),
                                    speech_db=round(sdb, 1), snr_db=round(snr, 1),
                                    score=round(sm, 4)))
    rows = R["A3"]["rows"]
    if rows:
        sc = [r["score"] for r in rows]
        R["A3"]["corr"] = {
            "pearson_score_vs_SNR":       round(pearson(sc, [r["snr_db"] for r in rows]), 3),
            "spearman_score_vs_SNR":      round(spearman(sc, [r["snr_db"] for r in rows]), 3),
            "pearson_score_vs_duration":  round(pearson(sc, [r["dur_s"] for r in rows]), 3),
            "pearson_score_vs_loudness":  round(pearson(sc, [r["speech_db"] for r in rows]), 3),
            "pearson_score_vs_truelabel": round(pearson(sc, [1 if r["label"] == "fake" else 0
                                                             for r in rows]), 3),
        }
        print("\n   correlations of head.pt score with:")
        for k, v in R["A3"]["corr"].items():
            print(f"      {k:30s} {v:+.3f}")

    # ================================================================= A4
    print("\n" + "=" * 72 + "\nA4  BACKGROUND-ONLY  (speech frames removed, then scored)\n" + "=" * 72)
    R["A4"] = {"rows": []}
    FR = 1024
    print(f"\n   {'clip':>14} {'lbl':>4}  {'bg_seconds':>10}  {'score':>7}")
    for lbl, nm, p in bank:
        if not os.path.isfile(p):
            continue
        a = load(p)
        fr = [a[i:i + FR] for i in range(0, len(a) - FR, FR)]
        bg = np.concatenate([f for f in fr if rms_dbfs(f) <= VAD_DB]) if fr else np.array([], np.float32)
        secs = len(bg) / 16000.0
        if secs < 4.0:
            print(f"   {nm:>14} {lbl:>4}  {secs:10.2f}  (too little non-speech)")
            R["A4"]["rows"].append(dict(clip=nm, label=lbl, bg_seconds=round(secs, 2), score=None))
            continue
        sm, _, _ = score_wav(bg.astype(np.float32))
        print(f"   {nm:>14} {lbl:>4}  {secs:10.2f}  {sm:7.3f}")
        R["A4"]["rows"].append(dict(clip=nm, label=lbl, bg_seconds=round(secs, 2), score=round(sm, 4)))

    dst = os.path.join(root, "docs", "handoff", "scripts", "tier_a_result.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, ensure_ascii=False)
    print(f"\n{el()}  DONE -> {dst}")


if __name__ == "__main__":
    main()
