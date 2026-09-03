#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_probe2.py -- SHORTCUT_INVESTIGATION.md section 6 probe clips, PAIR 2.

    real = sonix_real/sonix_real/real/<Mann Ki Baat ... 30th August 2026>.wav
           (genuine broadcast speech, Hindi -- the PM's radio address)
    fake = sonix_real/sonix_real/fake/output.wav
           (AI-generated clone of that voice, 24 kHz float)

Neither file is in git (sonix_real/ has only real_01..10 / fake_01..10 tracked);
both must be copied to the repo by hand before running this. Writes 16 kHz mono
PCM_16 to data/probe2/clean/.

    python docs/handoff/scripts/build_probe2.py [--repo-root .]
"""
import argparse
import glob
import os
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

TARGET_SR = 16000


def load_16k_mono(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = AF.resample(torch.from_numpy(wav).float().unsqueeze(0),
                          sr, TARGET_SR).squeeze(0).numpy()
    return wav.astype(np.float32)


def find_hindi_real(root):
    d = os.path.join(root, "sonix_real", "sonix_real", "real")
    # the only non 'real_NN.wav' file in that folder
    for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
        b = os.path.basename(p)
        if not b.lower().startswith("real_"):
            return p
    raise SystemExit("Hindi 'Mann Ki Baat' real clip not found in sonix_real/sonix_real/real/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.repo_root)

    src_real = find_hindi_real(root)
    src_fake = os.path.join(root, "sonix_real", "sonix_real", "fake", "output.wav")
    if not os.path.isfile(src_fake):
        raise SystemExit(f"missing {src_fake} -- copy output.wav into the repo first")
    out = os.path.join(root, "data", "probe2", "clean")
    os.makedirs(out, exist_ok=True)

    real = load_16k_mono(src_real)
    fake = load_16k_mono(src_fake)
    sr = TARGET_SR

    sf.write(os.path.join(out, "mkb_real.wav"),       real[10 * sr:40 * sr], sr, subtype="PCM_16")
    sf.write(os.path.join(out, "mkb_real_short.wav"), real[10 * sr:19 * sr], sr, subtype="PCM_16")
    sf.write(os.path.join(out, "output_fake.wav"),    fake,                  sr, subtype="PCM_16")

    print("source real:", os.path.basename(src_real))
    for fn in ("mkb_real.wav", "mkb_real_short.wav", "output_fake.wav"):
        i = sf.info(os.path.join(out, fn))
        print(f"{fn:20s} {i.samplerate} Hz  {i.channels} ch  {i.duration:6.2f} s  {i.subtype}")


if __name__ == "__main__":
    main()
