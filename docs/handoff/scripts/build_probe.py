#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_probe.py -- SHORTCUT_INVESTIGATION.md section 6.1 probe clips, PAIR 1.

    real = sonix_real/sonix_real/real/real_03.wav   (clean laptop-mic genuine, English)
    fake = sonix_real/sonix_real/fake/fake_03.wav   (free voice-clone, direct download)

Writes 16 kHz mono PCM_16 to data/probe/clean/ so add_noise.py's wave-based
reader can consume them. No ffmpeg needed.

    python docs/handoff/scripts/build_probe.py [--repo-root .]
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.repo_root)

    src_real = os.path.join(root, "sonix_real", "sonix_real", "real", "real_03.wav")
    src_fake = os.path.join(root, "sonix_real", "sonix_real", "fake", "fake_03.wav")
    out = os.path.join(root, "data", "probe", "clean")
    os.makedirs(out, exist_ok=True)

    real = load_16k_mono(src_real)
    fake = load_16k_mono(src_fake)
    sr = TARGET_SR

    sf.write(os.path.join(out, "real_03.wav"),       real[10 * sr:40 * sr], sr, subtype="PCM_16")
    sf.write(os.path.join(out, "real_03_short.wav"), real[10 * sr:19 * sr], sr, subtype="PCM_16")
    sf.write(os.path.join(out, "fake_03.wav"),       fake,                  sr, subtype="PCM_16")

    for fn in ("real_03.wav", "real_03_short.wav", "fake_03.wav"):
        i = sf.info(os.path.join(out, fn))
        print(f"{fn:20s} {i.samplerate} Hz  {i.channels} ch  {i.duration:6.2f} s  {i.subtype}")


if __name__ == "__main__":
    main()
