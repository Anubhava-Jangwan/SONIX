#!/usr/bin/env python3
"""
make_conditioned.py  --  SONIX / SIH26104        v4 stage S3

ONE conditioned copy of a training folder, drawn from ONE shared sampler, so
every corpus and both classes see the same channel distribution.

WHY
    Every earlier augmentation here was one-sided: g711 / rawboost / rirmusan
    were applied to ASVspoof only, and MMS-TTS / IndicSynth "aug" to fakes only.
    A model then learns "conditioned = ASVspoof" or "conditioned = fake" instead
    of learning to ignore the channel -- the RawBoost-marker failure in
    docs/V4_BUILD_BRIEF.md. The fix is not more augmentation, it is the SAME
    augmentation everywhere: this script, same settings, on every training
    folder (real and fake, English and Indic).

WHAT EACH CLIP GETS (seeded from the filename, so reruns are identical)
    1. crop to the first --crop seconds (the extractor only ever sees 4.0 s)
    2. one ACOUSTIC effect, uniform over:
         rir       convolve with a random RIRS_NOISES room impulse response
         musan     MUSAN noise 0-15 dB / music 5-15 dB / babble 13-20 dB SNR
         pink      pink noise at 5-20 dB SNR
         rawboost  RawBoost algo 5 (Tak et al. 2022), make_rawboost.py
    3. one CHANNEL codec, uniform over:
         none      no codec
         g711      mu-law at 8 kHz (landline / contact-centre PSTN leg)
         amrwb     AMR-WB 6.6 / 12.65 / 23.85 kbps (VoLTE / mobile HD voice)
         opus      Opus 8 / 12 / 16 / 24 kbps (Meet, WhatsApp, WebRTC VoIP)
       Acoustic first, codec second -- the room happens before the network.
    Output: 16 kHz mono PCM16 wav, SAME filename stem as the source, so the
    conditioned copy shares its recording id and lands on the same side of the
    dev carve as its clean original.

LABELS
    --audio-dir extraction labels every row 0. A folder holding ONE class is
    stamped afterwards like any other. ASVspoof holds both, so pass --protocol
    and the output is split into <out>/bonafide and <out>/spoof -- extract and
    stamp each separately.

ROOT NAMES FOR train.py (the "_cond" suffix is the channel tag)
    embeddings_v2_indicvoices_tr_cond     embeddings_v2_indicsynth_cond
    embeddings_v2_mlaad_cond              embeddings_v2_mms_tts_cond
    embeddings_v2_librisevoc_gt_cond      embeddings_v2_librisevoc_<voc>_cond
    embeddings_v2_asvspoof19_cond_bona    embeddings_v2_asvspoof19_cond_spoof

USAGE (Windows cmd, on the box that holds the audio)
    python make_conditioned.py --preflight
    python make_conditioned.py --audio-dir F:\\indicsynth --out-dir D:\\SONIX\\cond\\indicsynth ^
        --musan-root D:\\musan --rir-root D:\\RIRS_NOISES --workers 8
    # a vocoder folder uses the SAME list as its clean extraction
    python make_conditioned.py --audio-dir F:\\librisevoc\\wav\\melgan ^
        --file-list docs\\v4_splits\\librisevoc\\melgan.txt --out-dir D:\\SONIX\\cond\\librisevoc_melgan ...
    # ASVspoof: split by label
    python make_conditioned.py --audio-dir F:\\asvspoof19_la\\ASVspoof2019_LA_train\\flac ^
        --protocol F:\\asvspoof19_la\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt ^
        --out-dir D:\\SONIX\\cond\\asvspoof19 ...
    # before extracting: every folder must share one sampler
    python make_conditioned.py --verify D:\\SONIX\\cond\\indicsynth D:\\SONIX\\cond\\mlaad ...

TRAIN ONLY. Never condition eval, In-the-Wild, sonix_real or any holdout.
Resumable: finished files are skipped. A failed file is logged, never faked.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SR = 16000
AUDIO = (".wav", ".flac")
SAMPLER_VERSION = 1

ACOUSTIC = ("rir", "musan", "pink", "rawboost")
CODECS = ("none", "g711", "amrwb", "opus")
AMRWB_RATES = ("6.6k", "12.65k", "23.85k")
OPUS_RATES = ("8k", "12k", "16k", "24k")
MUSAN_SNR = {"noise": (0.0, 15.0), "music": (5.0, 15.0), "babble": (13.0, 20.0)}
PINK_SNR = (5.0, 20.0)
FFMPEG_ENCODERS = {"amrwb": "libvo_amrwbenc", "opus": "libopus"}


def sampler_config(crop):
    """Everything that decides a clip's condition. --verify compares this."""
    return {"version": SAMPLER_VERSION, "crop": crop, "acoustic": list(ACOUSTIC),
            "codecs": list(CODECS), "amrwb_rates": list(AMRWB_RATES),
            "opus_rates": list(OPUS_RATES), "musan_snr": MUSAN_SNR,
            "pink_snr": list(PINK_SNR), "rawboost_algo": 5}


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def clip_seed(seed, name):
    """Stable across machines and Python runs (unlike hash())."""
    return int(hashlib.sha1(f"{seed}:{name}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def read_head(path, seconds):
    """First `seconds` of a file as float32 mono 16 kHz. Reads only that much."""
    import soundfile as sf
    info = sf.info(str(path))
    frames = int(np.ceil(seconds * info.samplerate)) if seconds else -1
    x, sr = sf.read(str(path), frames=frames, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR and len(x) > 1:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(x[: int(seconds * SR)] if seconds else x, np.float32)


def load_any(path):
    return read_head(path, 0)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def fit(noise, n, rng):
    if noise.size == 0:
        return np.zeros(n, np.float32)
    if noise.size < n:
        noise = np.tile(noise, int(np.ceil(n / noise.size)))
    s = int(rng.integers(0, noise.size - n + 1))
    return noise[s:s + n]


def mix(x, noise, snr_db, rng):
    s, n = rms(x), fit(noise, len(x), rng)
    nr = rms(n)
    if s < 1e-8 or nr < 1e-8:
        return x
    y = x + n * (s / (10 ** (snr_db / 20.0)) / nr)
    return (y * (s / max(rms(y), 1e-8))).astype(np.float32)


def pink(n, rng):
    """1/f noise via spectral shaping."""
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    k = np.arange(len(f))
    k[0] = 1
    y = np.fft.irfft(f / np.sqrt(k), n)
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def reverb(x, h):
    from scipy.signal import fftconvolve
    peak = int(np.argmax(np.abs(h)))
    h = h[max(0, peak - 32): max(0, peak - 32) + SR]
    e = float(np.sqrt(np.sum(np.square(h, dtype=np.float64))))
    if e < 1e-8:
        return x
    y = fftconvolve(x, h / e)[: len(x)]
    return (y * (rms(x) / max(rms(y), 1e-8))).astype(np.float32)


def g711(x):
    """mu-law at 8 kHz, in-process: 16k -> 8k -> 8-bit mu-law -> 8k -> 16k."""
    from scipy.signal import resample_poly
    y = np.clip(resample_poly(x, 1, 2), -1.0, 1.0)
    mu = 255.0
    q = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round((q + 1) / 2 * 255) / 255 * 2 - 1           # 8-bit
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return resample_poly(y, 2, 1)[: len(x)].astype(np.float32)


def ffmpeg_codec(x, codec, rate):
    """Encode + decode through ffmpeg pipes; nothing touches the disk."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    if codec == "amrwb":
        enc = ["-c:a", "libvo_amrwbenc", "-b:a", rate, "-f", "amr"]
    else:
        enc = ["-c:a", "libopus", "-b:a", rate, "-application", "voip", "-f", "ogg"]
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    a = subprocess.run(base + ["-f", "s16le", "-ar", str(SR), "-ac", "1",
                               "-i", "pipe:0"] + enc + ["pipe:1"],
                       input=pcm, capture_output=True, check=True)
    b = subprocess.run(base + ["-i", "pipe:0", "-f", "s16le", "-ar", str(SR),
                               "-ac", "1", "pipe:1"],
                       input=a.stdout, capture_output=True, check=True)
    y = np.frombuffer(b.stdout, "<i2").astype(np.float32) / 32768.0
    if len(y) >= len(x):
        return y[: len(x)]
    return np.pad(y, (0, len(x) - len(y)))


# ---------------------------------------------------------------------------
# the sampler -- identical for every folder
# ---------------------------------------------------------------------------
def draw(rng, pools):
    """-> dict describing this clip's condition. Pure function of rng + pools."""
    c = {"acoustic": ACOUSTIC[int(rng.integers(len(ACOUSTIC)))],
         "codec": CODECS[int(rng.integers(len(CODECS)))]}
    if c["acoustic"] == "rir":
        c["rir"] = pools["rir"][int(rng.integers(len(pools["rir"])))]
    elif c["acoustic"] == "musan":
        kind = ("noise", "music", "babble")[int(rng.integers(3))]
        lo, hi = MUSAN_SNR[kind]
        c.update(musan_kind=kind, snr=round(float(rng.uniform(lo, hi)), 2))
        if kind == "babble":
            k = int(rng.integers(3, 8))
            c["musan"] = [pools["speech"][int(rng.integers(len(pools["speech"])))]
                          for _ in range(k)]
        else:
            c["musan"] = [pools[kind][int(rng.integers(len(pools[kind])))]]
    elif c["acoustic"] == "pink":
        c["snr"] = round(float(rng.uniform(*PINK_SNR)), 2)
    if c["codec"] == "amrwb":
        c["rate"] = AMRWB_RATES[int(rng.integers(len(AMRWB_RATES)))]
    elif c["codec"] == "opus":
        c["rate"] = OPUS_RATES[int(rng.integers(len(OPUS_RATES)))]
    return c


def condition(x, c, rng):
    if c["acoustic"] == "rir":
        x = reverb(x, load_any(c["rir"]))
    elif c["acoustic"] == "musan":
        if c["musan_kind"] == "babble":
            acc = np.zeros(len(x), np.float32)
            for p in c["musan"]:
                acc += fit(load_any(p), len(x), rng)
            noise = acc
        else:
            noise = load_any(c["musan"][0])
        x = mix(x, noise, c["snr"], rng)
    elif c["acoustic"] == "pink":
        x = mix(x, pink(len(x), rng), c["snr"], rng)
    elif c["acoustic"] == "rawboost":
        from make_rawboost import rawboost
        x = rawboost(x.astype(np.float64), SR, algo=5).astype(np.float32)

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)

    if c["codec"] == "g711":
        x = g711(x)
    elif c["codec"] in ("amrwb", "opus"):
        x = ffmpeg_codec(x, c["codec"], c["rate"])
    return np.clip(x, -1.0, 1.0).astype(np.float32)


_POOLS = None


def _init(pools):
    """Worker initializer: the pools (60k RIR paths) cross the process boundary
    ONCE per worker, not once per clip -- Windows spawns, and pickling them into
    every job would cost more than the audio work."""
    global _POOLS
    _POOLS = pools


def work(job):
    """One clip. Returns (name, status, condition-dict). Never raises."""
    src, dst, seed, crop = job
    pools = _POOLS
    name = Path(src).name
    try:
        rng = np.random.default_rng(seed)
        np.random.seed(seed % (2 ** 32))       # make_rawboost uses the global rng
        c = draw(rng, pools)
        if Path(dst).exists():
            return name, "skipped", c
        x = read_head(src, crop)
        if x.size == 0:
            return name, "empty", c
        y = condition(x, c, rng)
        import soundfile as sf
        part = f"{dst}.part"
        sf.write(part, y, SR, subtype="PCM_16", format="WAV")
        os.replace(part, dst)
        return name, "ok", c
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", "replace").strip()[:160]
        return name, f"error: ffmpeg {err}", None
    except Exception as e:                      # noqa: BLE001 -- log, never fake
        return name, f"error: {type(e).__name__}: {e}", None


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def files_in(d, file_list=None):
    d = Path(d)
    fs = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in AUDIO)
    if file_list:
        want = [ln.strip() for ln in Path(file_list).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        by = {f.stem: f for f in fs}
        miss = [w for w in want if Path(w).stem not in by]
        if miss:
            sys.exit(f"FATAL: {len(miss)} names in {file_list} are not in {d} "
                     f"(e.g. {miss[:3]})")
        keep = {Path(w).stem for w in want}
        fs = [f for f in fs if f.stem in keep]
    return fs


def pools_from(musan_root, rir_root):
    def walk(root, *parts):
        base = Path(root).joinpath(*parts)
        out = sorted(str(p) for p in base.rglob("*") if p.suffix.lower() in AUDIO)
        if not out:
            sys.exit(f"FATAL: no audio under {base}")
        return out
    rir_base = Path(rir_root)
    real = rir_base / "real_rirs_isotropic_noises"
    sim = rir_base / "simulated_rirs"
    rirs = []
    for b in (real, sim):
        if b.exists():
            rirs += [str(p) for p in b.rglob("*") if p.suffix.lower() == ".wav"
                     and "noise" not in p.name.lower()]
    if not rirs:
        rirs = walk(rir_root)
    return {"rir": sorted(rirs), "noise": walk(musan_root, "noise"),
            "music": walk(musan_root, "music"), "speech": walk(musan_root, "speech")}


def preflight():
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        sys.exit("FATAL: ffmpeg not on PATH. Windows: winget install Gyan.FFmpeg "
                 "(the FULL build, not essentials), then reopen the terminal.")
    missing = [f"{c} needs {e}" for c, e in FFMPEG_ENCODERS.items() if e not in out]
    if missing:
        sys.exit(f"FATAL: ffmpeg lacks {', '.join(missing)}. Install the FULL "
                 f"build (winget install Gyan.FFmpeg). Dropping a codec is NOT an "
                 f"option: every folder must share one sampler.")
    try:
        import scipy  # noqa: F401
        import soundfile  # noqa: F401
        from make_rawboost import rawboost  # noqa: F401
    except ImportError as e:
        sys.exit(f"FATAL: {e}. Run from the repo root with the .venv python.")
    x = (0.1 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)).astype(np.float32)
    for codec, rate in (("amrwb", "12.65k"), ("opus", "12k")):
        y = ffmpeg_codec(x, codec, rate)
        assert len(y) == len(x) and np.isfinite(y).all(), codec
    assert len(g711(x)) == len(x)
    print("preflight OK: ffmpeg libvo_amrwbenc + libopus, scipy, soundfile, "
          "make_rawboost all present.")


def verify(dirs):
    """Every folder must share one sampler, and the draws must look alike."""
    cfgs, bad = {}, 0
    print(f"  {'folder':<34}{'clips':>7}  " +
          "  ".join(f"{a:>8}" for a in ACOUSTIC) + "  " +
          "  ".join(f"{c:>6}" for c in CODECS))
    for d in dirs:
        d = Path(d)
        subs = [s for s in (d / "bonafide", d / "spoof") if s.exists()] or [d]
        for s in subs:
            mp = s / "conditioning.json"
            if not mp.exists():
                print(f"  {s}: no conditioning.json")
                bad += 1
                continue
            m = json.loads(mp.read_text())
            cfgs[str(s)] = m["config_hash"]
            rows = list(csv.DictReader(open(s / "conditions.csv", encoding="utf-8")))
            n = len(rows)
            a = [sum(r["acoustic"] == k for r in rows) / max(n, 1) for k in ACOUSTIC]
            c = [sum(r["codec"] == k for r in rows) / max(n, 1) for k in CODECS]
            print(f"  {str(s)[-34:]:<34}{n:>7}  " +
                  "  ".join(f"{v:>8.1%}" for v in a) + "  " +
                  "  ".join(f"{v:>6.1%}" for v in c))
    hashes = set(cfgs.values())
    if len(hashes) > 1:
        sys.exit(f"FATAL: folders were made with DIFFERENT samplers: {cfgs}. "
                 f"Re-run the odd ones out with the same script version.")
    if bad:
        sys.exit(f"FATAL: {bad} folder(s) have no conditioning record.")
    print(f"OK: {len(cfgs)} folder(s), one sampler ({hashes.pop() if hashes else '-'}).")


def run_one(args, files, out, pools, cfg, label=None):
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(out / (f.stem + ".wav")), clip_seed(args.seed, f.stem),
             args.crop) for f in files]
    tag = f" [{label}]" if label else ""
    print(f"  {len(jobs)} clips -> {out}{tag}", flush=True)
    rows, fails, counts = [], [], {"ok": 0, "skipped": 0}
    ex = None
    if args.workers > 1:
        ex = ProcessPoolExecutor(args.workers, initializer=_init, initargs=(pools,))
        it = ex.map(work, jobs, chunksize=16)
    else:
        _init(pools)
        it = map(work, jobs)
    for k, (name, st, c) in enumerate(it, 1):
        if st in counts:
            counts[st] += 1
            rows.append({"file": name, "acoustic": c["acoustic"], "codec": c["codec"],
                         "rate": c.get("rate", ""), "snr": c.get("snr", ""),
                         "detail": c.get("musan_kind", "") or
                         (Path(c["rir"]).name if "rir" in c else "")})
        else:
            fails.append((name, st))
        if k % 2000 == 0:
            print(f"    {k}/{len(jobs)}  ok={counts['ok']} skipped={counts['skipped']} "
                  f"failed={len(fails)}", flush=True)
    if ex:
        ex.shutdown()
    with open(out / "conditions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["file", "acoustic", "codec", "rate", "snr", "detail"])
        w.writeheader()
        w.writerows(rows)
    (out / "conditioning.json").write_text(json.dumps(
        {"config": cfg, "config_hash": config_hash(cfg), "seed": args.seed,
         "source": str(Path(args.audio_dir).resolve()), "label": label,
         "file_list": args.file_list, "n": len(rows), "failed": len(fails)}, indent=1))
    for name, st in fails[:10]:
        print(f"    FAILED {name}: {st}")
    print(f"  done{tag}: ok={counts['ok']} skipped={counts['skipped']} "
          f"failed={len(fails)}")
    return len(fails)


def main() -> int:
    ap = argparse.ArgumentParser(description="v4 S3: symmetric conditioned copy")
    ap.add_argument("--audio-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--file-list", default=None,
                    help="same list as the clean extraction (LibriSeVoc vocoders)")
    ap.add_argument("--protocol", default=None,
                    help="ASVspoof CM protocol: split output into bonafide/ and spoof/")
    ap.add_argument("--musan-root")
    ap.add_argument("--rir-root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=float, default=4.0,
                    help="seconds kept from the start (the extractor's window)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N files")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--verify", nargs="+", metavar="OUT_DIR")
    args = ap.parse_args()

    if args.verify:
        verify(args.verify)
        return 0
    preflight()
    if args.preflight:
        return 0
    for need in ("audio_dir", "out_dir", "musan_root", "rir_root"):
        if not getattr(args, need):
            sys.exit(f"FATAL: --{need.replace('_', '-')} is required")

    low = str(Path(args.audio_dir).resolve()).lower()
    for held in ("in_the_wild", "_ho", "holdout", "sonix_real", "_eval", "\\eval"):
        if held in low:
            sys.exit(f"FATAL: {args.audio_dir} looks like held-out data ({held!r}). "
                     f"S3 conditions TRAINING folders only.")

    cfg = sampler_config(args.crop)
    pools = pools_from(args.musan_root, args.rir_root)
    print(f"sampler {config_hash(cfg)}  seed {args.seed}  rir={len(pools['rir'])} "
          f"noise={len(pools['noise'])} music={len(pools['music'])} "
          f"speech={len(pools['speech'])}  workers={args.workers}")

    files = files_in(args.audio_dir, args.file_list)
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"FATAL: no .wav/.flac directly inside {args.audio_dir}")
    out = Path(args.out_dir)

    if args.protocol:
        lab = {}
        for ln in Path(args.protocol).read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) >= 5:
                lab[p[1]] = p[4]
        miss = [f.stem for f in files if f.stem not in lab]
        if miss:
            sys.exit(f"FATAL: {len(miss)} files not in {args.protocol} (e.g. {miss[:3]})")
        failed = 0
        for cls in ("bonafide", "spoof"):
            part = [f for f in files if lab[f.stem] == cls]
            failed += run_one(args, part, out / cls, pools, cfg, label=cls)
    else:
        failed = run_one(args, files, out, pools, cfg)

    if failed:
        print(f"INCOMPLETE: {failed} clip(s) failed. Re-run the same command; "
              f"finished clips are skipped. Exit code 2.")
        return 2
    print("next: extract with src\\extract_embeddings_v2.py --audio-dir <out> "
          "--layers 5,6,24, then stamp the SAME label as the source folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
