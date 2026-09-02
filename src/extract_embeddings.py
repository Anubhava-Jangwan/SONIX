#!/usr/bin/env python3
"""
extract_embeddings.py  --  SONIX / SIH26104

Cache frozen wav2vec2 embeddings to disk, once, so training and evaluation are
minutes instead of hours. This is the slow step in the whole project and the one
piece of code that runs UNATTENDED on three different machines by three different
people. So it is written to be dull and unbreakable:

  * fully self-contained -- the ONLY file you copy to Anubhav's / Navya's machine
  * resumable -- writes a shard every N files; on restart it skips finished shards
  * crash-tolerant -- one unreadable audio file is logged and skipped, never fatal
  * fp16 forward pass, frozen model, mean-pooled -> one 1024-dim vector per clip

WHAT IT PRODUCES
    <out>/<split>/shard_00000.npy         float16, (n, 1024)   the embeddings
    <out>/<split>/shard_00000.labels.npy  int8,    (n,)        1=spoof 0=bonafide
    <out>/<split>/shard_00000.files.txt    the filenames, one per line, in order
  Shard k always covers manifest rows [k*shard_size : (k+1)*shard_size). If some
  files in that block were unreadable the shard simply has fewer rows -- the three
  sidecars stay row-aligned, so labels never drift out of sync with embeddings.

USAGE
    # smoke test FIRST -- confirm the shape is (50, 1024) before any long run
    python extract_embeddings.py --split train --batch 8 --out embeddings/ --limit 50

    # the real overnight runs
    python extract_embeddings.py --split train --batch 8 --out embeddings/
    python extract_embeddings.py --split dev   --batch 8 --out embeddings/
    python extract_embeddings.py --split eval  --batch 8 --out embeddings/   # Anubhav, --batch 4

    # In-the-Wild (cross-dataset test); point --itw-root at the folder with meta.csv
    python extract_embeddings.py --split itw --batch 8 --out embeddings/ --itw-root In-the-Wild/

If you hit CUDA out-of-memory, lower --batch (8 -> 4). Nothing else changes.

Datasets:  ASVspoof 2019 LA (train/dev/eval)  ·  In-the-Wild (itw, eval only)
Model:     facebook/wav2vec2-xls-r-300m  (or --model microsoft/wavlm-large)
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np

TARGET_SR = 16000
TARGET_LEN = 64000          # exactly 4.0 seconds at 16 kHz
EMB_DIM = 1024

# --- ASVspoof 2019 LA layout ------------------------------------------------
PROTOCOL_FILE = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
FLAC_DIR = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}


# ===========================================================================
# Manifest building  (filename, path, label)   label: 1=spoof, 0=bonafide
# ===========================================================================
def build_manifest(split: str, args) -> list[tuple[str, str, int]]:
    if getattr(args, "audio_dir", None):
        return _build_manifest_dir(args.audio_dir)
    if split == "itw":
        return _build_manifest_itw(args.itw_root)
    return _build_manifest_asvspoof(split, args.data_root)


def _build_manifest_dir(audio_dir):
    """Extract EVERY .flac/.wav in a folder directly -- no protocol needed.
    Labels are stored as a placeholder (0) and joined later, at scoring time, from
    the dataset's own key file. Use this for ASVspoof 2021 DF and any dataset where
    you just have a folder of audio. Filenames are saved in the shard sidecars, so
    the labels can be matched back by filename afterwards."""
    root = Path(audio_dir)
    if not root.exists():
        sys.exit(f"FATAL: --audio-dir not found: {root.resolve()}")
    files = sorted(list(root.glob("*.flac")) + list(root.glob("*.wav")))
    if not files:
        sys.exit(f"FATAL: no .flac or .wav files directly in {root.resolve()}")
    return [(f.stem, str(f), 0) for f in files]


def _build_manifest_asvspoof(split, data_root):
    root = Path(data_root)
    proto = root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILE[split]
    if not proto.exists():
        sys.exit(f"FATAL: protocol file not found: {proto}\n"
                 f"       Is --data-root correct? (got {root.resolve()})")
    flac_root = root / FLAC_DIR[split] / "flac"

    rows = []
    with open(proto) as fh:
        for ln, line in enumerate(fh, 1):
            parts = line.split()
            if len(parts) < 5:
                print(f"  ! skipping malformed protocol line {ln}: {line!r}")
                continue
            filename, label = parts[1], parts[4]
            if label not in ("bonafide", "spoof"):
                print(f"  ! unexpected label {label!r} on line {ln}; skipping")
                continue
            rows.append((filename, str(flac_root / f"{filename}.flac"),
                         1 if label == "spoof" else 0))
    # deterministic order -> shard boundaries are reproducible across machines
    rows.sort(key=lambda r: r[0])
    return rows


def _build_manifest_itw(itw_root):
    """In-the-Wild (mueller91). meta.csv columns: file, speaker, label.
    label spelling varies ('bona-fide'/'bonafide' vs 'spoof'); handled here."""
    if not itw_root:
        sys.exit("FATAL: --split itw needs --itw-root pointing at the folder "
                 "that contains meta.csv and the .wav files.")
    root = Path(itw_root)
    meta = root / "meta.csv"
    if not meta.exists():
        # some releases nest audio under release_in_the_wild/
        alt = root / "release_in_the_wild" / "meta.csv"
        if alt.exists():
            meta, root = alt, alt.parent
        else:
            sys.exit(f"FATAL: meta.csv not found under {root.resolve()}")

    import csv
    rows = []
    with open(meta, newline="") as fh:
        reader = csv.DictReader(fh)
        # tolerate different header capitalisations
        cols = {c.lower(): c for c in reader.fieldnames or []}
        fcol = cols.get("file") or cols.get("filename")
        lcol = cols.get("label")
        if not fcol or not lcol:
            sys.exit(f"FATAL: meta.csv headers not understood: {reader.fieldnames}")
        for r in reader:
            fn = r[fcol].strip()
            lab = r[lcol].strip().lower().replace("_", "-")
            is_spoof = 1 if lab in ("spoof", "fake") else 0  # else bona-fide
            rows.append((fn, str(root / fn), is_spoof))
    rows.sort(key=lambda r: r[0])
    return rows


# ===========================================================================
# Audio loading  ->  exactly 64,000 samples, 16 kHz, mono, float32
# ===========================================================================
def load_audio_fixed(path: str) -> np.ndarray:
    """Decode to exactly TARGET_LEN samples, 16 kHz mono float32.

    Tries ffmpeg first (tolerant of odd containers/codecs -- needed for some
    DF21 and In-the-Wild files), then falls back to soundfile. Previously a
    missing ffmpeg made EVERY file fail and be skipped, which looked like
    "31779 bad files" rather than "ffmpeg is not installed".
    """
    wav = None

    try:
        import subprocess
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
             "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(TARGET_SR), "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        wav = np.frombuffer(raw, dtype=np.float32)
        if wav.size == 0:
            wav = None
    except (OSError, FileNotFoundError, subprocess.SubprocessError):
        wav = None                      # ffmpeg missing or failed on this file

    if wav is None:                     # fallback: soundfile + our resampler
        import soundfile as sf
        w, sr = sf.read(path, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        if sr != TARGET_SR:
            w = _resample(w, sr, TARGET_SR)
        wav = w

    if len(wav) >= TARGET_LEN:
        wav = wav[:TARGET_LEN]
    else:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))
    return np.ascontiguousarray(wav, dtype=np.float32)


def _resample(wav, sr_in, sr_out):
    # ASVspoof and In-the-Wild are already 16 kHz, so this rarely runs. Use
    # torchaudio's high-quality resampler when available; fall back to linear.
    try:
        import torch
        import torchaudio.functional as AF
        t = torch.from_numpy(wav).float().unsqueeze(0)
        return AF.resample(t, sr_in, sr_out).squeeze(0).numpy()
    except Exception:
        n_out = int(round(len(wav) * sr_out / sr_in))
        xp = np.linspace(0, 1, num=len(wav), endpoint=False)
        x = np.linspace(0, 1, num=n_out, endpoint=False)
        return np.interp(x, xp, wav).astype(np.float32)


# ===========================================================================
# Frozen front-end.  Kept behind two functions so a test can stub them out
# without a GPU or the 1.2 GB model download.
# ===========================================================================
def load_frontend(model_name: str, device: str):
    """Return an opaque handle passed straight back to embed_batch()."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    print(f"Loading front-end {model_name} on {device} ...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    use_half = device.startswith("cuda")
    if use_half:
        model.half()                       # fp16 forward -> fits 4-6 GB easily
    for p in model.parameters():           # frozen: no grad, ever
        p.requires_grad_(False)
    return {"model": model, "fe": fe, "device": device,
            "dtype": torch.float16 if use_half else torch.float32, "torch": torch}


def embed_batch(frontend, wavs: list[np.ndarray]) -> np.ndarray:
    """List of (64000,) float32 waveforms -> float16 array (len(wavs), 1024).
    Mean-pooled over the time axis of last_hidden_state."""
    torch = frontend["torch"]
    fe, model, device, dtype = (frontend["fe"], frontend["model"],
                                frontend["device"], frontend["dtype"])
    inputs = fe([w for w in wavs], sampling_rate=TARGET_SR,
                return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device=device, dtype=dtype)
    kw = {}
    if "attention_mask" in inputs:
        kw["attention_mask"] = inputs["attention_mask"].to(device)
    with torch.no_grad():
        hidden = model(input_values, **kw).last_hidden_state   # (b, T, 1024)
        pooled = hidden.mean(dim=1)                             # (b, 1024)
    return pooled.float().cpu().numpy().astype(np.float16)


# ===========================================================================
# Shard I/O  (atomic writes so a killed process never leaves a half shard)
# ===========================================================================
def shard_paths(out_dir: Path, idx: int):
    stem = out_dir / f"shard_{idx:05d}"
    return (Path(f"{stem}.npy"), Path(f"{stem}.labels.npy"),
            Path(f"{stem}.files.txt"))


def _atomic_np_save(path, arr):
    # write via a file handle so np.save does NOT append .npy to our .tmp name
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def save_shard_atomic(emb_p, lab_p, files_p, emb, labels, files):
    # order matters: the embeddings file is the resume marker, so it goes last.
    # Die midway and the shard looks unfinished, so it is redone cleanly.
    _atomic_np_save(lab_p, labels)
    tmp = str(files_p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(files) + ("\n" if files else ""))
    os.replace(tmp, files_p)
    _atomic_np_save(emb_p, emb)


# ===========================================================================
# Main
# ===========================================================================
def run(args, _load_frontend=load_frontend, _embed_batch=embed_batch) -> int:
    # progress bar is optional -- degrade gracefully if tqdm is missing
    try:
        from tqdm import tqdm
    except Exception:
        class tqdm:  # noqa: N801 -- stand-in with the call surface we use
            """No-op bar: a missing tqdm must never break an overnight run."""

            def __init__(self, iterable=None, **kwargs):
                self._it = iterable

            def __iter__(self):
                return iter(self._it if self._it is not None else ())

            def update(self, n=1):
                pass

            def set_postfix(self, **kwargs):
                pass

            def close(self):
                pass

    split = args.split
    out_dir = Path(args.out) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(split, args)
    if args.limit:
        manifest = manifest[: args.limit]
    total = len(manifest)
    shard_size = args.shard_size
    n_shards = (total + shard_size - 1) // shard_size
    print(f"[{split}] {total} files -> {n_shards} shards of up to {shard_size} "
          f"in {out_dir}")

    # ---- figure out what's already done (resume) --------------------------
    todo_shards = []
    done_files = 0
    for s in range(n_shards):
        # all three sidecars must exist -- the embeddings file is written last
        # (see save_shard_atomic), so a half-written shard is retried, not skipped
        if all(p.exists() for p in shard_paths(out_dir, s)):
            done_files += min(shard_size, total - s * shard_size)
        else:
            todo_shards.append(s)
    if done_files:
        print(f"[{split}] resume: {done_files} files already cached in "
              f"{n_shards - len(todo_shards)} shards; {len(todo_shards)} to go")
    if not todo_shards:
        print(f"[{split}] nothing to do -- every shard already exists.")
        return 0

    # ---- load the model only if there is real work -----------------------
    device = args.device or _auto_device()
    if device == "cpu":
        print("WARNING: no CUDA device found. This will run on CPU and be very "
              "slow -- fine for a 50-file smoke test, not for a full split.")
    frontend = _load_frontend(args.model, device)

    succeeded = skipped_bad = 0
    pbar = tqdm(total=total, initial=done_files, unit="file", desc=f"{split}")

    for s in todo_shards:
        block = manifest[s * shard_size:(s + 1) * shard_size]
        emb_p, lab_p, files_p = shard_paths(out_dir, s)

        embs, labels, files = [], [], []
        # process the block in mini-batches of --batch
        for i in range(0, len(block), args.batch):
            mb = block[i:i + args.batch]
            wavs, keep = [], []
            for fn, path, lab in mb:
                try:
                    wavs.append(load_audio_fixed(path))
                    keep.append((fn, lab))
                except Exception as e:            # one bad file must not stop us
                    skipped_bad += 1
                    print(f"\n  ! SKIP {fn}: {type(e).__name__}: {e}")
                    pbar.update(1)
            if not wavs:
                continue
            try:
                vecs = _embed_batch(frontend, wavs)
            except Exception as e:
                # a whole-batch failure (e.g. transient OOM): log, skip block,
                # do NOT write the shard, so it retries cleanly next run
                skipped_bad += len(wavs)
                print(f"\n  ! BATCH FAILED in shard {s}: "
                      f"{type(e).__name__}: {e}")
                traceback.print_exc()
                for _ in wavs:
                    pbar.update(1)
                continue
            for (fn, lab), v in zip(keep, vecs):
                embs.append(v)
                labels.append(lab)
                files.append(fn)
                succeeded += 1
                pbar.update(1)

        emb_arr = (np.stack(embs).astype(np.float16) if embs
                   else np.empty((0, EMB_DIM), np.float16))
        save_shard_atomic(emb_p, lab_p, files_p, emb_arr,
                          np.asarray(labels, np.int8), files)
        pbar.set_postfix(ok=succeeded, bad=skipped_bad)

    pbar.close()
    print(f"\n[{split}] DONE. succeeded={succeeded}  skipped(bad)={skipped_bad}  "
          f"already_cached={done_files}  total={total}")

    # smoke-test convenience: show the shape the brief tells you to check
    emb0, _, _ = shard_paths(out_dir, todo_shards[0])
    if emb0.exists():
        print(f"[{split}] first written shard shape = {np.load(emb0).shape}  "
              f"(expect (*, {EMB_DIM}))")
    if skipped_bad > total * 0.02:
        print(f"[{split}] NOTE: {skipped_bad} files were skipped (>2%). Check the "
              f"SKIP lines above -- that is more bad files than expected.")
    return 0


def _auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA:", torch.cuda.get_device_name(0))
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_argparser():
    ap = argparse.ArgumentParser(description="Cache frozen wav2vec2 embeddings.")
    ap.add_argument("--split", required=True,
                    choices=["train", "dev", "eval", "itw"])
    ap.add_argument("--batch", type=int, default=8,
                    help="mini-batch for the forward pass (drop to 4 on OOM)")
    ap.add_argument("--out", default="outputs/embeddings",
                    help="output root; shards go in <out>/<split>/")
    ap.add_argument("--data-root", default="data/asvspoof19_la",
                    help="folder containing ASVspoof2019_LA_* (train/dev/eval)")
    ap.add_argument("--itw-root", default="data/in_the_wild",
                    help="In-the-Wild folder containing meta.csv (--split itw)")
    ap.add_argument("--audio-dir", default=None,
                    help="extract EVERY .flac/.wav in this folder directly, no "
                         "protocol needed (labels joined later at scoring). Use for "
                         "ASVspoof 2021 DF. Pair with any --split name for the output.")
    ap.add_argument("--model", default="facebook/wav2vec2-xls-r-300m",
                    help="HF model id (or microsoft/wavlm-large)")
    ap.add_argument("--shard-size", type=int, default=100,
                    help="files per shard (default 100)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N files (use 50 to smoke test)")
    ap.add_argument("--device", default=None,
                    help="force 'cuda' or 'cpu' (default: auto-detect)")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_argparser().parse_args()))
