#!/usr/bin/env python3

import argparse
import io
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

TARGET_SR = 16000
TARGET_LEN = 64000          # exactly 4 seconds
EMB_DIM = 1024

# IndicVoices -> genuine human speech
BONAFIDE_LABEL = 0


# ------------------------------------------------------------
# Audio -> exact SONIX format
# ------------------------------------------------------------

def load_audio_bytes(audio_bytes: bytes, sample_rate_hint=None) -> np.ndarray:
    """
    Decode FLAC/WAV bytes and convert to:
        mono
        16 kHz
        float32
        exactly 64,000 samples
    """

    wav, sr = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    # Stereo / multichannel -> mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Resample to 16 kHz
    if sr != TARGET_SR:
        wav = resample_poly(
            wav,
            TARGET_SR,
            sr
        ).astype(np.float32)

    # Exactly 4 seconds, matching SONIX
    if len(wav) >= TARGET_LEN:
        wav = wav[:TARGET_LEN]
    else:
        wav = np.pad(
            wav,
            (0, TARGET_LEN - len(wav))
        )

    return np.ascontiguousarray(
        wav,
        dtype=np.float32
    )


# ------------------------------------------------------------
# XLS-R frontend
# ------------------------------------------------------------

def load_frontend(device: str):
    print(f"Loading XLS-R on {device}...", flush=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.eval()
    model.to(device)

    use_half = device.startswith("cuda")

    if use_half:
        model.half()

    # Frozen frontend
    for param in model.parameters():
        param.requires_grad_(False)

    print("XLS-R loaded.", flush=True)

    return feature_extractor, model


def embed_batch(feature_extractor, model, device, wavs):
    """
    Match SONIX:
        4 sec @ 16 kHz
        XLS-R
        mean pool
        1024 dimensions
        float16 output
    """

    inputs = feature_extractor(
        wavs,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs["input_values"].to(
        device=device,
        dtype=torch.float16 if device.startswith("cuda")
        else torch.float32
    )

    kwargs = {}

    if "attention_mask" in inputs:
        kwargs["attention_mask"] = inputs["attention_mask"].to(device)

    with torch.inference_mode():

        hidden = model(
            input_values,
            **kwargs
        ).last_hidden_state

        # Exactly what SONIX uses
        pooled = hidden.mean(dim=1)

    return (
        pooled
        .float()
        .cpu()
        .numpy()
        .astype(np.float16)
    )


# ------------------------------------------------------------
# Atomic shard writing
# ------------------------------------------------------------

def atomic_np_save(path: Path, array: np.ndarray):

    tmp = str(path) + ".tmp"

    with open(tmp, "wb") as fh:
        np.save(fh, array)

    os.replace(tmp, path)


def save_shard(
    output_dir: Path,
    shard_idx: int,
    embeddings,
    labels,
    files
):

    emb_path = output_dir / f"shard_{shard_idx:05d}.npy"
    lab_path = output_dir / f"shard_{shard_idx:05d}.labels.npy"
    file_path = output_dir / f"shard_{shard_idx:05d}.files.txt"

    emb_array = (
        np.stack(embeddings).astype(np.float16)
        if embeddings
        else np.empty((0, EMB_DIM), dtype=np.float16)
    )

    label_array = np.asarray(
        labels,
        dtype=np.int8
    )

    atomic_np_save(
        emb_path,
        emb_array
    )

    atomic_np_save(
        lab_path,
        label_array
    )

    tmp = str(file_path) + ".tmp"

    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(
            "\n".join(files)
            + ("\n" if files else "")
        )

    os.replace(tmp, file_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Stream IndicVoices and create SONIX-compatible embeddings."
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=[
            "hindi",
            "bengali",
            "marathi",
            "gujarati",
            "punjabi",
            "tamil",
            "telugu",
            "kannada",
        ],
    )

    parser.add_argument(
        "--split",
        default="train"
    )

    parser.add_argument(
        "--hours",
        type=float,
        default=10.0,
        help="Maximum raw IndicVoices duration per language."
    )

    parser.add_argument(
        "--max-per-speaker",
        type=int,
        default=20,
        help="Maximum recordings selected from one speaker."
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Debug limit. 0 = no sample-count limit."
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=100
    )

    parser.add_argument(
        "--out",
        default="outputs/embeddings_indicvoices"
    )

    parser.add_argument(
        "--device",
        default=None
    )

    args = parser.parse_args()

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("SONIX - INDICVOICES PRODUCTION EMBEDDING EXTRACTOR")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Hours/language: {args.hours}")
    print(f"Max recordings/speaker: {args.max_per_speaker}")
    print(f"Batch size: {args.batch}")
    print()

    feature_extractor, model = load_frontend(device)

    root = Path(args.out)
    root.mkdir(
        parents=True,
        exist_ok=True
    )

    total_saved = 0

    # --------------------------------------------------------
    # One language at a time
    # --------------------------------------------------------

    for language in args.languages:

        print()
        print("=" * 70)
        print(f"LANGUAGE: {language}")
        print("=" * 70)

        output_dir = root / language
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        max_seconds = args.hours * 3600.0

        duration_selected = 0.0
        sample_count = 0

        speaker_counts = {}

        embeddings = []
        labels = []
        files = []

        shard_idx = 0

        # Load streaming dataset
        print(f"Connecting to IndicVoices [{language}]...")

        ds = load_dataset(
            "ai4bharat/IndicVoices",
            language,
            split=args.split,
            streaming=True
        )

        # Do NOT let datasets invoke TorchCodec
        ds = ds.decode(False)

        iterator = iter(ds)

        progress = tqdm(
            total=None,
            desc=language,
            unit="sample"
        )

        try:

            while duration_selected < max_seconds:

                if (
                    args.max_samples
                    and sample_count >= args.max_samples
                ):
                    break

                try:
                    sample = next(iterator)

                except StopIteration:
                    break

                except Exception as exc:
                    print(
                        f"\n! STREAM ERROR: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break

                speaker = sample.get(
                    "speaker_id",
                    "UNKNOWN"
                )

                duration = float(
                    sample.get(
                        "duration",
                        0.0
                    )
                    or 0.0
                )

                # Ignore unusable metadata
                if duration <= 0:
                    continue

                # Speaker balancing
                used_by_speaker = speaker_counts.get(
                    speaker,
                    0
                )

                if used_by_speaker >= args.max_per_speaker:
                    continue

                audio_info = sample.get(
                    "audio_filepath"
                )

                if not isinstance(audio_info, dict):
                    continue

                audio_bytes = audio_info.get(
                    "bytes"
                )

                if not audio_bytes:
                    continue

                try:

                    wav = load_audio_bytes(
                        audio_bytes
                    )

                except Exception as exc:

                    print(
                        f"\n! SKIP "
                        f"{audio_info.get('path')}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    continue

                # Add one 4-second SONIX vector
                embeddings.append(wav)
                labels.append(BONAFIDE_LABEL)

                filename = audio_info.get(
                    "path",
                    f"{language}_{sample_count}"
                )

                files.append(filename)

                speaker_counts[speaker] = (
                    used_by_speaker + 1
                )

                duration_selected += duration
                sample_count += 1

                progress.update(1)

                # ------------------------------------------------
                # When enough wavs are collected, embed them
                # ------------------------------------------------

                if len(embeddings) >= args.batch:

                    try:

                        vecs = embed_batch(
                            feature_extractor,
                            model,
                            device,
                            embeddings
                        )

                    except RuntimeError as exc:

                        if "out of memory" in str(exc).lower():
                            print(
                                "\n! CUDA OOM. "
                                "Reduce --batch from "
                                f"{args.batch} to {max(1, args.batch // 2)}."
                            )
                            raise

                        raise

                    # Save vectors temporarily in list form.
                    # The labels/files stay row-aligned.
                    if "pending" not in locals():
                        pending = []

                    for vec, lab, filename in zip(
                        vecs,
                        labels,
                        files
                    ):
                        pending.append(
                            (
                                vec,
                                lab,
                                filename
                            )
                        )

                    embeddings.clear()
                    labels.clear()
                    files.clear()

                    # Write complete shards
                    while len(pending) >= args.shard_size:

                        chunk = pending[
                            :args.shard_size
                        ]

                        pending = pending[
                            args.shard_size:
                        ]

                        save_shard(
                            output_dir,
                            shard_idx,
                            [x[0] for x in chunk],
                            [x[1] for x in chunk],
                            [x[2] for x in chunk]
                        )

                        print(
                            f"\n[{language}] "
                            f"saved shard "
                            f"{shard_idx:05d} "
                            f"({len(chunk)} vectors)"
                        )

                        shard_idx += 1

        finally:

            progress.close()

        # --------------------------------------------------------
        # Embed leftovers
        # --------------------------------------------------------

        if embeddings:

            vecs = embed_batch(
                feature_extractor,
                model,
                device,
                embeddings
            )

            if "pending" not in locals():
                pending = []

            for vec, lab, filename in zip(
                vecs,
                labels,
                files
            ):
                pending.append(
                    (
                        vec,
                        lab,
                        filename
                    )
                )

            embeddings.clear()
            labels.clear()
            files.clear()

        # --------------------------------------------------------
        # Save final partial shard
        # --------------------------------------------------------

        if "pending" in locals() and pending:

            save_shard(
                output_dir,
                shard_idx,
                [x[0] for x in pending],
                [x[1] for x in pending],
                [x[2] for x in pending]
            )

            print(
                f"[{language}] "
                f"saved final shard "
                f"{shard_idx:05d} "
                f"({len(pending)} vectors)"
            )

            shard_idx += 1

            pending.clear()

        total_saved += sample_count

        print()
        print(
            f"[{language}] DONE"
        )
        print(
            f"Selected recordings: {sample_count}"
        )
        print(
            f"Raw audio duration: "
            f"{duration_selected / 3600:.2f} hours"
        )
        print(
            f"Speakers represented: "
            f"{len(speaker_counts)}"
        )
        print(
            f"Output: {output_dir}"
        )

    print()
    print("=" * 70)
    print("ALL INDICVOICES EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Total recordings selected: {total_saved}")
    print(f"Output root: {root}")


if __name__ == "__main__":
    main()