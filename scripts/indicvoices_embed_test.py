import io
import os

import numpy as np
import soundfile as sf
import torch

from datasets import load_dataset
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
LANGUAGE = "hindi"
SPLIT = "train"

# Tiny smoke test only.
NUM_SAMPLES = 5

OUTPUT_DIR = os.path.join(
    "outputs",
    "embeddings_indicvoices_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 60)
print("SONIX - INDICVOICES EMBEDDING TEST")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Language: {LANGUAGE}")
print(f"Samples: {NUM_SAMPLES}")


# ---------------------------------------------------------
# 1. Load IndicVoices
# ---------------------------------------------------------

print("\n[1/5] Loading IndicVoices...")

ds = load_dataset(
    "ai4bharat/IndicVoices",
    LANGUAGE,
    split=SPLIT,
    streaming=True
)

# Prevent automatic audio decoding through TorchCodec.
ds = ds.decode(False)

print("Dataset connected.")


# ---------------------------------------------------------
# 2. Load XLS-R feature extractor + model
# ---------------------------------------------------------

print("\n[2/5] Loading XLS-R feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Feature extractor loaded.")

print("\nLoading XLS-R 300M model...")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()

print("XLS-R model loaded.")


# ---------------------------------------------------------
# 3. Stream audio
# ---------------------------------------------------------

print("\n[3/5] Streaming audio samples...")

iterator = iter(ds)

embeddings = []
metadata = []


for i in range(NUM_SAMPLES):

    sample = next(iterator)

    audio_info = sample["audio_filepath"]

    audio_bytes = audio_info["bytes"]

    # Decode FLAC directly from memory.
    waveform, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    # Stereo/multichannel -> mono.
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # XLS-R expects 16 kHz audio.
    if sample_rate != 16000:

        waveform = resample_poly(
            waveform,
            16000,
            sample_rate
        ).astype(np.float32)

        sample_rate = 16000

    duration = len(waveform) / sample_rate

    print(
        f"\nSample {i + 1}/{NUM_SAMPLES}"
    )

    print(
        f"  File: {audio_info.get('path')}"
    )

    print(
        f"  Speaker: {sample.get('speaker_id')}"
    )

    print(
        f"  Language: {sample.get('lang')}"
    )

    print(
        f"  Duration: {duration:.2f}s"
    )


    # -----------------------------------------------------
    # 4. Extract XLS-R embedding
    # -----------------------------------------------------

    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_values = inputs["input_values"].to(DEVICE)

    attention_mask = inputs.get("attention_mask")

    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask
        )

    # Mean-pool the temporal dimension.
    embedding = outputs.last_hidden_state.mean(
        dim=1
    ).squeeze(0).cpu().numpy()

    print(
        f"  Embedding shape: {embedding.shape}"
    )

    embeddings.append(embedding)

    metadata.append({
        "path": audio_info.get("path"),
        "speaker_id": sample.get("speaker_id"),
        "language": sample.get("lang"),
        "duration": sample.get("duration"),
        "label": 0
    })


# ---------------------------------------------------------
# 5. Save test embeddings
# ---------------------------------------------------------

print("\n[4/5] Saving embeddings...")

embeddings = np.stack(embeddings)

np.save(
    os.path.join(OUTPUT_DIR, "embeddings.npy"),
    embeddings
)

np.save(
    os.path.join(OUTPUT_DIR, "labels.npy"),
    np.zeros(
        len(embeddings),
        dtype=np.int64
    )
)

np.save(
    os.path.join(OUTPUT_DIR, "files.npy"),
    np.array(
        [m["path"] for m in metadata],
        dtype=object
    )
)

print("\n[5/5] RESULT")
print("-" * 60)

print(
    f"Embedding matrix: {embeddings.shape}"
)

print(
    f"Labels: {len(embeddings)}"
)

print(
    f"Label values: {np.unique(np.zeros(len(embeddings), dtype=np.int64))}"
)

print(
    f"Output: {OUTPUT_DIR}"
)

print("\nIndicVoices embedding smoke test PASSED.")