from datasets import load_dataset

print("Loading IndicVoices Hindi...")

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "hindi",
    split="train",
    streaming=True
)

# Disable automatic TorchCodec decoding
ds = ds.decode(False)

print("Dataset connected.")
print("Streaming audio decoding disabled.")

sample = next(iter(ds))

print("\nSample keys:")
print(sample.keys())

print("\nAudio field:")
print(sample["audio_filepath"])

print("\nMetadata:")
print("Language:", sample.get("lang"))
print("Duration:", sample.get("duration"))
print("Samples:", sample.get("samples"))
print("Speaker:", sample.get("speaker_id"))

print("\nTEST PASSED.")