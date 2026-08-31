"""End-to-end check of the LIVE scoring path, plus the latency it runs at.

Answers the two questions you cannot answer by reading code:

  1. Is capture -> 4 s window -> wav2vec2 -> head -> score actually connected?
  2. Does a forward pass finish inside the 0.5 s hop, on THIS GPU? If it does
     not, windows queue and the on-screen band falls behind the call.

Works with any checkpoint. With the untrained dev head it proves plumbing and
measures latency; the scores it prints are noise and are labelled as such.

    python -m realtime.make_dev_head
    python -m realtime.selftest --ckpt outputs/models/head_dev.pt

Once the real head arrives, the same command is the acceptance test:

    python -m realtime.selftest --ckpt outputs/models/head.pt --wav testcall.wav
"""

import argparse
import time
from pathlib import Path

import numpy as np

WIN = 64000          # 4 s @ 16 kHz
HOP_SECONDS = 0.5    # the budget one batch must fit inside


def load_windows(wav_path, count):
    """Real audio if a wav was given, otherwise noise (shape is what matters)."""
    if wav_path is None:
        rng = np.random.default_rng(0)
        return [rng.standard_normal(WIN).astype(np.float32) * 0.05 for _ in range(count)]

    import soundfile as sf
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from realtime.resample import resample
        wav = resample(wav, sr)

    wins, start = [], 0
    while start + WIN <= len(wav) and len(wins) < count:
        wins.append(wav[start:start + WIN])
        start += int(16000 * HOP_SECONDS)
    if not wins:
        raise SystemExit(f"{wav_path} is shorter than one 4-second window")
    return wins


def main():
    ap = argparse.ArgumentParser(description="Live-path self-test")
    ap.add_argument("--ckpt", default="outputs/models/head_dev.pt")
    ap.add_argument("--wav", default=None, help="audio to score (default: noise)")
    ap.add_argument("--batch", type=int, default=8, help="engine max_batch_size")
    ap.add_argument("--rounds", type=int, default=3, help="timed batches after warm-up")
    args = ap.parse_args()

    from realtime.checkpoint import load_checkpoint

    print(f"checkpoint : {args.ckpt}")
    t0 = time.perf_counter()
    scorer, config = load_checkpoint(args.ckpt)
    print(f"loaded in    {time.perf_counter() - t0:.1f}s   "
          f"(front-end + head, once per process)")

    if config.get("synthetic"):
        print("\n  !! UNTRAINED DEV CHECKPOINT. The scores below are NOISE.")
        print("     They say nothing about any voice. Latency numbers are real.\n")
    else:
        print(f"dev EER    : {config.get('dev_eer')}")

    wins = load_windows(Path(args.wav) if args.wav else None, args.batch)
    source = args.wav or "synthetic noise"
    print(f"audio      : {source}  ->  {len(wins)} windows of {WIN} samples")

    scorer.score(wins[:1])                      # warm up CUDA kernels
    times, scores = [], []
    for _ in range(args.rounds):
        t = time.perf_counter()
        scores = scorer.score(wins)
        times.append(time.perf_counter() - t)

    per_batch = float(np.median(times))
    per_window = per_batch / len(wins)
    print(f"\nbatch of {len(wins)} : {per_batch * 1000:.0f} ms median "
          f"({per_window * 1000:.0f} ms/window)")

    assert len(scores) == len(wins), "scorer returned the wrong number of scores"
    assert np.all((np.asarray(scores) >= 0) & (np.asarray(scores) <= 1)), \
        "scores outside [0,1] - the head is not producing probabilities"
    print(f"scores     : {', '.join(f'{s:.3f}' for s in scores[:6])}"
          f"{' ...' if len(scores) > 6 else ''}")

    # A batch is produced every batch_interval (0.5 s). If scoring one takes
    # longer than that, the queue grows without bound and the band lags.
    print()
    if per_batch < HOP_SECONDS:
        head = HOP_SECONDS - per_batch
        print(f"PASS  one batch fits the {HOP_SECONDS}s budget with {head * 1000:.0f} ms to spare.")
        print("      The live path keeps up on this machine.")
    else:
        over = per_batch / HOP_SECONDS
        print(f"SLOW  one batch takes {over:.1f}x the {HOP_SECONDS}s budget.")
        print("      Windows will queue and the band will lag behind the call.")
        print("      Fix by raising --batch-interval on the server, or scoring")
        print("      every Nth window. Do NOT discover this on demo day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
