#!/usr/bin/env python3
"""
calibration.py -- SONIX / SIH26104

Temperature-scaling calibration.
Fits one temperature value T using DEV logits and labels,
then applies the same T to another split such as EVAL.

Example:
    python src/eval.py --split dev  --model-ckpt outputs/models/head.pt
    python src/eval.py --split eval --model-ckpt outputs/models/head.pt
    python src/calibration.py --fit-split dev --apply-split eval

Expected input files (eval.py now writes all of these):
    outputs/scores/dev_logits.npy
    outputs/scores/dev_labels.npy
    outputs/scores/eval_logits.npy

Output:
    outputs/scores/eval_calibrated_scores.npy
    outputs/models/temperature.npy

--------------------------------------------------------------------------
TWO CHANGES vs Yukti's original draft (method unchanged, still a BCE grid
search over T):
  1. Search range widened from 0.05-10.0 to 0.05-100.0. Our logits are large
     (sigmoid saturates to exactly 1.0 on 26k of 71k eval samples), so the
     optimal T can easily exceed 10 and the old grid would silently return
     the boundary value 10.0.
  2. A warning is printed if the best T lands on the edge of the grid, so a
     clipped result can never quietly end up on a slide.
--------------------------------------------------------------------------
"""

import argparse
from pathlib import Path

import numpy as np

T_MIN, T_MAX, T_STEPS = 0.05, 100.0, 4000


def sigmoid(x):
    """Convert logits into probabilities."""
    x = np.asarray(x, dtype=np.float64)
    # Prevent overflow for very large/small values.
    x = np.clip(x, -100, 100)
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(labels, probabilities):
    """
    Calculate Binary Cross Entropy.
    Lower loss means the predicted probabilities match
    the true labels better.
    """
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    # Avoid log(0).
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    loss = -np.mean(
        labels * np.log(probabilities)
        + (1.0 - labels) * np.log(1.0 - probabilities)
    )
    return float(loss)


def find_best_temperature(logits, labels):
    """
    Find the temperature T that gives the lowest
    Binary Cross Entropy on the DEV set.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()

    if len(logits) != len(labels):
        raise ValueError(
            f"Length mismatch: "
            f"{len(logits)} logits but {len(labels)} labels"
        )
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Calibration requires both bonafide (0) "
            "and spoof (1) samples."
        )

    temperatures = np.linspace(T_MIN, T_MAX, T_STEPS)
    best_temperature = None
    best_loss = float("inf")

    for temperature in temperatures:
        # Temperature scaling: calibrated probability = sigmoid(logit / T)
        calibrated_scores = sigmoid(logits / temperature)
        loss = binary_cross_entropy(labels, calibrated_scores)
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature

    return float(best_temperature), float(best_loss)


def calibrate_scores(logits, temperature):
    """
    Apply temperature scaling to logits.
        calibrated_score = sigmoid(logit / T)
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    return sigmoid(logits / temperature)


def main():
    parser = argparse.ArgumentParser(
        description="SONIX temperature-scaling calibration."
    )
    parser.add_argument("--fit-split", default="dev",
                        help="Split used to find the temperature (default: dev)")
    parser.add_argument("--apply-split", default="eval",
                        help="Split to calibrate using the learned temperature")
    parser.add_argument("--scores-dir", default="outputs/scores",
                        help="Directory containing logits and labels")
    parser.add_argument("--out-dir", default="outputs/scores",
                        help="Directory for calibrated scores")
    parser.add_argument("--model-dir", default="outputs/models",
                        help="Directory for saving the temperature value")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)

    # STEP 1: Load fit-split logits and labels
    dev_logits_path = scores_dir / f"{args.fit_split}_logits.npy"
    dev_labels_path = scores_dir / f"{args.fit_split}_labels.npy"

    if not dev_logits_path.exists():
        raise FileNotFoundError(
            f"Logits not found: {dev_logits_path}\n"
            f"Run:  python src/eval.py --split {args.fit_split} "
            f"--model-ckpt outputs/models/head.pt"
        )
    if not dev_labels_path.exists():
        raise FileNotFoundError(
            f"Labels not found: {dev_labels_path}\n"
            f"Run:  python src/eval.py --split {args.fit_split} "
            f"--model-ckpt outputs/models/head.pt"
        )

    dev_logits = np.load(dev_logits_path)
    dev_labels = np.load(dev_labels_path)

    # STEP 2: Find the best temperature T
    print("=" * 60)
    print("SONIX TEMPERATURE CALIBRATION")
    print("=" * 60)
    print(f"Fitting temperature on: {args.fit_split}  ({len(dev_labels)} samples)")

    temperature, loss = find_best_temperature(dev_logits, dev_labels)

    print(f"Best temperature: {temperature:.6f}")
    print(f"DEV calibration loss: {loss:.6f}")

    edge = (T_MAX - T_MIN) / (T_STEPS - 1) * 2
    if temperature >= T_MAX - edge or temperature <= T_MIN + edge:
        print()
        print("!" * 60)
        print("WARNING: the best temperature landed on the EDGE of the search")
        print(f"range ({T_MIN} to {T_MAX}). The true optimum is probably outside")
        print("it, so this value is clipped and NOT trustworthy. Widen T_MIN /")
        print("T_MAX at the top of this file and re-run before reporting it.")
        print("!" * 60)
        print()

    model_dir.mkdir(parents=True, exist_ok=True)
    temperature_path = model_dir / "temperature.npy"
    np.save(temperature_path, np.array([temperature], dtype=np.float64))
    print(f"Saved temperature: {temperature_path}")

    # STEP 3: Load logits to calibrate
    apply_logits_path = scores_dir / f"{args.apply_split}_logits.npy"
    if not apply_logits_path.exists():
        raise FileNotFoundError(
            f"Logits not found: {apply_logits_path}\n"
            f"Run:  python src/eval.py --split {args.apply_split} "
            f"--model-ckpt outputs/models/head.pt"
        )
    apply_logits = np.load(apply_logits_path)

    # STEP 4: Apply the SAME temperature
    calibrated_scores = calibrate_scores(apply_logits, temperature)

    # STEP 5: Save calibrated scores
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{args.apply_split}_calibrated_scores.npy"
    np.save(output_path, calibrated_scores.astype(np.float32))

    raw = sigmoid(apply_logits)
    print(f"Applying temperature to: {args.apply_split}")
    print(f"  saturated at exactly 0 or 1  before: "
          f"{int(((raw == 0) | (raw == 1)).sum())}   after: "
          f"{int(((calibrated_scores == 0) | (calibrated_scores == 1)).sum())}")
    print(f"Saved calibrated scores: {output_path}")
    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print("NOTE: calibration does NOT change EER (it is a monotonic transform).")
    print("      It makes the 0-1 number mean a real probability, so thresholds")
    print("      transfer sensibly across datasets.")


if __name__ == "__main__":
    main()
