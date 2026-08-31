#!/usr/bin/env python3
"""
metrics.py -- SONIX / SIH26104

Loads labels and scores produced by eval.py and calculates:
1. Equal Error Rate (EER)
2. EER threshold
3. DET-style curve
4. Score distribution histogram

Label convention:
    0 = bonafide / real
    1 = spoof / fake

Score convention:
    Higher score = more likely spoof

Example:
    python src/metrics.py --split eval
    python src/metrics.py --split itw
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def calculate_eer(labels, scores):
    """
    Calculate Equal Error Rate (EER).

    Returns:
        eer_value: EER as a decimal
                   Example: 0.05 means 5%
        threshold: Score threshold at the EER point
    """
    # Make sure the inputs are NumPy arrays.
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    # EER needs both real and spoof samples.
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    # Calculate False Positive Rate, True Positive Rate
    # and the corresponding thresholds.
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # False Negative Rate.
    fnr = 1.0 - tpr

    # Find the point where FPR and FNR are closest.
    index = np.nanargmin(np.abs(fpr - fnr))

    # At the EER point, use the average of FPR and FNR.
    eer_value = (fpr[index] + fnr[index]) / 2.0
    eer_threshold = thresholds[index]

    return float(eer_value), float(eer_threshold)


def create_det_curve(labels, scores, output_path):
    """
    Create a DET-style graph.
    X-axis: False Positive Rate
    Y-axis: False Negative Rate
    """
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    # Get ROC information.
    fpr, tpr, _ = roc_curve(labels, scores)

    # Convert True Positive Rate to False Negative Rate.
    fnr = 1.0 - tpr

    # Create the graph.
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, fnr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("SONIX DET Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved DET curve: {output_path}")


def create_score_histogram(labels, scores, output_path):
    """
    Create a histogram comparing real and spoof scores.
    """
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    # Separate the scores according to their true labels.
    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]

    plt.figure(figsize=(8, 6))

    # Scores for real audio.
    plt.hist(
        bonafide_scores,
        bins=50,
        alpha=0.6,
        density=True,
        label="Bonafide (Real)"
    )

    # Scores for spoof audio.
    plt.hist(
        spoof_scores,
        bins=50,
        alpha=0.6,
        density=True,
        label="Spoof (Fake)"
    )

    plt.xlabel("Spoof Score")
    plt.ylabel("Density")
    plt.title("SONIX Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved score histogram: {output_path}")


def main():
    """
    Load evaluation results and generate all metrics.
    """
    # Create command-line arguments.
    parser = argparse.ArgumentParser(
        description="Calculate SONIX EER and generate evaluation plots."
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Dataset split to evaluate, for example: eval or itw"
    )
    parser.add_argument(
        "--scores-dir",
        default="outputs/scores",
        help="Directory containing labels and scores"
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/plots",
        help="Directory where graphs will be saved"
    )
    args = parser.parse_args()

    # Convert folder paths into Path objects.
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)

    # These names match the files created by eval.py.
    labels_path = scores_dir / f"{args.split}_labels.npy"
    scores_path = scores_dir / f"{args.split}_scores.npy"

    # Check that the files exist.
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}"
        )
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Scores file not found: {scores_path}"
        )

    # Load the correct labels and model scores.
    labels = np.load(labels_path)
    scores = np.load(scores_path)

    # Safety check: every sample must have one label and one score.
    if len(labels) != len(scores):
        raise ValueError(
            f"Length mismatch: "
            f"{len(labels)} labels but {len(scores)} scores"
        )

    # Check that both classes are present.
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Both bonafide (0) and spoof (1) samples are required."
        )

    # Create the output directory if it does not already exist.
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Calculate EER
    # -----------------------------
    eer_value, threshold = calculate_eer(labels, scores)

    # Print results.
    print("=" * 60)
    print("SONIX METRICS")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Total samples: {len(labels)}")
    print(f"Bonafide / Real (0): {(labels == 0).sum()}")
    print(f"Spoof / Fake (1): {(labels == 1).sum()}")
    print("-" * 60)
    print(f"EER: {eer_value * 100:.4f}%")
    print(f"EER Threshold: {threshold:.6f}")
    print("=" * 60)

    # -----------------------------
    # Create graphs
    # -----------------------------
    det_path = out_dir / f"{args.split}_det.png"
    histogram_path = out_dir / f"{args.split}_scores.png"

    create_det_curve(labels, scores, det_path)
    create_score_histogram(labels, scores, histogram_path)

    print("\nMetrics calculation complete.")


if __name__ == "__main__":
    main()
