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


def create_det_curve(labels, scores, output_path, title="SONIX DET Curve"):
    """
    Create a DET-style graph on LOG-LOG axes.

    Why log-log: our EER is ~1.5%, so both error rates live below 0.05. On
    linear axes the whole curve collapses into the bottom-left corner and the
    plot reads as an empty box. Real DET curves use normal-deviate (probit)
    axes; log-log is the simple version and is perfectly readable on a slide.

    X-axis: False Positive Rate  (bonafide wrongly called spoof)
    Y-axis: False Negative Rate  (spoof wrongly called bonafide)
    """
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    # Get ROC information.
    fpr, tpr, _ = roc_curve(labels, scores)

    # Convert True Positive Rate to False Negative Rate.
    fnr = 1.0 - tpr

    # The EER point: where FPR and FNR are closest.
    index = np.nanargmin(np.abs(fpr - fnr))
    eer_value = (fpr[index] + fnr[index]) / 2.0

    # Log axes cannot show zero. Pick a lower limit from the data itself so we
    # never silently crop part of the curve: the smallest error rate we can
    # even represent is 1/n for each class.
    positives = np.concatenate([fpr[fpr > 0], fnr[fnr > 0]])
    if positives.size:
        # one decade of headroom below the smallest observed non-zero rate
        lower = 10 ** np.floor(np.log10(positives.min()))
    else:
        lower = 1e-3
    lower = min(lower, 1e-3)      # never zoom in tighter than 1e-3
    lower = max(lower, 1e-6)      # but keep the plot sane

    # Create the graph.
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, fnr, linewidth=2, label="DET curve")

    # Mark the EER point so the headline number is visible on the plot itself.
    plt.plot(fpr[index], fnr[index], "o", ms=9, color="crimson", zorder=5)
    plt.annotate(
        f"EER = {eer_value * 100:.2f}%",
        (fpr[index], fnr[index]),
        textcoords="offset points",
        xytext=(12, 12),
        fontsize=11,
        fontweight="bold",
        color="crimson",
    )

    # The EER line: FPR == FNR. The curve crosses it exactly at the EER point.
    diag = np.array([lower, 1.0])
    plt.plot(diag, diag, "--", linewidth=1, color="grey",
             label="FPR = FNR (EER line)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(lower, 1.0)
    plt.ylim(lower, 1.0)

    plt.xlabel("False Positive Rate  (real voice flagged as fake)")
    plt.ylabel("False Negative Rate  (fake voice passed as real)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="upper right")
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
    parser.add_argument(
        "--scores-file",
        default=None,
        help="score .npy to use instead of <split>_scores.npy "
             "(e.g. outputs/scores/eval_calibrated_scores.npy)"
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="suffix for the output plot filenames. Defaults to the score "
             "file's stem when --scores-file is used, so a calibrated run "
             "never overwrites the uncalibrated plots."
    )
    args = parser.parse_args()

    # Convert folder paths into Path objects.
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)

    # These names match the files created by eval.py. --scores-file lets us
    # point at any other score array (calibrated scores, a different model)
    # while still using the split's labels.
    labels_path = scores_dir / f"{args.split}_labels.npy"
    if args.scores_file:
        scores_path = Path(args.scores_file)
    else:
        scores_path = scores_dir / f"{args.split}_scores.npy"

    # Name the plots after whatever we actually scored, so re-running with
    # calibrated scores does not silently overwrite the figure already on a
    # slide.
    if args.tag:
        stem = f"{args.split}_{args.tag}"
    elif args.scores_file:
        stem = scores_path.stem.replace("_scores", "")
    else:
        stem = args.split

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
    print(f"Scores file: {scores_path}")
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
    det_path = out_dir / f"{stem}_det.png"
    histogram_path = out_dir / f"{stem}_scores.png"

    create_det_curve(
        labels, scores, det_path,
        title=f"SONIX DET Curve - {stem} (EER {eer_value * 100:.2f}%)"
    )
    create_score_histogram(labels, scores, histogram_path)

    print("\nMetrics calculation complete.")


if __name__ == "__main__":
    main()