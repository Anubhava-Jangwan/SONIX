#!/usr/bin/env python3
"""
padding_plot.py -- SONIX / SIH26104

The padding-bug slide.

THE BUG. The model always listens in fixed 4-second chunks. When a clip was
shorter than 4 seconds we filled the remainder with DIGITAL SILENCE. A
1-second clip therefore became a chunk that was three-quarters nothing, and
the model -- which has never been trained on silence -- called real people
fake.

THE FIX. We now REPEAT the audio to fill the gap instead of padding with
zeros, so every chunk is 4 seconds of actual speech.

THE MEASUREMENT. Same genuine human recording (chetan voice.ogg, 66.1 s),
same baseline model (head.pt), identical excerpt positions in both runs.
Only the padding strategy changed.

    python src/padding_plot.py
    -> outputs/plots/padding_fix.png

These numbers are MEASURED. Do not edit them without re-running the
measurement that produced them.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# MEASURED DATA -- baseline head.pt on "chetan voice.ogg" (66.1 s, genuine
# human). Scores run 0-1; HIGHER means the model thinks it is FAKE.
# ----------------------------------------------------------------------
CLIP_LABELS = ["1 second", "2 seconds", "3 seconds", "full 66 s"]
BEFORE = [0.8795, 0.5061, 0.2224, 0.1171]   # zero-padded  (the bug)
AFTER = [0.0329, 0.0005, 0.0096, 0.1171]    # repeat-padded (the fix)

AMBER_THRESHOLD = 0.10   # our current amber alarm level

# Supporting detail for the full-length run, for the speaker notes:
FULL_MEDIAN_WINDOW = 0.0064
FULL_WINDOWS = 125
FULL_PCT_ABOVE_090 = 2   # percent of the 125 windows scoring above 0.90


def make_plot(output_path):
    x = list(range(len(CLIP_LABELS)))

    fig, ax = plt.subplots(figsize=(9, 6))

    # Shade the region where the system raises an alarm.
    ax.axhspan(AMBER_THRESHOLD, 1.02, color="#d62728", alpha=0.06, zorder=0)
    ax.axhspan(-0.02, AMBER_THRESHOLD, color="#2ca02c", alpha=0.06, zorder=0)

    # The two measurement runs.
    ax.plot(x, BEFORE, "o-", linewidth=2.5, markersize=10,
            color="#d62728", label="BEFORE - gap filled with silence (the bug)",
            zorder=3)
    ax.plot(x, AFTER, "s-", linewidth=2.5, markersize=9,
            color="#2ca02c", label="AFTER - gap filled by repeating the audio",
            zorder=3)

    # The alarm level.
    ax.axhline(AMBER_THRESHOLD, linestyle="--", linewidth=1.8,
               color="#ff7f0e", zorder=2)
    ax.text(-0.33, AMBER_THRESHOLD + 0.025,
            f"amber alarm level ({AMBER_THRESHOLD:.2f})",
            color="#ff7f0e", fontsize=10, fontweight="bold", ha="left")

    # Put the actual number next to every point -- a judge should not have to
    # read values off an axis. The last point is shared by both lines, so it
    # gets one label instead of two overlapping ones.
    last = len(x) - 1
    for xi, yi in zip(x[:last], BEFORE[:last]):
        ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=11,
                    fontweight="bold", color="#d62728")
    for xi, yi in zip(x[:last], AFTER[:last]):
        ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 16), ha="center", fontsize=11,
                    fontweight="bold", color="#2ca02c")

    ax.annotate(f"{BEFORE[last]:.4f}\nboth runs", (x[last], BEFORE[last]),
                textcoords="offset points", xytext=(14, -6), ha="left",
                fontsize=10.5, fontweight="bold", color="#444444")

    # The full-length point is the control: it contains no padding at all, so
    # it must be identical in both runs. Saying so out loud is what makes the
    # rest of the chart believable.
    ax.annotate(
        "no padding needed at full length -\nidentical in both runs (control)",
        xy=(x[last] - 0.04, BEFORE[last] + 0.02),
        xytext=(x[last] - 0.55, 0.62),
        fontsize=9.5, color="#444444", ha="center",
        arrowprops=dict(arrowstyle="->", color="#888888", linewidth=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLIP_LABELS, fontsize=11)
    ax.set_xlim(-0.4, len(x) - 0.45)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("Length of the clip given to the model", fontsize=12)
    ax.set_ylabel("Model's fake score  (higher = thinks it's fake)", fontsize=12)
    ax.set_title(
        "Short clips of a REAL human voice, before and after the padding fix\n"
        "same recording, same model - only the padding changed",
        fontsize=13, fontweight="bold", pad=14,
    )

    ax.legend(loc="upper right", fontsize=10.5, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved padding-fix plot: {output_path}")


def print_speaker_notes():
    print()
    print("=" * 66)
    print("SPEAKER NOTES  (say these, do not put them on the slide)")
    print("=" * 66)
    print("Source clip : 'chetan voice.ogg', 66.1 s, genuine human speech")
    print("Model       : baseline head.pt (unchanged between the two runs)")
    print("Controlled  : identical excerpt positions in both runs")
    print()
    print("A 1-second clip of a real person went from 0.8795 -- confidently")
    print("FAKE, well above our 0.10 alarm level -- to 0.0329, confidently")
    print("REAL. We changed no model weights. We fixed how we fill the gap.")
    print()
    print(f"Full-length run: median window score {FULL_MEDIAN_WINDOW:.4f}, and")
    print(f"only {FULL_PCT_ABOVE_090}% of its {FULL_WINDOWS} windows scored above 0.90.")
    print()
    print("Why it matters: this is not a better number, it is a found and")
    print("fixed defect, measured before and after. That is what makes the")
    print("rest of our numbers credible.")
    print("=" * 66)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the SONIX padding-fix slide.")
    ap.add_argument("--out", default="outputs/plots/padding_fix.png")
    a = ap.parse_args()
    make_plot(a.out)
    print_speaker_notes()