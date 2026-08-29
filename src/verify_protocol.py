#!/usr/bin/env python3
"""
verify_protocol.py  --  SONIX / SIH26104

The gate before any ML runs. It reads the three ASVspoof 2019 LA protocol files,
counts them, and checks them against the known-good canonical numbers. If ANYTHING
is off it prints a loud STOP and exits non-zero, so a wrong parse can never quietly
poison an EER you won't catch until the demo.

It is also a protocol loader: `load_protocol(split, data_root)` returns a DataFrame
with columns  speaker, filename, attack, label, path  (label: 'bonafide'/'spoof').
Other scripts can import that. No torch, no GPU -- pandas only.

USAGE
    python verify_protocol.py --data-root LA
    python verify_protocol.py --data-root LA --check-files     # also stat the .flac

Expected layout (standard LA.zip):
    LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
    LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
    LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
    LA/ASVspoof2019_LA_{train,dev,eval}/flac/<filename>.flac
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical, published ASVspoof 2019 LA numbers. These are the reference truth
# the parse is checked against. Source: the official protocol + the SONIX vault
# note "ASVspoof 2019 LA". Do NOT edit these to make a run pass.
# ---------------------------------------------------------------------------
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
EXPECTED = {
    "train": {"total": 25380, "bonafide": 2580, "spoof": 22800,
              "attacks": {f"A0{i}" for i in range(1, 7)}},                 # A01-A06
    "dev":   {"total": 24844, "bonafide": 2548, "spoof": 22296,
              "attacks": {f"A0{i}" for i in range(1, 7)}},                 # A01-A06
    "eval":  {"total": 71237, "bonafide": 7355, "spoof": 63882,
              "attacks": {f"A{i:02d}" for i in range(7, 20)}},            # A07-A19
}

COLUMNS = ["speaker", "filename", "unused", "attack", "label"]


def load_protocol(split: str, data_root: str) -> pd.DataFrame:
    """Read one protocol file into a DataFrame.

    Columns: speaker, filename, attack, label, path
      - label is 'bonafide' or 'spoof'
      - attack is 'A01'..'A19' for spoof, '-' for bonafide
      - path points at the .flac for this file (may or may not exist on disk)
    """
    root = Path(data_root)
    proto = root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILE[split]
    if not proto.exists():
        raise FileNotFoundError(f"Protocol file not found: {proto}")

    # whitespace-separated, no header, five columns
    df = pd.read_csv(proto, sep=r"\s+", header=None, names=COLUMNS,
                     dtype=str, engine="python")
    df = df.drop(columns=["unused"])

    flac_root = root / FLAC_DIR[split] / "flac"
    df["path"] = df["filename"].map(lambda fn: str(flac_root / f"{fn}.flac"))
    return df


def _check_split(split: str, df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list == all good)."""
    exp = EXPECTED[split]
    problems: list[str] = []

    total = len(df)
    n_bona = int((df["label"] == "bonafide").sum())
    n_spoof = int((df["label"] == "spoof").sum())
    attacks = set(df.loc[df["label"] == "spoof", "attack"].unique())

    print(f"  [{split}] lines={total}  bonafide={n_bona}  spoof={n_spoof}")
    print(f"  [{split}] spoof attacks present: {sorted(attacks)}")

    # unexpected label strings (typo / wrong column / wrong delimiter)
    bad_labels = set(df["label"].unique()) - {"bonafide", "spoof"}
    if bad_labels:
        problems.append(
            f"[{split}] unexpected label values {sorted(bad_labels)} -- "
            f"delimiter or column order is wrong")

    if total != exp["total"]:
        problems.append(f"[{split}] line count {total} != expected {exp['total']}")
    if n_bona != exp["bonafide"]:
        problems.append(
            f"[{split}] bonafide {n_bona} != expected {exp['bonafide']}")
    if n_spoof != exp["spoof"]:
        problems.append(
            f"[{split}] spoof {n_spoof} != expected {exp['spoof']}")

    # The classic silent killer: labels swapped. Call it out explicitly.
    if n_bona == exp["spoof"] and n_spoof == exp["bonafide"]:
        problems.append(
            f"[{split}] bonafide/spoof counts are SWAPPED -- your labels are "
            f"inverted. Any EER from this is meaningless.")

    if attacks != exp["attacks"]:
        missing = exp["attacks"] - attacks
        extra = attacks - exp["attacks"]
        msg = f"[{split}] attack set mismatch."
        if missing:
            msg += f" missing={sorted(missing)}"
        if extra:
            msg += f" unexpected={sorted(extra)}"
        problems.append(msg)

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify ASVspoof 2019 LA protocols.")
    ap.add_argument("--data-root", default="data/asvspoof19_la",
                    help="Folder containing ASVspoof2019_LA_* (default: data/asvspoof19_la)")
    ap.add_argument("--check-files", action="store_true",
                    help="Also confirm every .flac referenced actually exists "
                         "(slower; stats ~120k files)")
    args = ap.parse_args()

    print("=" * 64)
    print("SONIX protocol gate  --  ASVspoof 2019 LA")
    print("data-root:", Path(args.data_root).resolve())
    print("=" * 64)

    all_problems: list[str] = []
    for split in ("train", "dev", "eval"):
        try:
            df = load_protocol(split, args.data_root)
        except FileNotFoundError as e:
            all_problems.append(str(e))
            continue

        all_problems.extend(_check_split(split, df))

        if args.check_files:
            missing = [p for p in df["path"] if not Path(p).exists()]
            if missing:
                all_problems.append(
                    f"[{split}] {len(missing)} referenced .flac files are "
                    f"missing on disk (first: {missing[0]})")
            else:
                print(f"  [{split}] all {len(df)} .flac files present")
        print()

    print("=" * 64)
    if all_problems:
        print("STOP.  Do NOT write or run another line of ML code.")
        print("The protocol parse does not match the known-good numbers:")
        for p in all_problems:
            print("   -  " + p)
        print("=" * 64)
        return 1

    print("PASS.  All three splits match the canonical ASVspoof 2019 LA numbers.")
    print("   train 25380 (2580 bona / 22800 spoof, A01-A06)")
    print("   dev   24844 (2548 bona / 22296 spoof, A01-A06)")
    print("   eval  71237 (7355 bona / 63882 spoof, A07-A19)")
    print("Safe to proceed to extraction.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
