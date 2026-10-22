"""Checks for the two failure modes that corrupt data silently.

    python test_prep.py

Not a framework, not exhaustive. These two are here because neither shows up as
an error at runtime: a bad filename becomes a wrong group label, and an unstable
speaker split moves speakers across the holdout on a rerun.
"""
from build_manifest import parse_stem, speaker_of, assert_disjoint
from prep_common_voice import speaker_split

# --- filenames: wrong metadata must fail loudly, not default quietly
assert parse_stem("ashish_01__hi__phone__real") == ("ashish_01", "hi", "phone", "real")
for bad in ("ashish__hi__phone",              # too few fields
            "ashish_01__hin__phone__real",    # language typo
            "ashish_01__hi__phne__xttsv2",    # channel typo
            "__hi__phone__real"):             # empty recording_id
    try:
        parse_stem(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"parse_stem accepted {bad!r}")

assert speaker_of("ashish_01") == "ashish"
assert speaker_of("ashish") == "ashish"        # no take number
assert speaker_of("s01_u2_03") == "s01_u2"

# --- split: must be identical across processes and reruns, or the holdout moves
assert len({speaker_split("abc") for _ in range(100)}) == 1
assert speaker_split("abc", train_pct=100) == "train"   # pct is honoured
assert speaker_split("abc", train_pct=0) == "eval"
frac = sum(speaker_split(f"id{i}") == "train" for i in range(5000)) / 5000
assert 0.75 < frac < 0.85, f"split is {frac:.2%} train, expected ~80%"

# --- disjointness: matching is case-insensitive and tolerates the \t-suffixed file
import tempfile, pathlib
f = pathlib.Path(tempfile.mkdtemp()) / "other.txt"
f.write_text("Ashish\thi\t12\nbhuvan\tbn\t4\n", encoding="utf-8")
assert assert_disjoint(["ashish", "modi"], f) == ["ashish"]
assert assert_disjoint(["modi"], f) == []

print("ok")
