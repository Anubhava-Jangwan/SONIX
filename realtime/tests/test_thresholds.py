"""Every surface that renders a risk band must use the same two numbers.

The bands were previously written out independently in the mic page, the
website and the Chrome extension, each with a comment asking the others to stay
in sync. This is that comment, enforced.
"""

import re
from pathlib import Path

import pytest

from realtime.thresholds import AMBER_AT, RED_AT, band_for

ROOT = Path(__file__).resolve().parents[2]

# file -> whether it must exist for the suite to be meaningful
SURFACES = {
    ROOT / "realtime" / "miccapture.py": True,
    ROOT / "website" / "script.js": True,
    ROOT / "extension" / "offscreen.js": False,
    ROOT / "extension" / "content.js": False,
}


def _js_consts(text):
    """Every `AMBER_AT = <num>` / `RED_AT = <num>` literal in a file."""
    return (
        [float(m) for m in re.findall(r"AMBER_AT\s*=\s*([0-9.]+)", text)],
        [float(m) for m in re.findall(r"RED_AT\s*=\s*([0-9.]+)", text)],
    )


@pytest.mark.parametrize("path,required", list(SURFACES.items()),
                         ids=lambda p: getattr(p, "name", str(p)))
def test_surface_thresholds_match_canonical(path, required):
    if not path.exists():
        if required:
            pytest.fail(f"{path} is missing")
        pytest.skip(f"{path.name} not present")

    ambers, reds = _js_consts(path.read_text(encoding="utf-8", errors="replace"))
    if not ambers and not reds:
        pytest.skip(f"{path.name} declares no thresholds")

    for got in ambers:
        assert got == AMBER_AT, f"{path.name}: AMBER_AT {got} != {AMBER_AT}"
    for got in reds:
        assert got == RED_AT, f"{path.name}: RED_AT {got} != {RED_AT}"


def test_bands_are_ordered():
    assert 0.0 < AMBER_AT < RED_AT < 1.0


@pytest.mark.parametrize("score,expected", [
    (0.0, "Green"), (0.34, "Green"),
    (AMBER_AT, "Amber"), (0.5, "Amber"), (0.649, "Amber"),
    (RED_AT, "Red"), (1.0, "Red"),
])
def test_band_boundaries_are_inclusive_upward(score, expected):
    assert band_for(score) == expected
