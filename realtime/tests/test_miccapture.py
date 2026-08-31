"""The /mic page and the dashboard must show the same verdict for the same score."""

import re

from realtime.miccapture import PAGE
from realtime.live_ui import AMBER_AT, RED_AT


def _js_const(name):
    m = re.search(rf"{name}\s*=\s*([0-9.]+)", PAGE)
    assert m, f"{name} not found in the mic page"
    return float(m.group(1))


def test_thresholds_match_dashboard():
    assert _js_const("AMBER_AT") == AMBER_AT
    assert _js_const("RED_AT") == RED_AT


def test_page_has_band_and_spectrogram():
    for needed in ('id="verdict"', 'id="spec"', "getByteFrequencyData",
                   'm.type === "scores"', "scoring_available"):
        assert needed in PAGE, f"missing: {needed}"


def test_verdict_is_gated_on_scoring_available():
    # A mock number must never render as a verdict.
    assert "if (!scoringAvailable)" in PAGE
    assert "Scoring unavailable" in PAGE


if __name__ == "__main__":
    test_thresholds_match_dashboard()
    test_page_has_band_and_spectrogram()
    test_verdict_is_gated_on_scoring_available()
    print("ok")
