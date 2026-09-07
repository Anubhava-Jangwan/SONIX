"""The /mic page and the dashboard must show the same verdict for the same score."""

import re

from realtime.miccapture import PAGE
# Was: `from realtime.live_ui import AMBER_AT, RED_AT`. Those are Streamlit
# sliders (live_ui.py:185) defaulting to 0.10/0.90, not the production bands --
# so this test compared the mic page's 0.35 against 0.10, and importing it
# dragged Streamlit into the test run. realtime/thresholds.py is the one source.
from realtime.thresholds import AMBER_AT, RED_AT


def _js_const(name):
    m = re.search(rf"{name}\s*=\s*([0-9.]+)", PAGE)
    assert m, f"{name} not found in the mic page"
    return float(m.group(1))


def test_thresholds_match_dashboard():
    assert _js_const("AMBER_AT") == AMBER_AT
    assert _js_const("RED_AT") == RED_AT


<<<<<<< HEAD
def test_page_has_band_and_risk_chart():
    # Was `test_page_has_band_and_spectrogram`, asserting id="spec" and
    # getByteFrequencyData. The spectrogram was deliberately removed --
    # miccapture.py:171 says so: "the 'danger level' plot that replaced the
    # spectrogram". The page now draws canvas#risk, and getByteFrequencyData
    # appears nowhere in it. Pin what is actually there.
    for needed in ('id="verdict"', 'id="risk"', "function drawRisk",
                   'm.type === "scores"', "scoring_available"):
        assert needed in PAGE, f"missing: {needed}"


def test_page_has_no_stale_spectrogram_hooks():
    """If the spectrogram ever comes back it should come back with its test."""
    for gone in ('id="spec"', "getByteFrequencyData"):
        assert gone not in PAGE, f"unexpected {gone} -- update the chart tests"
=======
def test_page_has_band_and_score_chart():
    for needed in ('id="headline"', 'id="graph"', 'm.type === "scores"',
                   "scoring_available"):
        assert needed in PAGE, f"missing: {needed}"


def test_palette_comes_from_the_shared_theme():
    # No hand-copied hex list: the page carries theme.css_vars() verbatim.
    import theme
    assert theme.css_vars() in PAGE
    assert "__TOKENS__" not in PAGE
>>>>>>> 19ae017eee4118e8f66a7b904649d392682e181d


def test_verdict_is_gated_on_scoring_available():
    # A mock number must never render as a verdict.
    assert "if (!scoringAvailable)" in PAGE
    assert "Scoring unavailable" in PAGE


if __name__ == "__main__":
    test_thresholds_match_dashboard()
<<<<<<< HEAD
    test_page_has_band_and_risk_chart()
    test_page_has_no_stale_spectrogram_hooks()
=======
    test_page_has_band_and_score_chart()
    test_palette_comes_from_the_shared_theme()
>>>>>>> 19ae017eee4118e8f66a7b904649d392682e181d
    test_verdict_is_gated_on_scoring_available()
    print("ok")
