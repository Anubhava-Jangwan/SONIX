"""The risk-band thresholds. One definition, imported everywhere.

    Green  score <  AMBER_AT
    Amber  AMBER_AT <= score < RED_AT
    Red    score >= RED_AT

Provisional and uncalibrated -- see CODE_MAP.md. Changing them changes what the
product tells a user about a live call, so treat this as ML semantics, not
configuration, and get Yugal/Suryansh sign-off.

Why this file exists: the same two numbers were written out separately in
realtime/miccapture.py, website/script.js and the Chrome extension, with a
comment in each saying "must match the others". realtime/live_ui.py is NOT one
of these -- its AMBER_AT/RED_AT are Streamlit sliders an operator moves at
runtime (defaulting to 0.10/0.90), which is a different thing that happens to
share a name. test_thresholds.py pins the copies that must agree.
"""

AMBER_AT = 0.35
RED_AT = 0.65

assert 0.0 < AMBER_AT < RED_AT < 1.0, "bands must be ordered and inside [0,1]"


def band_for(score: float) -> str:
    """"Green" / "Amber" / "Red" for a P(synthetic) in [0,1]."""
    if score >= RED_AT:
        return "Red"
    if score >= AMBER_AT:
        return "Amber"
    return "Green"
