"""Which trained heads the live server can score with, and where they live.

The demo UI (demo/app.py) has always let you compare the three heads on the same
clip. The live server could only ever hold the ONE checkpoint passed with --ckpt,
so the live dashboard had no model choice at all. This registry is what lets both
surfaces talk about the same three models by the same names.
"""

from pathlib import Path

# key -> (label, checkpoint path relative to the repo root, one-line description)
REGISTRY = {
    "baseline": (
        "Baseline",
        "outputs/models/head.pt",
        "Trained on clean ASVspoof 2019 LA only. 1.4937% eval EER.",
    ),
    "augmented": (
        "Augmented (codec)",
        "outputs/models/head_aug.pt",
        "Clean + G.711 mu-law copies, so phone-codec audio is in-domain.",
    ),
    "robust": (
        "Robust (codec + RawBoost)",
        "outputs/models/head_robust.pt",
        "Clean + G.711 + RawBoost channel/impulsive augmentation.",
    ),
    "robust_v2": (
        "Robust v2 (+ RIR/MUSAN)",
        "outputs/models/head_robust_v2.pt",
        "Clean + G.711 + RawBoost + room reverb and MUSAN noise. The only head "
        "trained on audio whose silences contain room tone rather than digital "
        "silence -- i.e. the one aimed at our real-clip false alarms.",
    ),
    "full_ho": (
        "Full + Indic (holdout-safe)",
        "outputs/models/head_full_ho.pt",
        "Clean + G.711 + RawBoost + RIR/MUSAN + IndicVoices, with 12% of the "
        "Indic recordings held out of training. Genuine Indian speech flagged "
        "0.02% vs 57% for baseline; DF21 EER 5.27% vs 9.48%. The only head "
        "safe to quote Indic numbers from.",
    ),
    "v3": (
        "SONIX v3 (multilingual)",
        "outputs/models/head_v3.pt",
        "Clean + G.711 + RawBoost + RIR/MUSAN + IndicVoices + 35,200 Indic spoofs across "
        "three synthesis families (MMS-TTS, IndicSynth voice conversion, channel-augmented "
        "copies of both). The first head with Indian languages on BOTH sides of the label.",
    ),
}

# The head the server loads and the dashboard opens on when nothing else is
# named. head_full_ho is best on every axis measured so far -- DF21 EER,
# dev EER, and genuine-Indian-speech false alarms -- with no trade-off against
# detection. If it is missing from outputs/models/, the server falls back to
# whichever registered head IS present, so a teammate without the file still
# gets a working server.
DEFAULT_KEY = "v3"


def resolve_ckpt(ckpt_path):
    """Find a checkpoint whether the server was launched from the repo root or not.

    Tries the path as given, then walks upward from the cwd and from this file
    looking for outputs/models/<name>. Without this, launching the server from
    realtime/ made every checkpoint 'missing' for no good reason.
    """
    p = Path(ckpt_path)
    if p.exists():
        return str(p.resolve())

    name = p.name
    roots = [Path.cwd(), *Path.cwd().parents,
             Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    for root in roots:
        cand = root / "outputs" / "models" / name
        if cand.exists():
            return str(cand.resolve())
        cand = root / p
        if cand.exists():
            return str(cand.resolve())
    return str(p)


def key_for_path(ckpt_path):
    """Map a --ckpt path back onto a registry key, so `--ckpt .../head_aug.pt`
    shows up in the UI as 'Augmented' instead of an anonymous custom model."""
    if not ckpt_path:
        return None
    name = Path(ckpt_path).name
    for key, (_, path, _) in REGISTRY.items():
        if Path(path).name == name:
            return key
    return None


def catalogue():
    """Every registered model plus whether its checkpoint file actually exists."""
    out = []
    for key, (label, path, note) in REGISTRY.items():
        resolved = resolve_ckpt(path)
        out.append({
            "key": key,
            "label": label,
            "path": path,
            "resolved_path": resolved,
            "note": note,
            "exists": Path(resolved).exists(),
        })
    return out
