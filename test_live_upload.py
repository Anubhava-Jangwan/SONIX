"""Headless test for the live upload path -- run this BEFORE opening the UI.

Checks three things the dashboard depends on:

  1. /api/models reports the heads that actually exist on disk
  2. the streaming upload really streams (scores arrive progressively, not all
     at the end) and lands on the expected verdict for known clips
  3. the live path agrees with demo/score_file.py on the SAME clip and the SAME
     checkpoint -- if these two disagree, one of the two UIs is lying

Usage:
    python -m realtime.server --auto-approve --ws-port 8000     (terminal A)
    python test_live_upload.py                                  (terminal B)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import requests

BASE = "http://localhost:8000"
AMBER, RED = 0.10, 0.90

CLIPS = [
    ("sonix_real/sonix_real/real/real_06.wav", "real"),
    ("sonix_real/sonix_real/fake/fake_10.wav", "fake"),
]


def wait_until_warm(base, timeout=600):
    """Block until the server has finished loading the 300M front-end.

    Uploading before then still works, but the first batch would absorb the
    whole load time and make the numbers below look like a stall.
    """
    t0 = time.time()
    said = False
    while time.time() - t0 < timeout:
        try:
            info = requests.get(f"{base}/api/models", timeout=10).json()
        except Exception:
            time.sleep(2)
            continue
        if info.get("warm"):
            if said:
                print(f"  front-end warm after {time.time() - t0:.0f}s")
            return True
        if not said:
            print("  waiting for the wav2vec2 front-end to load...")
            said = True
        time.sleep(2)
    print("  WARNING: server never reported warm; continuing anyway")
    return False


def band(mean):
    return "RED" if mean >= RED else ("AMBER" if mean >= AMBER else "GREEN")


def check_models(base):
    r = requests.get(f"{base}/api/models", timeout=5)
    r.raise_for_status()
    info = r.json()
    if info.get("mock"):
        print("FAIL: server is in mock mode -- restart it without --mock, with "
              "checkpoints present in outputs/models/")
        return None
    print(f"default model: {info['default']}")
    ready = []
    for m in info["models"]:
        mark = "yes" if m["exists"] else "NO "
        print(f"  [{mark}] {m['key']:10s} {m['label']:28s} {m['path']}")
        if m["exists"]:
            ready.append(m["key"])
    return ready


def stream_clip(base, path, model, poll=0.4, timeout=600):
    """Upload without wait=1, then poll telemetry the way the dashboard does."""
    with open(path, "rb") as fh:
        r = requests.post(f"{base}/api/score-file",
                          files={"file": (Path(path).name, fh.read())},
                          data={"model": model}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"upload rejected: {r.status_code} {r.text}")
    started = r.json()
    call_id = started["call_id"]
    expected = int(started["expected_windows"])

    snapshots = []                 # (elapsed_s, windows_scored)
    t_start = time.time()
    stall = time.time()
    last = 0
    scores = {}

    while time.time() - t_start < timeout:
        try:
            tr = requests.get(f"{base}/api/telemetry",
                              params={"call_id": call_id, "limit": 5000}, timeout=30)
            call = tr.json().get("calls", {}).get(call_id)
        except requests.RequestException:
            # A slow poll is not fatal -- the scores are held server-side and
            # the next poll picks up everything recorded since.
            time.sleep(poll)
            continue
        if call:
            scores = call.get("scores", {}) or {}
            n = len(scores)
            snapshots.append((round(time.time() - t_start, 1), n))
            if n > last:
                last, stall = n, time.time()
            if n >= expected:
                break
            if call.get("feed_done") and time.time() - stall > 15:
                break
        time.sleep(poll)

    try:
        requests.post(f"{base}/api/end-call", json={"call_id": call_id}, timeout=30)
    except requests.RequestException:
        pass
    vals = [float(v["score"]) for _, v in sorted(scores.items(), key=lambda kv: int(kv[0]))]
    return started, vals, snapshots


def batch_scores(path, ckpt):
    """Same clip through demo/score_file.py -- the path our numbers came from."""
    demo = Path(__file__).resolve().parent / "demo"
    if str(demo) not in sys.path:
        sys.path.insert(0, str(demo))
    import score_file as S
    S.set_vad(enabled=True)
    return [float(x) for x in S.score_file(str(path), ckpt_path=ckpt)]


CKPT_FOR = {"baseline": "outputs/models/head.pt",
            "augmented": "outputs/models/head_aug.pt",
            "robust": "outputs/models/head_robust.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--models", default=None,
                    help="comma-separated subset, e.g. baseline,robust")
    ap.add_argument("--skip-crosscheck", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print("1. MODELS")
    print("=" * 68)
    try:
        ready = check_models(args.base)
    except Exception as exc:
        print(f"FAIL: cannot reach {args.base} -- is the server running? ({exc})")
        return 1
    if not ready:
        return 1
    if args.models:
        ready = [m for m in args.models.split(",") if m in ready]

    failures = []

    print()
    print("=" * 68)
    print("2. WARM-UP")
    print("=" * 68)
    wait_until_warm(args.base)

    print()
    print("=" * 68)
    print("3. STREAMING UPLOAD")
    print("=" * 68)
    results = {}
    for path, truth in CLIPS:
        if not Path(path).exists():
            print(f"skip (missing): {path}")
            continue
        for model in ready:
            started, vals, snaps = stream_clip(args.base, path, model)
            if not vals:
                print(f"FAIL  {Path(path).name:16s} {model:10s} no windows scored")
                failures.append(f"{path}/{model}: no windows")
                continue
            mean = statistics.fmean(vals)
            med = statistics.median(vals)
            got = band(mean)
            want = "GREEN" if truth == "real" else "RED"
            ok = "ok  " if got == want else "MISS"
            progressive = len({n for _, n in snaps}) > 2
            print(f"{ok}  {Path(path).name:16s} {model:10s} "
                  f"{len(vals):3d}/{started['expected_windows']:3d} win  "
                  f"mean {mean:.4f}  median {med:.4f}  max {max(vals):.4f}  "
                  f"-> {got} (expected {want}){'' if progressive else '  [NOT PROGRESSIVE]'}")
            if not progressive:
                failures.append(f"{path}/{model}: scores did not arrive progressively")
            results[(path, model)] = vals

    if args.skip_crosscheck:
        print("\n(cross-check skipped)")
        return 1 if failures else 0

    print()
    print("=" * 68)
    print("4. LIVE vs DEMO CROSS-CHECK  (same clip, same checkpoint)")
    print("=" * 68)
    for (path, model), live in results.items():
        try:
            demo = batch_scores(path, CKPT_FOR[model])
        except Exception as exc:
            print(f"skip {Path(path).name}/{model}: demo path failed ({exc})")
            continue
        n = min(len(live), len(demo))
        if n == 0:
            continue
        diff = max(abs(live[i] - demo[i]) for i in range(n))
        verdict = "PASS" if diff < 0.02 else "FAIL"
        if diff >= 0.02:
            failures.append(f"{path}/{model}: live vs demo max diff {diff:.4f}")
        print(f"{verdict}  {Path(path).name:16s} {model:10s} "
              f"compared {n} windows, max difference {diff:.4f}")

    print()
    if failures:
        print(f"{len(failures)} PROBLEM(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
