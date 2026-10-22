"""Why did an upload score 0 of N windows?

Runs the exact upload path -- WavFileSource -> RingBuffer -> adaptive VAD floor
-> VAD -- against a file on disk, and prints where the windows die. No server,
so a stale process cannot confound the answer, and no head.pt is needed: this
covers everything up to the scorer, which is where "0 / N windows scored" is
decided.

    python -m realtime.diagnose_upload path\\to\\output.wav

Exit code 0 if windows survive the gate, 1 if none do.
"""

import argparse
import sys

import numpy as np

from realtime.ringbuffer import RingBuffer
from realtime.resample import TARGET_SR
from realtime.source import WavFileSource
from realtime.vad import VAD


def adaptive_floor(samples, frame=400):
    """Verbatim copy of SonicServer._adaptive_vad_floor, minus the aiohttp import."""
    w = np.asarray(samples, dtype=np.float32).reshape(-1)
    if w.size < frame:
        return None
    nf = w.size // frame
    rms = np.sqrt(np.mean(np.square(w[:nf * frame].reshape(nf, frame).astype(np.float64)), axis=1))
    speech = float(np.percentile(rms, 90))
    floor = float(np.clip(speech * 0.12, 0.0006, 0.01))
    floor = min(floor, speech * 0.5)
    return floor, speech, 20.0 * np.log10(max(floor, 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--gate", choices=("auto", "strict", "off"), default="auto",
                    help="matches the vad= form field on /api/score-file")
    ap.add_argument("--show", type=int, default=8, help="per-window rows to print")
    args = ap.parse_args(argv)

    src = WavFileSource(args.path)
    try:
        src._load()
    except Exception as exc:
        print(f"could not decode {args.path}: {exc}")
        return 2
    a = src._samples

    print(f"file          {args.path}")
    print(f"decoded       {a.size} samples @ {TARGET_SR} Hz  ({a.size / TARGET_SR:.2f}s), "
          f"source rate {getattr(src, 'source_sample_rate', '?')}")
    print(f"overall RMS   {float(np.sqrt(np.mean(a.astype(np.float64) ** 2))):.6f}   "
          f"peak {float(np.max(np.abs(a))) if a.size else 0:.6f}")

    if a.size == 0:
        print("\nVERDICT: the file decoded to zero samples. Nothing downstream can score.")
        return 1

    # --- the gate the server would use -------------------------------------
    if args.gate == "off":
        vad, note = VAD(threshold_energy=0.0, zcr_ceiling=1.0, min_speech_ratio=0.0), \
                    "disabled - every window scored"
    elif args.gate == "strict":
        vad, note = VAD(), "server default (fixed 0.01 floor)"
    else:
        adapt = adaptive_floor(a)
        if adapt:
            floor, speech, db = adapt
            vad = VAD(threshold_energy=floor)
            note = f"auto {floor:.5f} ({db:.1f} dBFS) from speech level {speech:.5f}"
        else:
            vad, note = VAD(), "clip too short to adapt; server default"
    print(f"silence gate  {note}")

    # --- windowing, exactly as the live path does it -----------------------
    rb = RingBuffer()
    windows = []
    for i in range(0, a.size, 8000):
        rb.push(a[i:i + 8000])
        windows.extend(rb.get_emitted_windows())
    print(f"windows       {len(windows)} emitted by the ring buffer "
          f"({rb.window} samples, hop {rb.hop})")

    if not windows:
        need = rb.window / TARGET_SR
        print(f"\nVERDICT: no window was ever completed -- the clip is shorter than "
              f"{need:.1f}s once decoded. The server repeat-pads this case; a raw "
              f"clip this short would score nothing.")
        return 1

    passed, rows = 0, []
    for i, w in enumerate(windows):
        ok = vad.is_speech(w)
        passed += ok
        s = vad.last_stats
        rows.append((i, s.get("rms", 0.0), s.get("peak", 0.0),
                     s.get("zcr", 0.0), s.get("speech_ratio", 0.0), ok))

    print(f"passed VAD    {passed} / {len(windows)}")
    print()
    print(f"  {'win':>4} {'rms':>9} {'peak':>8} {'zcr':>6} {'speech':>7}  gate")
    shown = rows if len(rows) <= args.show else rows[:args.show // 2] + rows[-(args.show // 2):]
    last = None
    for i, rms, peak, zcr, ratio, ok in shown:
        if last is not None and i != last + 1:
            print(f"  {'...':>4}")
        print(f"  {i:>4} {rms:>9.6f} {peak:>8.5f} {zcr:>6.3f} {ratio:>7.3f}  "
              f"{'pass' if ok else 'REJECT'}")
        last = i

    print()
    if passed:
        print(f"VERDICT: the gate passes {passed} window(s). If the upload still "
              f"reports 0 scored, the windows are dying AFTER the gate -- check "
              f"the server console for 'upload feed finished (N windows emitted)' "
              f"and whether /api/models still says \"warming\": true.")
        return 0

    # Nothing passed -- say which of the three gates did it.
    rms_fail = sum(1 for r in rows if r[1] <= vad.threshold_energy)
    zcr_fail = sum(1 for r in rows if r[3] >= vad.zcr_ceiling)
    print("VERDICT: every window was rejected by the silence gate.")
    print(f"  energy below floor : {rms_fail}/{len(rows)} windows "
          f"(floor {vad.threshold_energy:.6f})")
    print(f"  zcr above ceiling  : {zcr_fail}/{len(rows)} windows "
          f"(ceiling {vad.zcr_ceiling:.2f})")
    print(f"  needed speech ratio: {vad.min_speech_ratio:.2f}")
    print()
    print("  Re-run with --gate off to confirm the audio scores when ungated.")
    print("  If it does, this clip is quieter or noisier than the gate expects;")
    print("  start the server with --vad-energy 0.0003, or send vad=off on the upload.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
