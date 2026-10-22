"""Continuous microphone scoring, inside the Streamlit process.

This is the "live" path that used to require `python -m realtime.server` on a
second port. It runs on whatever port the demo is already served from, because
nothing here is a server -- it is two plain threads writing into two deques.

Why threads rather than doing the work in the rerun: a Streamlit rerun is not a
clock. If capture lived inside the rerun, every gap between reruns would be
audio that nobody recorded, and the timeline would have holes exactly where the
UI happened to be busy rendering. Instead:

    PortAudio callback ---> self.audio   (ring buffer, ~12 s of tail)
    scorer thread      ---> self.scores  (bounded history of (t, score, key))
    st.fragment        ---> reads both, draws, and touches nothing else

The fragment only ever *reads*. That is what keeps the series continuous while
the head is swapped underneath it.

Swapping heads mid-stream is nearly free, and that is a property of the
architecture rather than a trick: the 300M-param front-end is frozen and shared
across every head, so changing model means loading a ~300k-param MLP and
nothing else. score_file caches the front-end in _FE and heads in _HEADS, so
after the first use of a head the swap is a dict lookup. The capture thread is
never told about any of it, so the audio -- and the graph -- do not stop.

Audio is buffered at the DEVICE's native rate and resampled to 16 kHz one whole
window at a time, not per callback block. Resampling each 100 ms block
independently would put a polyphase edge artefact every 100 ms into audio whose
whole purpose is to be judged on its fine detail.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from realtime.resample import resample                          # noqa: E402

SR = 16000            # what wav2vec2 expects
WIN = 64000           # 4.0 s at SR -- one model window
WIN_SECONDS = 4.0
HOP = 0.5             # seconds between scored windows
TAIL_SECONDS = 12.0   # how much audio to keep for the waveform strip


class LiveMic:
    """One microphone, scored continuously. Safe to keep in st.cache_resource.

    Every attribute the UI reads is either a deque (bounded, append-only from
    one thread) or a plain scalar. No Streamlit API is touched off the main
    thread -- the background threads have no ScriptRunContext and must not.
    """

    def __init__(self, history: int = 600):
        self.audio: deque = deque()             # native-rate float32 samples
        self.scores: deque = deque(maxlen=history)   # (t_sec, score, model_key)
        self.lock = threading.Lock()

        # Read by the scorer on every window, written by the UI whenever the
        # picker changes. A plain attribute is enough: worst case a single
        # window is scored with the head selected a moment ago, which is
        # labelled per-point anyway.
        self.ckpt: str | None = None
        self.model_key: str | None = None

        self.running = False
        self.error: str | None = None
        self.started_at: float | None = None
        self.native_sr: int = SR
        self.device_name: str = ""
        self.windows_scored = 0
        self.windows_dropped = 0        # skipped to stay real-time, see _scorer
        self.warming = False            # loading the frozen front-end
        self.warm = False

        self._stream = None
        self._thread: threading.Thread | None = None

    # -- devices -----------------------------------------------------------

    @staticmethod
    def input_devices() -> list[tuple[int, str]]:
        """(index, name) for every device that can record.

        Worth surfacing in the UI, because picking one is a real choice: a
        microphone is your own voice, while a loopback device ("Stereo Mix",
        "What U Hear") carries whoever is on the call -- which is the side you
        actually want to screen for a clone.
        """
        try:
            import sounddevice as sd
        except Exception:
            return []
        try:
            return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                    if d["max_input_channels"] > 0]
        except Exception:
            return []

    @staticmethod
    def _resolve_device(sd, device):
        """PortAudio's default input is not always usable. On this machine
        MME, DirectSound and WASAPI each expose zero inputs and
        sd.default.device reports (-1, 1); only WDM-KS lists the microphone.
        Passing -1 fails with 'Error querying device -1', so fall back to a
        device that actually has a recording channel."""
        if device is not None:
            return int(device)
        try:
            default = sd.default.device[0]
        except Exception:
            default = -1
        if isinstance(default, int) and default >= 0:
            return default
        devs = LiveMic.input_devices()
        if not devs:
            raise RuntimeError("no input device with a recording channel")
        for idx, name in devs:                  # prefer a mic over a loopback
            if "mic" in name.lower():
                return idx
        return devs[0][0]

    # -- lifecycle ---------------------------------------------------------

    def start(self, ckpt: str, model_key: str, device=None) -> bool:
        """Open an input device and begin scoring. Idempotent."""
        if self.running:
            self.ckpt, self.model_key = ckpt, model_key
            return True

        try:
            import sounddevice as sd            # imported late: needs PortAudio
        except Exception as exc:                # pragma: no cover - env specific
            self.error = (f"sounddevice is not usable on this machine ({exc}). "
                          "Install it with: pip install sounddevice")
            return False

        try:
            idx = self._resolve_device(sd, device)
            info = sd.query_devices(idx)
            # Open at the device's OWN rate. WDM-KS does no rate conversion,
            # so asking a 44.1 kHz mic for 16 kHz fails outright with
            # "Invalid device" -- which reads like a broken mic, not a
            # rejected sample rate.
            native = int(round(float(info["default_samplerate"])))
            self.device_name = str(info["name"])
        except Exception as exc:
            self.error = f"Could not query an input device: {exc}"
            return False

        self.ckpt, self.model_key = ckpt, model_key
        self.error = None
        self.native_sr = native
        with self.lock:
            self.audio = deque(maxlen=int(native * TAIL_SECONDS))
        self.scores.clear()
        self.windows_scored = self.windows_dropped = 0
        self.started_at = time.monotonic()

        def on_audio(indata, _frames, _time, status):
            if status:                                   # overflow etc.
                self.error = str(status)
            with self.lock:
                self.audio.extend(indata[:, 0].astype(np.float32))

        try:
            self._stream = sd.InputStream(
                samplerate=native, channels=1, dtype="float32", device=idx,
                blocksize=int(native * 0.1), callback=on_audio)
            self._stream.start()
        except Exception as exc:
            self.error = f"Could not open '{self.device_name}': {exc}"
            self._stream = None
            return False

        self.running = True
        self._thread = threading.Thread(target=self._scorer, daemon=True,
                                        name="sonix-live-scorer")
        self._thread.start()
        return True

    def stop(self):
        self.running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # -- the scoring loop --------------------------------------------------

    def _window(self) -> np.ndarray | None:
        """The newest 4 s of audio, resampled to 16 kHz, or None if the buffer
        has not filled yet."""
        need = int(self.native_sr * WIN_SECONDS)
        with self.lock:
            if len(self.audio) < need:
                return None
            buf = np.fromiter(self.audio, dtype=np.float32, count=len(self.audio))
        win = resample(buf[-need:], self.native_sr, SR)
        if win.size < WIN:                      # rounding at odd rates
            win = np.pad(win, (0, WIN - win.size), mode="edge")
        return win[:WIN].astype(np.float32)

    def _scorer(self):
        """Score the NEWEST full window every HOP seconds, forever.

        Deliberately always takes the newest window instead of draining a
        queue. The front-end is ~300M params: if a window takes longer than
        HOP to score, a queue would grow without bound and the "live" reading
        would drift further behind the speaker with every second. Dropping the
        windows we could not keep up with bounds the lag at one window, and
        windows_dropped makes that visible rather than silent.
        """
        import score_file as S

        # Load the frozen front-end BEFORE the loop. It is ~300M params and
        # takes tens of seconds; done lazily inside the first scored window it
        # looks exactly like a hang -- you press Start, speak, and nothing
        # appears for a minute. Note this is invisible in a silent room: VAD
        # short-circuits silent windows without ever touching the model, so
        # the warmup only bites the moment someone actually speaks.
        self.warming = True
        try:
            S._load_head(self.ckpt)
            self.warm = True
        except Exception as exc:
            self.error = f"Could not load the model: {exc}"
        finally:
            self.warming = False

        next_at = time.monotonic()
        while self.running:
            now = time.monotonic()
            if now < next_at:
                time.sleep(min(0.05, next_at - now))
                continue
            behind = int((now - next_at) / HOP)
            if behind:
                self.windows_dropped += behind
            next_at = now + HOP

            win = self._window()
            if win is None:
                continue

            ckpt, key = self.ckpt, self.model_key
            try:
                score = S._score_windows([win], ckpt_path=ckpt)[0]
            except Exception as exc:
                self.error = f"Scoring failed: {exc}"
                time.sleep(0.5)
                continue

            self.scores.append((time.monotonic() - (self.started_at or now),
                                float(score), key))
            self.windows_scored += 1

    # -- what the UI reads -------------------------------------------------

    def series(self):
        """(times, scores, model_keys) as lists -- a snapshot, safe to plot."""
        snap = list(self.scores)
        if not snap:
            return [], [], []
        t, s, k = zip(*snap)
        return list(t), list(s), list(k)

    def tail(self, seconds: float = 8.0) -> np.ndarray:
        """The most recent `seconds` of captured audio, for the waveform. Left
        at the native rate: it is only ever reduced to a peak envelope."""
        with self.lock:
            if not self.audio:
                return np.zeros(1, dtype=np.float32)
            buf = np.fromiter(self.audio, dtype=np.float32, count=len(self.audio))
        return buf[-int(self.native_sr * seconds):]

    def level(self, seconds: float = 0.35):
        """(rms, peak) of the most recent `seconds` of audio.

        Read straight off the capture buffer rather than inferred from the
        scores, because it has to answer a question the scores cannot: when
        nothing is being scored, is the microphone silent, or is it picking up
        sound that never clears the silence gate? Those look identical on a
        score timeline and need completely different fixes.
        """
        n = int(self.native_sr * seconds)
        with self.lock:
            if not self.audio:
                return 0.0, 0.0
            buf = np.fromiter(self.audio, dtype=np.float32, count=len(self.audio))
        buf = buf[-n:]
        if buf.size == 0:
            return 0.0, 0.0
        return float(np.sqrt(np.mean(buf.astype(np.float64) ** 2))), \
            float(np.max(np.abs(buf)))

    def envelope(self, seconds: float = 1.6, points: int = 64):
        """Peak-per-bucket envelope of the recent tail, normalised to 0..1.

        This is what the on-screen bubble is actually shaped by -- the real
        captured audio, not a decorative animation. A bubble that wobbles while
        the room is silent would be worse than no bubble at all.
        """
        n = int(self.native_sr * seconds)
        with self.lock:
            if not self.audio:
                return np.zeros(points, dtype=float)
            buf = np.fromiter(self.audio, dtype=np.float32, count=len(self.audio))
        buf = buf[-n:]
        if buf.size < points:
            return np.zeros(points, dtype=float)
        per = buf.size // points
        env = np.abs(buf[:per * points].reshape(points, per)).max(axis=1)
        peak = float(env.max())
        return (env / peak) if peak > 0 else env

    def buffering(self) -> float:
        """0..1 -- how full the first 4 s window is. Before this hits 1.0 there
        is nothing honest to score, so the UI shows a filling state instead of
        a repeat-padded number (padding measurably shifts the score)."""
        need = int(self.native_sr * WIN_SECONDS)
        with self.lock:
            return min(1.0, len(self.audio) / need)
