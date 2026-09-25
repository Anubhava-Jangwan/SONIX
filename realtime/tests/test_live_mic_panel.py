"""Backlog drop, the embedded mic panel, and the chart's clock.

The inline Live-microphone block is the /mic page in embed view. What is worth
pinning down is the part that is not visible on screen: that a scorer running
behind real time drops the oldest window instead of queueing forever, that the
embed view drops its own frame rather than nesting a card inside the dashboard's,
and that a score carries the time its AUDIO was captured - not the time it
finished scoring, which is what used to make the live line trail the call.
"""

import asyncio

import numpy as np

from realtime.miccapture import EMBED_CSS, PAGE
from realtime.session import CallState, Session


class _Source:
    caller = "test-mic"


def _session():
    return Session(call_id="mic_test", source=_Source(), pairing_code="123456")


def test_backlog_drops_oldest_and_counts_it():
    s = _session()
    s.max_pending_windows = 4

    # Windows are tagged by their first sample so we can tell which survived.
    for i in range(10):
        s.pending_windows.append(np.full(4, float(i), dtype=np.float32))
        s._trim_pending()

    assert len(s.pending_windows) == 4, "backlog is not bounded"
    assert s.windows_dropped_backlog == 6
    # The NEWEST four survive - an old window says nothing a newer one does not.
    assert [float(w[0]) for w in s.pending_windows] == [6, 7, 8, 9]


def test_requeue_is_also_bounded():
    """The engine puts unbatched windows back; that path must be capped too,
    or a slow scorer refills the queue as fast as _trim_pending drains it."""
    s = _session()
    s.max_pending_windows = 4
    s.pending_windows = [np.zeros(4, dtype=np.float32) for _ in range(3)]

    asyncio.run(s.requeue_windows([np.ones(4, dtype=np.float32) for _ in range(5)]))

    assert len(s.pending_windows) == 4
    assert s.windows_dropped_backlog == 4


def test_telemetry_reports_drops():
    s = _session()
    s.max_pending_windows = 1
    for _ in range(3):
        s.pending_windows.append(np.zeros(4, dtype=np.float32))
        s._trim_pending()

    backlog = s.telemetry()["backlog"]
    assert backlog["dropped"] == 2
    assert backlog["pending"] == 1


def test_embed_view_drops_its_own_frame_but_reuses_the_capture_code():
    embedded = PAGE + EMBED_CSS
    # No card inside the dashboard's own panel: the wrapper loses its width cap
    # and padding, the page heading goes, and the ground turns transparent.
    for rule in ("max-width:none;padding:0", "h1,.sub{display:none}",
                 "background:transparent"):
        assert rule in EMBED_CSS, f"embed view still frames itself: missing {rule}"
    assert 'class="wrap"' in PAGE, "nothing for the embed CSS to unframe"
    # The live chart and the controls are the reason the panel exists - they stay.
    hidden = EMBED_CSS[EMBED_CSS.index("display:none")-40:EMBED_CSS.index("display:none")]
    for kept in ("graph", "btn", "headline", "status"):
        assert kept not in hidden, f"embed view hid the {kept} it exists to show"
    # Same worklet + socket, not a second implementation.
    for shared in ("registerProcessor('cap'", "start_mic_call", "floatToPCM16"):
        assert shared in embedded, f"embed view lost: {shared}"


def test_score_is_plotted_on_the_audio_clock_not_the_arrival_clock():
    """A window scored late must still land at the time it was captured."""
    s = _session()
    s.state = CallState.LISTENING

    t0 = s.metadata.started_at.timestamp()
    s.window_log.append({"t": t0 + 7.5, "window_idx": 3, "vad_passed": True})

    assert s.window_time(3) == 7.5
    assert s.window_time(99) is None, "unknown window must not invent a time"

    # And the page plots that t rather than reading its own clock on arrival.
    assert "pushScore(it.t, it.score)" in PAGE


def test_every_window_of_a_batch_is_plotted():
    # A batch of 8 used to draw one point and leave 3.5 s of call unplotted.
    assert "entry.batch && entry.batch.length" in PAGE
    assert "items.forEach" in PAGE


if __name__ == "__main__":
    test_backlog_drops_oldest_and_counts_it()
    test_requeue_is_also_bounded()
    test_telemetry_reports_drops()
    test_embed_view_drops_its_own_frame_but_reuses_the_capture_code()
    test_score_is_plotted_on_the_audio_clock_not_the_arrival_clock()
    test_every_window_of_a_batch_is_plotted()
    print("ok")
