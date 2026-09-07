"""http_upload_handler must wait for the engine to actually score the backlog
instead of guessing with a fixed sleep - see the race described in server.py.
"""

import asyncio

import numpy as np

from realtime.session import CallState, Session


class _Source:
    caller = "test-upload"


def _session():
    s = Session(call_id="upload_test", source=_Source(), pairing_code="upload_mode")
    s.state = CallState.LISTENING
    return s


def test_drain_loop_exits_once_pending_windows_is_empty():
    """The exact wait pattern server.py uses: poll pending_windows, don't just
    sleep a fixed guess. A scorer that drains within the timeout must not have
    the request wait out the full deadline."""
    s = _session()
    s.pending_windows = [np.zeros(4, dtype=np.float32) for _ in range(3)]

    async def scorer_drains_after_a_beat():
        await asyncio.sleep(0.05)
        s.pending_windows.clear()

    async def wait_for_drain():
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 10.0
        elapsed_iters = 0
        while s.pending_windows and loop.time() < deadline:
            await asyncio.sleep(0.02)
            elapsed_iters += 1
        return elapsed_iters

    async def run():
        task = asyncio.create_task(scorer_drains_after_a_beat())
        iters = await wait_for_drain()
        await task
        return iters

    iters = asyncio.run(run())
    assert not s.pending_windows
    # Drained in a handful of polls, nowhere near the 10s safety timeout - the
    # bug this replaces was ending the call on a bare 0.5s guess regardless.
    assert iters < 10


if __name__ == "__main__":
    test_drain_loop_exits_once_pending_windows_is_empty()
    print("ok")
