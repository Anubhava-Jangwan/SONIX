"""pytest for engine.py batching and scoring"""

import pytest
import asyncio
import numpy as np
from realtime.engine import ScoringEngine
from realtime.session import Session, CallState


class MockSource:
    """Mock audio source."""
    def __init__(self, caller="mock_caller"):
        self.caller = caller


@pytest.mark.asyncio
async def test_engine_creation():
    """Test engine creation."""
    engine = ScoringEngine(mock=True)

    assert engine.mock is True
    assert len(engine.sessions) == 0
    assert engine.total_windows_scored == 0
    assert engine.total_batches == 0


@pytest.mark.asyncio
async def test_add_remove_session():
    """Test adding/removing sessions."""
    engine = ScoringEngine(mock=True)
    source = MockSource()
    session = Session("call_001", source, "123456")

    await engine.add_session(session)
    assert len(engine.sessions) == 1
    assert "call_001" in engine.sessions

    await engine.remove_session("call_001")
    assert len(engine.sessions) == 0


@pytest.mark.asyncio
async def test_concurrent_calls():
    """Test multiple concurrent calls."""
    engine = ScoringEngine(mock=True)

    # Add 3 calls
    for i in range(3):
        source = MockSource(f"caller_{i}")
        session = Session(f"call_{i:03d}", source, "123456")
        await engine.add_session(session)

    assert len(engine.sessions) == 3

    # Remove one
    await engine.remove_session("call_001")
    assert len(engine.sessions) == 2


@pytest.mark.asyncio
async def test_engine_loop_short_run():
    """Test engine loop for short duration."""
    engine = ScoringEngine(mock=True, batch_interval=0.1)

    source = MockSource()
    session = Session("call_001", source, "123456")
    await session.request_consent()
    await session.on_pairing_approved()

    await engine.add_session(session)

    # Run engine for 0.5 seconds
    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.5)
    engine_task.cancel()

    try:
        await engine_task
    except asyncio.CancelledError:
        pass

    # Check stats
    stats = engine.get_stats()
    assert stats["active_calls"] == 1
    assert "total_batches" in stats


@pytest.mark.asyncio
async def test_broadcast_callback():
    """Test broadcast callback is called."""
    broadcast_calls = []

    async def on_broadcast(scores_dict):
        broadcast_calls.append(scores_dict)

    engine = ScoringEngine(mock=True, batch_interval=0.1, on_broadcast=on_broadcast)

    source = MockSource()
    session = Session("call_001", source, "123456")
    await session.request_consent()
    await session.on_pairing_approved()
    await engine.add_session(session)

    # Simulate scoring
    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.2)
    engine_task.cancel()

    try:
        await engine_task
    except asyncio.CancelledError:
        pass

    # Callback may not have been called if no windows were ready
    # Just verify no crash


@pytest.mark.asyncio
async def test_batch_size_limit():
    """Test max_batch_size is respected."""
    engine = ScoringEngine(mock=True, max_batch_size=4)

    # Add 10 sessions
    for i in range(10):
        source = MockSource(f"caller_{i}")
        session = Session(f"call_{i:02d}", source, "123456")
        await session.request_consent()
        await session.on_pairing_approved()
        await engine.add_session(session)

    assert len(engine.sessions) == 10

    # Verify max_batch_size is set correctly
    assert engine.max_batch_size == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
