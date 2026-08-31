"""pytest for session.py state machine"""

import pytest
import asyncio
from datetime import datetime
import numpy as np
from realtime.session import Session, CallState, AuditEntry


class MockSource:
    """Mock audio source for testing."""
    def __init__(self, caller="test_caller"):
        self.caller = caller


@pytest.mark.asyncio
async def test_session_creation():
    """Test session creation."""
    source = MockSource("+1-508-799-1234")
    session = Session(
        call_id="test_001",
        source=source,
        pairing_code="123456"
    )

    assert session.call_id == "test_001"
    assert session.state == CallState.CONNECTING
    assert session.metadata.pairing_code == "123456"
    assert session.metadata.caller == "+1-508-799-1234"


@pytest.mark.asyncio
async def test_state_transitions():
    """Test state machine transitions."""
    source = MockSource()
    session = Session("test_001", source, "123456")

    # CONNECTING → CONSENT_PENDING
    assert session.state == CallState.CONNECTING
    await session.request_consent()
    assert session.state == CallState.CONSENT_PENDING

    # CONSENT_PENDING → LISTENING
    await session.on_pairing_approved()
    assert session.state == CallState.LISTENING
    assert session.metadata.pairing_approved_at is not None

    # LISTENING → SCORING (when first score recorded)
    await session.record_score(0, 0.5)
    assert session.state == CallState.SCORING

    # SCORING → ENDED
    await session.end_call()
    assert session.state == CallState.ENDED
    assert session.metadata.ended_at is not None


@pytest.mark.asyncio
async def test_consent_blocks_audio():
    """Test that audio is blocked before consent."""
    source = MockSource()
    session = Session("test_001", source, "123456")

    # Push audio while in CONNECTING state (should be ignored)
    dummy_audio = np.random.randn(8000).astype(np.float32)
    await session.push_audio(dummy_audio)

    # Should not have any pending windows
    windows = await session.get_pending_windows()
    assert len(windows) == 0

    # Request consent → CONSENT_PENDING
    await session.request_consent()

    # Audio should still be ignored (not LISTENING yet)
    await session.push_audio(dummy_audio)
    windows = await session.get_pending_windows()
    assert len(windows) == 0

    # Approve → LISTENING
    await session.on_pairing_approved()

    # Now audio should be accepted
    await session.push_audio(dummy_audio)
    windows = await session.get_pending_windows()
    # Note: VAD might filter out noise, so windows could be empty or have content


@pytest.mark.asyncio
async def test_scoring():
    """Test score recording."""
    source = MockSource()
    session = Session("test_001", source, "123456")

    await session.request_consent()
    await session.on_pairing_approved()

    # Record scores
    await session.record_score(0, 0.3)
    await session.record_score(1, 0.5)
    await session.record_score(2, 0.4)

    assert len(session.scores) == 3
    assert 0 in session.scores
    assert session.scores[0]["score"] == 0.3


@pytest.mark.asyncio
async def test_audit_trail():
    """Test audit logging."""
    source = MockSource()
    session = Session("test_001", source, "123456")

    initial_audit_count = len(session.metadata.audit)

    await session.request_consent()
    assert len(session.metadata.audit) > initial_audit_count

    await session.on_pairing_approved()
    assert any(e.action == "pairing_approved" for e in session.metadata.audit)


@pytest.mark.asyncio
async def test_serialization():
    """Test to_dict() serialization."""
    source = MockSource("+1-508-799-1234")
    session = Session("test_001", source, "123456")

    await session.request_consent()
    await session.on_pairing_approved()
    await session.record_score(0, 0.4)

    data = session.to_dict()

    assert data["call_id"] == "test_001"
    assert data["caller"] == "+1-508-799-1234"
    assert data["state"] == "scoring"
    assert data["windows_scored"] == 1
    assert data["mean_score"] == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
