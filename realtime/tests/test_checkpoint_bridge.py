"""The live path must score through the SAME detector the offline pipeline uses.

These run without a trained head: they check the bridge and the fallback, which
is exactly the state the repo is in until head.pt arrives.
"""

import numpy as np
import pytest

from realtime import checkpoint
from realtime.engine import ScoringEngine

WIN = 64000


def test_bridge_imports_the_real_detector():
    mod = checkpoint._module()
    # The functions the live path depends on must exist in demo/score_file.py.
    for name in ("_score_windows", "_load_head", "checkpoint_available", "WIN"):
        assert hasattr(mod, name), f"demo/score_file.py has no {name}"
    assert mod.WIN == WIN and mod.TARGET_SR == 16000


def test_missing_checkpoint_raises_not_crashes_later():
    # No head.pt exists in this checkout; the failure must be at load time.
    with pytest.raises(FileNotFoundError):
        checkpoint.load_checkpoint("outputs/models/definitely_absent.pt")
    assert checkpoint.checkpoint_available("outputs/models/definitely_absent.pt") is False


@pytest.mark.asyncio
async def test_engine_scores_raw_windows_in_mock_mode():
    # After the rewiring the engine batches raw 4 s windows, not embeddings.
    engine = ScoringEngine(mock=True)
    batch = np.zeros((3, WIN), dtype=np.float32)
    scores = await engine._score_windows(batch)
    assert scores.shape == (3,)
    assert scores.dtype == np.float32
    assert np.all((scores >= 0) & (scores <= 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
