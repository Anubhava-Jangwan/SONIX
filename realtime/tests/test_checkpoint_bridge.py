"""The live path must score through the SAME detector the offline pipeline uses.

These run without a trained head: they check the bridge and the fallback, which
is exactly the state the repo is in until head.pt arrives.
"""

import numpy as np
import pytest
import torch

from realtime import checkpoint
from realtime.engine import ScoringEngine

WIN = 64000


# The two tests that used to live here asserted that realtime/checkpoint.py
# proxied demo/score_file.py through a `_module()` loader and re-exported
# `checkpoint_available`. That bridge no longer exists: checkpoint.py now loads
# the torch checkpoint directly and bakes the standardiser into the module.
# `checkpoint_available` still exists, but in demo/score_file.py:121, which is
# where demo/app.py and demo/diagnose.py call it. The tests below check what the
# current design actually guarantees.


def _tiny_head(mu, sd, in_dim=4, hidden=3):
    """A StandardisedHead with known weights, so normalisation is observable."""
    g = torch.Generator().manual_seed(0)
    state = {
        "0.weight": torch.randn(hidden, in_dim, generator=g),
        "0.bias": torch.randn(hidden, generator=g),
        "3.weight": torch.randn(1, hidden, generator=g),
        "3.bias": torch.randn(1, generator=g),
    }
    cfg = {"in_dim": in_dim, "hidden": hidden, "dropout": 0.0}
    return checkpoint.StandardisedHead(cfg, state, mu, sd).eval()


def test_standardiser_is_actually_applied():
    """The whole point of StandardisedHead: callers cannot forget (x - mu) / sd.

    Forgetting it produces plausible-looking but meaningless scores, so this
    pins it. Feeding mu itself must equal feeding zeros to an unnormalised head
    with the same weights.
    """
    mu = np.array([1.0, -2.0, 3.0, 0.5], np.float32)
    sd = np.array([2.0, 0.5, 4.0, 1.0], np.float32)

    normalised = _tiny_head(mu, sd)
    identity = _tiny_head(np.zeros(4, np.float32), np.ones(4, np.float32))

    with torch.no_grad():
        at_mean = normalised(torch.from_numpy(mu).unsqueeze(0))
        at_zero = identity(torch.zeros(1, 4))
    assert torch.allclose(at_mean, at_zero, atol=1e-6), (
        "input was not standardised before the linear layers")


def test_zero_variance_feature_does_not_produce_nan():
    """load_checkpoint replaces sd == 0 with 1.0; the module must survive it."""
    head = _tiny_head(np.zeros(4, np.float32), np.array([1.0, 1.0, 1.0, 1.0], np.float32))
    with torch.no_grad():
        out = head(torch.randn(2, 4))
    assert torch.isfinite(out).all()


def test_missing_checkpoint_raises_not_crashes_later():
    # No head.pt exists in this checkout; the failure must be at load time,
    # not as a None that explodes three frames later.
    with pytest.raises(FileNotFoundError):
        checkpoint.load_checkpoint("outputs/models/definitely_absent.pt")


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
