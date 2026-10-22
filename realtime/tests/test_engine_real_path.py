"""The non-mock scoring path must actually run, and a bad batch must not kill the engine.

Regression tests for uploads reporting "0 / N windows scored" while the silence
gate passed most windows. Two defects, one symptom:

  1. engine._head_forward used `torch` without importing it, so every real batch
     raised NameError. Mock mode returns before that line, so the whole test
     suite stayed green while real scoring was dead.
  2. ScoringEngine.run() caught only CancelledError, so that NameError escaped
     the loop and killed the scoring task silently. The server kept accepting
     audio and reporting zero scored windows with nothing in the log.
"""

import ast
import asyncio
from pathlib import Path

import numpy as np
import pytest

from realtime.engine import ScoringEngine

ENGINE_PY = Path(__file__).resolve().parents[1] / "engine.py"


def test_every_name_used_in_engine_is_bound():
    """No module-level or function-local use of an unimported top-level name.

    This is the general form of the `torch` bug: a name used in a rarely taken
    branch that nothing imports. Mock-mode tests cannot catch it, because they
    never execute that branch.
    """
    tree = ast.parse(ENGINE_PY.read_text(encoding="utf-8"))

    bound = set(dir(__builtins__)) | set(vars(__builtins__)) if isinstance(__builtins__, dict) \
        else set(dir(__builtins__))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for a in (list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs)):
                    bound.add(a.arg)
                if args.vararg:
                    bound.add(args.vararg.arg)
                if args.kwarg:
                    bound.add(args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.comprehension,)):
            pass
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)

    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(used - bound)
    assert not missing, f"engine.py uses unbound name(s): {missing}"


def test_head_forward_imports_torch():
    """Specifically: the real inference path must be able to reach torch."""
    src = ENGINE_PY.read_text(encoding="utf-8")
    body = src.split("def _head_forward", 1)[1].split("\n    # ---", 1)[0]
    assert "import torch" in body or "\nimport torch" in src, \
        "_head_forward uses torch but nothing imports it"


@pytest.mark.asyncio
async def test_a_failing_batch_does_not_kill_the_loop():
    """One bad batch is dropped and counted; the engine keeps scoring."""
    engine = ScoringEngine(mock=True)

    class _Session:
        call_id = "t1"
        window_count = 1
        model_key = None

        def __init__(self):
            from realtime.session import CallState
            self.state = CallState.LISTENING
            self._queue = [np.zeros(64000, np.float32)]

        async def get_pending_windows(self):
            out, self._queue = self._queue, []
            return out

        async def requeue_windows(self, w):
            self._queue[:0] = w

        async def record_score(self, *a, **k):
            pass

    engine.sessions["t1"] = _Session()

    boom = {"n": 0}

    async def _explode(_windows):
        boom["n"] += 1
        raise RuntimeError("simulated inference failure")

    engine._embed_windows = _explode

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.35)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert boom["n"] >= 1, "the batch never reached the scorer"
    assert engine.failed_batches >= 1, "the failure was not counted"
    assert task.cancelled() or task.done(), "loop should have survived to cancellation"


def test_stats_expose_failed_batches():
    stats = ScoringEngine(mock=True).get_stats()
    assert "failed_batches" in stats, \
        "silent batch failures must be visible in /api/status"
