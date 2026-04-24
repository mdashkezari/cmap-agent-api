"""test_upsert_retry.py — unit tests for v226.

Covers the retry-with-backoff behavior added in v226:

- 502/503/504 proxy-layer errors are retried up to QDRANT_UPSERT_MAX_RETRIES
- Non-transient errors (400, 404, 500) are not retried
- Retry count honors the settings knob; 0 disables retry
- Backoff schedule is exponential + jittered (1s, 2s, 4s, 8s, 16s, cap 30s)
- Batch size is read from QDRANT_UPSERT_BATCH_SIZE
- The transient-error predicate classifies correctly

The tests stub out the Qdrant client, the embedder, and the sparse
encoder so no network calls are made.
"""
from __future__ import annotations

import re as _re
import sys
import types as _types
from pathlib import Path
from unittest.mock import MagicMock, patch


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# --- Load helpers without importing the full module (avoids conda/fastembed) ---

def _load_predicate_and_backoff():
    """Pull ``_is_transient_proxy_error`` and ``_retry_backoff_seconds``
    from qdrant_kb.py source directly, so tests don't need fastembed
    or the OpenAI client installed."""
    path = _SRC / "cmap_agent" / "rag" / "qdrant_kb.py"
    src = path.read_text()

    pred_m = _re.search(
        r"_RETRYABLE_STATUS.*?def _retry_backoff_seconds\(attempt: int\) -> float:.*?return base \+ random\.uniform\(0, _RETRY_BACKOFF_JITTER_S\)",
        src,
        _re.DOTALL,
    )
    if not pred_m:
        raise RuntimeError("could not locate retry helpers")

    # Stub UnexpectedResponse with the minimum shape we need.
    class _FakeUnexpectedResponse(Exception):
        def __init__(self, status_code: int):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    ns: dict = {
        "UnexpectedResponse": _FakeUnexpectedResponse,
        "frozenset": frozenset,
        "BaseException": BaseException,
        "isinstance": isinstance,
        "getattr": getattr,
        "min": min,
        "max": max,
    }
    # import random, into ns
    import random as _random
    ns["random"] = _random

    exec(pred_m.group(0), ns)
    return (
        ns["_is_transient_proxy_error"],
        ns["_retry_backoff_seconds"],
        ns["_RETRYABLE_STATUS"],
        _FakeUnexpectedResponse,
    )


(
    _is_transient_proxy_error,
    _retry_backoff_seconds,
    _RETRYABLE_STATUS,
    _FakeUnexpectedResponse,
) = _load_predicate_and_backoff()


_PASS = 0
_FAIL = 0


def _check(cond: bool, label: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [pass] {label}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {label}")


# ---------------------------------------------------------------------
# Predicate tests — _is_transient_proxy_error
# ---------------------------------------------------------------------

def test_predicate_502_is_transient() -> None:
    exc = _FakeUnexpectedResponse(502)
    _check(_is_transient_proxy_error(exc) is True, "502 is classified transient")


def test_predicate_503_is_transient() -> None:
    exc = _FakeUnexpectedResponse(503)
    _check(_is_transient_proxy_error(exc) is True, "503 is classified transient")


def test_predicate_504_is_transient() -> None:
    exc = _FakeUnexpectedResponse(504)
    _check(_is_transient_proxy_error(exc) is True, "504 is classified transient")


def test_predicate_500_is_NOT_transient() -> None:
    """500 is a real cluster-internal error, not a proxy hiccup."""
    exc = _FakeUnexpectedResponse(500)
    _check(_is_transient_proxy_error(exc) is False, "500 is NOT retried")


def test_predicate_400_is_NOT_transient() -> None:
    exc = _FakeUnexpectedResponse(400)
    _check(_is_transient_proxy_error(exc) is False, "400 is NOT retried")


def test_predicate_404_is_NOT_transient() -> None:
    exc = _FakeUnexpectedResponse(404)
    _check(_is_transient_proxy_error(exc) is False, "404 is NOT retried")


def test_predicate_connection_error_NOT_retried() -> None:
    """Generic exceptions (e.g. ConnectionError) are not in the retry set.
    Only the specific UnexpectedResponse 5xx proxy codes are.  Network
    errors at the socket level are handled by qdrant-client itself."""
    exc = ConnectionError("network down")
    _check(_is_transient_proxy_error(exc) is False, "ConnectionError NOT retried")


# ---------------------------------------------------------------------
# Backoff schedule tests
# ---------------------------------------------------------------------

def test_backoff_attempt_0_around_1s() -> None:
    """Attempt 0 → 2^0 = 1s base + [0, 0.5) jitter → [1.0, 1.5)."""
    vals = [_retry_backoff_seconds(0) for _ in range(20)]
    _check(all(1.0 <= v < 1.5 for v in vals), f"attempt 0 in [1.0, 1.5) — got min={min(vals):.3f}, max={max(vals):.3f}")


def test_backoff_attempt_2_around_4s() -> None:
    vals = [_retry_backoff_seconds(2) for _ in range(20)]
    _check(all(4.0 <= v < 4.5 for v in vals), f"attempt 2 in [4.0, 4.5)")


def test_backoff_attempt_5_capped_at_30s() -> None:
    """2^5 = 32 but cap is 30.  Output should be in [30.0, 30.5)."""
    vals = [_retry_backoff_seconds(5) for _ in range(20)]
    _check(all(30.0 <= v < 30.5 for v in vals), f"attempt 5 capped at 30.0-30.5 — got {vals[:3]}")


def test_backoff_attempt_10_still_capped() -> None:
    """Very large attempt counts should still respect the 30s cap."""
    v = _retry_backoff_seconds(10)
    _check(30.0 <= v < 30.5, f"attempt 10 capped — got {v:.3f}")


# ---------------------------------------------------------------------
# Integration test — simulate upsert() with failing-then-succeeding client
# ---------------------------------------------------------------------
#
# These tests exercise the full QdrantKB.upsert retry loop.  To avoid
# pulling in qdrant-client / fastembed / openai at test time, we:
#   - stub those modules in sys.modules before importing QdrantKB
#   - then import and instantiate a QdrantKB subclass that short-circuits
#     __init__
#   - drive upsert() via a mock client.upsert() with a scripted side_effect


def _install_qdrant_kb_stubs() -> None:
    """Ensure qdrant_client / fastembed / cmap_agent.rag.embedder are
    importable with stubbed modules.  Idempotent."""
    # qdrant_client
    if "qdrant_client" not in sys.modules:
        mod = _types.ModuleType("qdrant_client")

        class _QC:
            def __init__(self, *a, **kw):
                pass
        mod.QdrantClient = _QC
        mod.models = _types.SimpleNamespace(
            PointStruct=lambda **kw: _types.SimpleNamespace(**kw),
            SparseVector=lambda **kw: _types.SimpleNamespace(**kw),
            VectorParams=lambda **kw: _types.SimpleNamespace(**kw),
            SparseVectorParams=lambda **kw: _types.SimpleNamespace(**kw),
            Distance=_types.SimpleNamespace(COSINE="cosine"),
            Prefetch=lambda **kw: None,
            Query=lambda **kw: None,
            FusionQuery=lambda **kw: None,
            Fusion=_types.SimpleNamespace(RRF="rrf", DBSF="dbsf"),
            Filter=lambda **kw: None,
            FieldCondition=lambda **kw: None,
            MatchValue=lambda **kw: None,
            MatchAny=lambda **kw: None,
            PointIdsList=lambda **kw: None,
        )
        sys.modules["qdrant_client"] = mod
    if "qdrant_client.models" not in sys.modules:
        sys.modules["qdrant_client.models"] = sys.modules["qdrant_client"].models
    if "qdrant_client.http" not in sys.modules:
        sys.modules["qdrant_client.http"] = _types.ModuleType("qdrant_client.http")
    if "qdrant_client.http.exceptions" not in sys.modules:
        exc_mod = _types.ModuleType("qdrant_client.http.exceptions")

        class UnexpectedResponse(Exception):
            def __init__(self, status_code=500, reason_phrase="", content=b"", headers=None):
                super().__init__(f"HTTP {status_code}")
                self.status_code = status_code
                self.reason_phrase = reason_phrase
                self.content = content
                self.headers = headers or {}
        exc_mod.UnexpectedResponse = UnexpectedResponse
        sys.modules["qdrant_client.http.exceptions"] = exc_mod

    # fastembed (used by _get_sparse_encoder — lazy)
    if "fastembed" not in sys.modules:
        fe = _types.ModuleType("fastembed")
        class _SparseTE:
            def __init__(self, *a, **kw): pass
            def embed(self, texts):
                return [_types.SimpleNamespace(indices=_Arr([0]), values=_Arr([1.0])) for _ in texts]
            def query_embed(self, text):
                yield _types.SimpleNamespace(indices=_Arr([0]), values=_Arr([1.0]))
        fe.SparseTextEmbedding = _SparseTE
        sys.modules["fastembed"] = fe


class _Arr(list):
    """Tiny list subclass that pretends to be a numpy array for .tolist()."""
    def tolist(self):
        return list(self)


def _make_kb_with_mock_client(mock_client, max_retries: int = 3, batch_size: int = 2):
    """Build a QdrantKB without running real __init__.

    Returns (kb, qdrant_kb_module) or raises ImportError in sandbox
    environments where pydantic_settings is not installed.
    """
    _install_qdrant_kb_stubs()

    # Patch settings BEFORE importing qdrant_kb so the module-level
    # settings reference picks up our values.
    from cmap_agent.config.settings import settings
    settings.QDRANT_UPSERT_MAX_RETRIES = max_retries
    settings.QDRANT_UPSERT_BATCH_SIZE = batch_size

    from cmap_agent.rag import qdrant_kb as qk

    class _TestKB(qk.QdrantKB):
        def __init__(self):
            # Bypass real __init__ — no network, no collection create
            self.collection_name = "test_collection"
            self.client = mock_client
            self.url = "http://test"
            self.api_key = None

        def ensure_collection(self):
            pass  # pretend it exists

    # Stub get_embedder to avoid OpenAI
    def _fake_embed(texts):
        return [[0.0] * 4 for _ in texts]

    class _FakeEmbdr:
        def embed(self, texts):
            return _fake_embed(texts)
    qk.get_embedder = lambda: _FakeEmbdr()

    return _TestKB(), qk


# Probe whether the integration tests can actually run in this env.
try:
    import pydantic_settings  # noqa: F401
    _INTEGRATION_OK = True
except ImportError:
    _INTEGRATION_OK = False
    print("  [note] skipping integration tests — pydantic_settings not installed")
    print("  [note] these tests will run in the project dev env after `pip install -e .`")


# -----

def test_upsert_succeeds_first_try() -> None:
    """Happy path: no retries, upsert called exactly once per batch."""
    mock_client = MagicMock()
    kb, qk = _make_kb_with_mock_client(mock_client, max_retries=3, batch_size=2)
    kb.upsert(ids=["a", "b", "c"], texts=["t1", "t2", "t3"], metadatas=[{}, {}, {}])
    # 3 docs, batch 2 → 2 batches → 2 upsert calls
    _check(mock_client.upsert.call_count == 2, f"happy path: 2 batches → 2 calls (got {mock_client.upsert.call_count})")


def test_upsert_retries_on_502() -> None:
    """Client raises 502 once, then succeeds.  Should be called twice for one batch."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    mock_client = MagicMock()
    mock_client.upsert.side_effect = [
        UnexpectedResponse(status_code=502),
        None,  # success on retry
    ]
    # Patch time.sleep to avoid actually waiting.
    with patch("cmap_agent.rag.qdrant_kb.time.sleep") as mock_sleep:
        kb, qk = _make_kb_with_mock_client(mock_client, max_retries=3, batch_size=10)
        kb.upsert(ids=["a"], texts=["t"], metadatas=[{}])
    _check(mock_client.upsert.call_count == 2, f"502 → retry once (got {mock_client.upsert.call_count} calls)")
    _check(mock_sleep.call_count == 1, f"slept exactly once before retry (got {mock_sleep.call_count})")


def test_upsert_retries_on_503_and_504() -> None:
    """Cycle through 503 → 504 → success.  Three attempts total."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    mock_client = MagicMock()
    mock_client.upsert.side_effect = [
        UnexpectedResponse(status_code=503),
        UnexpectedResponse(status_code=504),
        None,
    ]
    with patch("cmap_agent.rag.qdrant_kb.time.sleep"):
        kb, qk = _make_kb_with_mock_client(mock_client, max_retries=3, batch_size=10)
        kb.upsert(ids=["a"], texts=["t"], metadatas=[{}])
    _check(mock_client.upsert.call_count == 3, f"503→504→OK ran 3 times (got {mock_client.upsert.call_count})")


def test_upsert_does_not_retry_500() -> None:
    """500 is NOT retried — re-raised immediately."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    mock_client = MagicMock()
    mock_client.upsert.side_effect = UnexpectedResponse(status_code=500)

    caught = None
    with patch("cmap_agent.rag.qdrant_kb.time.sleep"):
        kb, qk = _make_kb_with_mock_client(mock_client, max_retries=3, batch_size=10)
        try:
            kb.upsert(ids=["a"], texts=["t"], metadatas=[{}])
        except UnexpectedResponse as e:
            caught = e
    _check(caught is not None and caught.status_code == 500, "500 re-raised")
    _check(mock_client.upsert.call_count == 1, f"500 NOT retried (got {mock_client.upsert.call_count} calls)")


def test_upsert_does_not_retry_400() -> None:
    """400 is NOT retried — client-side error, retry would not help."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    mock_client = MagicMock()
    mock_client.upsert.side_effect = UnexpectedResponse(status_code=400)

    caught = None
    with patch("cmap_agent.rag.qdrant_kb.time.sleep"):
        kb, qk = _make_kb_with_mock_client(mock_client, max_retries=3, batch_size=10)
        try:
            kb.upsert(ids=["a"], texts=["t"], metadatas=[{}])
        except UnexpectedResponse as e:
            caught = e
    _check(caught is not None and caught.status_code == 400, "400 re-raised immediately")
    _check(mock_client.upsert.call_count == 1, "400 NOT retried")


def test_upsert_gives_up_after_max_retries() -> None:
    """Persistent 502: retries up to max_retries then raises."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    mock_client = MagicMock()
    mock_client.upsert.side_effect = UnexpectedResponse(status_code=502)

    caught = None
    with patch("cmap_agent.rag.qdrant_kb.time.sleep"):
        kb, qk = _make_kb_with_mock_client(mock_client, max_retries=2, batch_size=10)
        try:
            kb.upsert(ids=["a"], texts=["t"], metadatas=[{}])
        except UnexpectedResponse as e:
            caught = e
    _check(caught is not None, "persistent 502 eventually raised")
    # max_retries=2 → attempt 0 + retry 1 + retry 2 = 3 total
    _check(mock_client.upsert.call_count == 3, f"max_retries=2 → 3 total calls (got {mock_client.upsert.call_count})")


def test_upsert_retries_zero_disables_retry() -> None:
    """max_retries=0 → no retry; first 502 raises immediately."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    mock_client = MagicMock()
    mock_client.upsert.side_effect = UnexpectedResponse(status_code=502)

    caught = None
    with patch("cmap_agent.rag.qdrant_kb.time.sleep") as mock_sleep:
        kb, qk = _make_kb_with_mock_client(mock_client, max_retries=0, batch_size=10)
        try:
            kb.upsert(ids=["a"], texts=["t"], metadatas=[{}])
        except UnexpectedResponse:
            caught = True
    _check(caught is True, "retries=0 still raises")
    _check(mock_client.upsert.call_count == 1, "retries=0 → exactly 1 call")
    _check(mock_sleep.call_count == 0, "retries=0 → no sleep")


def test_upsert_respects_batch_size_setting() -> None:
    """QDRANT_UPSERT_BATCH_SIZE=3 → 7 docs = 3 batches (3, 3, 1)."""
    mock_client = MagicMock()
    kb, qk = _make_kb_with_mock_client(mock_client, max_retries=0, batch_size=3)
    kb.upsert(
        ids=[f"id{i}" for i in range(7)],
        texts=[f"t{i}" for i in range(7)],
        metadatas=[{} for _ in range(7)],
    )
    _check(mock_client.upsert.call_count == 3, f"7 docs @ batch=3 → 3 calls (got {mock_client.upsert.call_count})")


def test_upsert_retry_then_terminal_error() -> None:
    """Transient 502 first, then terminal 500: retries once then raises 500."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    mock_client = MagicMock()
    mock_client.upsert.side_effect = [
        UnexpectedResponse(status_code=502),
        UnexpectedResponse(status_code=500),
    ]
    caught = None
    with patch("cmap_agent.rag.qdrant_kb.time.sleep"):
        kb, qk = _make_kb_with_mock_client(mock_client, max_retries=3, batch_size=10)
        try:
            kb.upsert(ids=["a"], texts=["t"], metadatas=[{}])
        except UnexpectedResponse as e:
            caught = e
    _check(caught is not None and caught.status_code == 500, "500 raised after retry succeeded transiently")
    _check(mock_client.upsert.call_count == 2, "502 then 500 → 2 calls")


def main() -> int:
    print("Running v226 upsert retry tests:")
    print()
    test_predicate_502_is_transient()
    test_predicate_503_is_transient()
    test_predicate_504_is_transient()
    test_predicate_500_is_NOT_transient()
    test_predicate_400_is_NOT_transient()
    test_predicate_404_is_NOT_transient()
    test_predicate_connection_error_NOT_retried()
    print()
    test_backoff_attempt_0_around_1s()
    test_backoff_attempt_2_around_4s()
    test_backoff_attempt_5_capped_at_30s()
    test_backoff_attempt_10_still_capped()
    print()
    if _INTEGRATION_OK:
        test_upsert_succeeds_first_try()
        test_upsert_retries_on_502()
        test_upsert_retries_on_503_and_504()
        test_upsert_does_not_retry_500()
        test_upsert_does_not_retry_400()
        test_upsert_gives_up_after_max_retries()
        test_upsert_retries_zero_disables_retry()
        test_upsert_respects_batch_size_setting()
        test_upsert_retry_then_terminal_error()
    else:
        print("  [skip] 9 integration tests skipped — run in project env with deps installed")
    print()
    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
