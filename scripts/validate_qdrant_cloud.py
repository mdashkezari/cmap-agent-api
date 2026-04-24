"""validate_qdrant_cloud.py — pre-cutover validation for Phase C.

Runs four checks against a Qdrant Cloud cluster before pointing
ECS at it:

1. **Connect**: the cluster is reachable with the provided API key.
2. **Collection**: the expected collection exists.
3. **Count**: the collection is populated (floor-only sanity check —
   see the DOC_COUNT_FLOOR constant).
4. **Smoke test**: a known-answer retrieval query returns a hit
   whose chunk text contains the expected substring.

Any failure exits non-zero and prints a clear diagnostic.  This
script is intended to be run as a go/no-go gate per the
``docs/phase-c-cutover-runbook.md`` Step 6.

Usage::

    python scripts/validate_qdrant_cloud.py \\
        --url "https://<cluster>.us-west-2.aws.cloud.qdrant.io:6333" \\
        --api-key "<api-key>" \\
        --collection cmap_kb_v1

Environment fallback: if ``--url`` / ``--api-key`` are omitted the
script reads ``QDRANT_URL`` and ``QDRANT_API_KEY`` from the
environment.  ``--collection`` defaults to ``cmap_kb_v1``.

Optional skip flags:
    --skip-count  — skip the count check (useful when the KB is mid-sync
                    or has been intentionally shrunk)
    --skip-smoke  — skip the known-answer smoke test (useful when the
                    OpenAI embedding key is unavailable)

The smoke-test query/answer pair and the count floor are defined
near the top of this file as constants — adjust them when the KB's
expected shape changes (rare).

Exit codes
----------
0 — all enabled checks pass
1 — a check failed (see stderr for diagnostic)
2 — usage/argument error
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# -------------------------------------------------------------------
# Expected-shape constants (v226 update).
# -------------------------------------------------------------------
#
# The count check is a floor-only sanity check — "has the KB been
# populated at all?" — rather than a tight band around a hardcoded
# target.  The previous ±5% band around 36,919 (the session 202
# snapshot) became stale as the reference bank grew; the v225 cutover
# produced 41,697 points and the old check failed even though the KB
# was healthy.  A floor-only check avoids this rot while still
# catching the real failure cases (cluster is empty, or the sync
# never finished its first wave).
#
# If the actual count drops BELOW the floor the check still fails —
# that is the genuine "something's wrong" signal.  Adjust the floor
# downward only if the KB legitimately shrinks (e.g. reference bank
# pruned, catalog restructured).

DOC_COUNT_FLOOR = 30_000  # minimum healthy size for a populated KB

# Known-answer smoke test.  The question about GRUMP's minimum
# sequencing depth threshold has a documented correct answer of
# "5000" — see the session 202 / v199 history in the handoff.  The
# answer text should appear in the top-K retrieved chunks.
SMOKE_QUERY = (
    "minimum sequencing depth threshold used to filter GRUMP samples"
)
SMOKE_EXPECTED_SUBSTRING = "5000"
# The smoke top-K matches the production CMAP_AGENT_KB_TOP_K default
# (32 as of v228).  Validating at the same window production uses
# avoids the class of false-positive where the validator passes at
# a smaller window than production queries at.
SMOKE_TOP_K = 32


# -------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _pass(label: str, detail: str = "") -> None:
    msg = f"  [PASS] {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)


def _fail(label: str, detail: str = "") -> None:
    msg = f"  [FAIL] {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg, file=sys.stderr)


def _check_connect(url: str, api_key: str) -> tuple[bool, object]:
    """Returns (ok, client_or_error_msg)."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        return False, f"qdrant-client not installed: {e}"
    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=30.0)
        # Trigger a lightweight call to force authentication.
        client.get_collections()
        return True, client
    except Exception as e:
        return False, f"connect error: {e}"


def _check_collection_exists(
    client: object, collection: str,
) -> tuple[bool, str]:
    try:
        cols = client.get_collections().collections  # type: ignore[attr-defined]
        names = [c.name for c in cols]
        if collection in names:
            return True, f"found ({len(names)} collection(s) visible)"
        return False, f"collection {collection!r} not in: {names}"
    except Exception as e:
        return False, f"list error: {e}"


def _check_doc_count(
    client: object, collection: str,
) -> tuple[bool, str]:
    try:
        info = client.get_collection(collection_name=collection)  # type: ignore[attr-defined]
        # Qdrant client API: points_count on CollectionInfo
        actual = getattr(info, "points_count", None)
        if actual is None:
            # Fall back to a count call
            c = client.count(collection_name=collection)  # type: ignore[attr-defined]
            actual = getattr(c, "count", None)
        if actual is None:
            return False, "could not determine point count from API"
        detail = (
            f"actual={actual:,}  "
            f"floor={DOC_COUNT_FLOOR:,} "
            f"(floor-only sanity check — see validate_qdrant_cloud.py header)"
        )
        if actual >= DOC_COUNT_FLOOR:
            return True, detail
        return False, detail
    except Exception as e:
        return False, f"count error: {e}"


def _check_smoke_query(collection: str) -> tuple[bool, str]:
    """Run the known-answer retrieval and check the top chunks for
    the expected substring.

    This uses the project's own ``QdrantKB`` class rather than
    calling the raw Qdrant client, because the production code path
    applies embedding + hybrid fusion that a raw client call would
    miss.  The env vars QDRANT_URL / QDRANT_API_KEY must be set at
    this point — which the argparse handler below ensures.
    """
    try:
        os.environ["CMAP_AGENT_KB_COLLECTION"] = collection
        from cmap_agent.rag.qdrant_kb import QdrantKB
    except Exception as e:
        return False, f"QdrantKB import failed: {e}"
    try:
        kb = QdrantKB()
        hits = kb.query(SMOKE_QUERY, k=SMOKE_TOP_K)
    except Exception as e:
        return False, f"query error: {e}"
    if not hits:
        return False, "zero hits returned"
    for i, h in enumerate(hits, start=1):
        text = str(h.get("text") or "")
        if SMOKE_EXPECTED_SUBSTRING in text:
            return True, (
                f"found {SMOKE_EXPECTED_SUBSTRING!r} at rank {i} "
                f"of {len(hits)}"
            )
    # Not found — give a helpful dump of top-3 snippets
    snippets = []
    for i, h in enumerate(hits[:3], start=1):
        txt = str(h.get("text") or "")[:120].replace("\n", " ")
        snippets.append(f"#{i}: {txt}...")
    return False, (
        f"substring {SMOKE_EXPECTED_SUBSTRING!r} not in top-{SMOKE_TOP_K}.  "
        f"Top-3 previews: " + " | ".join(snippets)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url",
                    default=os.environ.get("QDRANT_URL", ""),
                    help="Qdrant Cloud URL (default: $QDRANT_URL)")
    ap.add_argument("--api-key",
                    default=os.environ.get("QDRANT_API_KEY", ""),
                    help="Qdrant Cloud API key (default: $QDRANT_API_KEY)")
    ap.add_argument("--collection",
                    default="cmap_kb_v1",
                    help="Collection name (default: cmap_kb_v1)")
    ap.add_argument("--skip-smoke",
                    action="store_true",
                    help="Skip the known-answer smoke test "
                         "(useful when OpenAI key is unavailable)")
    ap.add_argument("--skip-count",
                    action="store_true",
                    help="Skip the point-count sanity check "
                         "(useful when the KB has been intentionally "
                         "shrunk or is mid-sync)")
    args = ap.parse_args()

    if not args.url or not args.api_key:
        print(
            "ERROR: both --url and --api-key are required "
            "(or QDRANT_URL / QDRANT_API_KEY must be set)",
            file=sys.stderr,
        )
        return 2

    # Thread the URL/API key through to the QdrantKB import path too.
    os.environ["QDRANT_URL"] = args.url
    os.environ["QDRANT_API_KEY"] = args.api_key

    print(f"Validating Qdrant Cloud: {args.url}")
    print(f"  Collection: {args.collection}")
    print()

    # 1. Connect
    ok1, result = _check_connect(args.url, args.api_key)
    if ok1:
        _pass("connect")
        client = result
    else:
        _fail("connect", str(result))
        return 1

    # 2. Collection
    ok2, detail2 = _check_collection_exists(client, args.collection)
    if ok2:
        _pass("collection", detail2)
    else:
        _fail("collection", detail2)
        return 1

    # 3. Doc count (optional skip)
    if args.skip_count:
        print("  [skip] count — --skip-count flag was set")
    else:
        ok3, detail3 = _check_doc_count(client, args.collection)
        if ok3:
            _pass("count", detail3)
        else:
            _fail("count", detail3)
            return 1

    # 4. Smoke test (optional skip)
    if args.skip_smoke:
        print("  [skip] smoke — --skip-smoke flag was set")
    else:
        ok4, detail4 = _check_smoke_query(args.collection)
        if ok4:
            _pass("smoke", detail4)
        else:
            _fail("smoke", detail4)
            return 1

    print()
    print("All checks passed.  Cluster is ready for ECS cutover.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
