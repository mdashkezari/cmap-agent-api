"""qdrant_kb — Qdrant-backed knowledge base with hybrid search (dense + BM25).

Replaces ChromaDB to enable hybrid retrieval: dense cosine similarity
(OpenAI embeddings) combined with BM25 sparse keyword matching via
Reciprocal Rank Fusion (RRF).  This fixes the structural limitation where
pure-dense retrieval cannot surface specific technical terms buried in
methods sections of papers (e.g. "minimum sequencing depth threshold 5000").

Qdrant supports named vector spaces per point, allowing both a dense vector
and a sparse BM25 vector to be stored and queried independently, then fused.

Connection is controlled by environment variables:
    QDRANT_URL       — Qdrant server URL (default: http://localhost:6333)
    QDRANT_API_KEY   — API key for Qdrant Cloud (empty for local Docker)
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import random
import time
import uuid as _uuid
from typing import Any, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from cmap_agent.config.settings import settings
from cmap_agent.rag.embedder import get_embedder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata sanitization (same rules as chroma_kb, Qdrant is similarly strict)
# ---------------------------------------------------------------------------

def _to_qdrant_scalar(value: Any) -> str | int | float | bool | None:
    """Coerce arbitrary values into Qdrant-acceptable payload types."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    # numpy scalars
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            v = value.item()
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
        except Exception:
            pass
    # datetimes/dates
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    # lists/tuples/sets → compact string
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(x) for x in list(value)[:50])
    # dict → json string
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)
    return str(value)


def _sanitize_payload(meta: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _to_qdrant_scalar(v) for k, v in (meta or {}).items()}


# ---------------------------------------------------------------------------
# BM25 sparse encoding via fastembed
# ---------------------------------------------------------------------------

_sparse_encoder = None


def _get_sparse_encoder():
    """Lazy-load the fastembed BM25 sparse encoder.

    The ``Qdrant/bm25`` model is a lightweight tokenizer (~1 MB), not a
    neural network.  It generates sparse token-weight vectors compatible
    with Qdrant's ``Modifier.IDF`` sparse vector configuration.
    """
    global _sparse_encoder
    if _sparse_encoder is None:
        from fastembed import SparseTextEmbedding
        _sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_encoder


def _sparse_embed_batch(texts: Sequence[str]) -> list[models.SparseVector]:
    """Generate BM25 sparse vectors for a batch of texts."""
    encoder = _get_sparse_encoder()
    sparse_list = list(encoder.embed(list(texts)))
    return [
        models.SparseVector(
            indices=sv.indices.tolist(),
            values=sv.values.tolist(),
        )
        for sv in sparse_list
    ]


def _sparse_embed_query(text: str) -> models.SparseVector:
    """Generate a BM25 sparse query vector (uses query_embed for IDF weighting)."""
    encoder = _get_sparse_encoder()
    sv = next(encoder.query_embed(text))
    return models.SparseVector(
        indices=sv.indices.tolist(),
        values=sv.values.tolist(),
    )


# ---------------------------------------------------------------------------
# Transient-error retry helper (v226)
# ---------------------------------------------------------------------------
# Qdrant Cloud's managed ingress can return 502/503/504 during bulk writes
# even when the cluster itself is idle and under-utilised (confirmed by
# dashboard metrics during the v225 Phase C cutover: RAM <20%, CPU <10%,
# disk <1% at the time of the 502).  Upstream issue qdrant/qdrant#3263
# documents the same pattern on self-hosted deployments behind nginx
# ingress.  These errors are not caused by the client's request content
# or by cluster capacity — retrying with backoff after a short pause
# almost always succeeds.
#
# This helper is conservative: it retries ONLY on the three proxy-layer
# 5xx codes and ONLY on the upsert-point path.  It does not retry on 4xx
# (bad request), on cluster-internal 500 (real error), or on network
# exceptions other than UnexpectedResponse.  Caller supplies max attempts
# via ``max_attempts``; 0 disables retry entirely and re-raises the first
# failure.


_RETRYABLE_STATUS = frozenset({502, 503, 504})
_RETRY_BACKOFF_CAP_S = 30.0
_RETRY_BACKOFF_JITTER_S = 0.5


def _is_transient_proxy_error(exc: BaseException) -> bool:
    """Return True when ``exc`` is a 502/503/504 from the Qdrant HTTP API."""
    if not isinstance(exc, UnexpectedResponse):
        return False
    code = getattr(exc, "status_code", None)
    return code in _RETRYABLE_STATUS


def _retry_backoff_seconds(attempt: int) -> float:
    """Exponential backoff with small jitter: 1s, 2s, 4s, 8s, 16s, capped at 30s."""
    base = min(2 ** attempt, _RETRY_BACKOFF_CAP_S)
    return base + random.uniform(0, _RETRY_BACKOFF_JITTER_S)


# ---------------------------------------------------------------------------
# Filter construction — Chroma-style operators mapped to Qdrant conditions
# ---------------------------------------------------------------------------

# The KB protocol (shared with ChromaKB) accepts a Chroma-style ``where``
# dict: ``{field: value}`` for equality, ``{field: {"$in": [a, b, c]}}``
# for inclusion, and ``{field: {"$eq": value}}`` as a verbose equality
# form.  Chroma's native query API understands all of these.  Qdrant's
# native filter syntax is different, so this helper performs the
# translation.
#
# Prior to v217, this translation was missing: every ``where`` value was
# wrapped in ``MatchValue(value=...)``, which silently misinterpreted
# ``{"$in": [...]}`` as a literal dict value and returned nothing.
# Upstream code in ``catalog_tools`` that sends ``$in`` filters was
# therefore being quietly broken on the Qdrant backend.  This helper
# restores parity.


def _build_field_condition(key: str, value: Any) -> "models.FieldCondition":
    """Translate a Chroma-style where clause entry into a Qdrant condition.

    Accepted shapes:
      - scalar (``str | int | float | bool``) -> ``MatchValue``
      - ``{"$eq": scalar}``                   -> ``MatchValue``
      - ``{"$in": [v1, v2, ...]}``            -> ``MatchAny``

    Any other shape is treated as a literal value via ``MatchValue`` and
    a warning is logged.  Callers should not rely on the fallback — it
    exists only so that a stray unknown operator does not crash the
    whole query path; it will still usually return empty results.
    """
    if isinstance(value, dict):
        if "$in" in value:
            values = value["$in"]
            if not isinstance(values, (list, tuple)) or not values:
                # Empty $in list: match nothing.  Qdrant has no explicit
                # "match nothing" condition; an empty MatchAny raises.
                # Use MatchAny with a sentinel that won't be present.
                return models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=["__cmap_empty_in_sentinel__"]),
                )
            return models.FieldCondition(
                key=key,
                match=models.MatchAny(any=list(values)),
            )
        if "$eq" in value:
            return models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value["$eq"]),
            )
        log.warning(
            "Unknown filter operator in where[%r]=%r; "
            "falling back to MatchValue (likely returns nothing).",
            key, value,
        )
        return models.FieldCondition(
            key=key,
            match=models.MatchValue(value=value),
        )
    return models.FieldCondition(
        key=key,
        match=models.MatchValue(value=value),
    )


# ---------------------------------------------------------------------------
# QdrantKB
# ---------------------------------------------------------------------------

class QdrantKB:
    """Knowledge base backed by Qdrant with hybrid dense + BM25 search."""

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
    ):
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection or settings.CMAP_AGENT_KB_COLLECTION

        # Support in-memory mode for testing: QdrantClient(location=":memory:")
        # uses a local embedded instance and does not accept url= or timeout=.
        if self.url in (":memory:", "memory"):
            self.client = QdrantClient(location=":memory:")
        else:
            client_kwargs: dict[str, Any] = {"url": self.url, "timeout": 120}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            self.client = QdrantClient(**client_kwargs)

    @classmethod
    def in_memory(cls, collection: str | None = None) -> "QdrantKB":
        """Create an in-memory QdrantKB instance for testing."""
        return cls(url=":memory:", collection=collection)

    @classmethod
    def from_settings(
        cls,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
    ) -> "QdrantKB":
        """Create a QdrantKB instance using app settings (with optional overrides)."""
        return cls(
            url=url or settings.QDRANT_URL,
            api_key=api_key or settings.QDRANT_API_KEY,
            collection=collection or settings.CMAP_AGENT_KB_COLLECTION,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, dense_dim: int | None = None) -> None:
        """Create the collection if it does not already exist.

        Sets up named vector spaces:
          - ``dense``: cosine similarity, dimension from settings or argument
          - ``bm25``:  sparse BM25 with IDF modifier
        """
        dim = dense_dim or settings.QDRANT_DENSE_DIM
        try:
            self.client.get_collection(self.collection_name)
            log.debug("Collection '%s' already exists.", self.collection_name)
            return
        except (UnexpectedResponse, KeyError, ValueError, Exception):
            # Collection does not exist — fall through to create it.
            pass

        log.info("Creating collection '%s' (dense=%d dims).", self.collection_name, dim)
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
        except UnexpectedResponse as exc:
            # The most common 500 on create_collection after a host-side
            # ``rm -rf qdrant_data`` is "File exists (os error 17)": the
            # server has a stray on-disk directory for the collection but
            # the metadata says it doesn't exist, so ``get_collection``
            # reports missing yet ``create_collection`` refuses.  Surface a
            # clear remediation path instead of an opaque stack trace.
            body = str(getattr(exc, "content", b"") or "") + " " + str(exc)
            if "File exists" in body or "already exists" in body:
                raise RuntimeError(
                    f"Qdrant refused to create collection "
                    f"'{self.collection_name}' because a stray storage "
                    f"directory already exists on the server.  Recovery: "
                    f"ask Qdrant itself to reconcile "
                    f"(curl -X DELETE {self.url}/collections/"
                    f"{self.collection_name}) and retry.  If that still "
                    f"fails, stop the Qdrant container, wipe its storage "
                    f"mount, and restart it."
                ) from exc
            raise

    def delete_collection(self) -> None:
        """Delete the entire collection (used for --rebuild)."""
        try:
            self.client.delete_collection(self.collection_name)
            log.info("Deleted collection '%s'.", self.collection_name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(
        self,
        *,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert documents with both dense and sparse vectors.

        Dense embeddings are generated via the OpenAI embedding model.
        Sparse BM25 vectors are generated via fastembed.

        v226: retry with exponential backoff on 502/503/504 proxy-layer
        errors.  See ``_is_transient_proxy_error`` and
        ``QDRANT_UPSERT_MAX_RETRIES`` in settings.  Batch size is now
        configurable via ``QDRANT_UPSERT_BATCH_SIZE`` (default 64).
        """
        if not ids:
            return
        if not (len(ids) == len(texts) == len(metadatas)):
            raise ValueError("ids/texts/metadatas length mismatch")

        self.ensure_collection()
        # Batch size is configurable (QDRANT_UPSERT_BATCH_SIZE).  Smaller
        # batches reduce per-request payload and are less likely to hit
        # proxy-layer size/time limits on managed Qdrant tiers.
        chunk_size = max(1, int(settings.QDRANT_UPSERT_BATCH_SIZE))
        max_retries = max(0, int(settings.QDRANT_UPSERT_MAX_RETRIES))

        embdr = get_embedder()

        for i in range(0, len(ids), chunk_size):
            j = i + chunk_size
            ids_b = ids[i:j]
            texts_b = texts[i:j]
            metas_b = [_sanitize_payload(m) for m in metadatas[i:j]]

            # Dense embeddings (OpenAI)
            dense_b = embdr.embed(texts_b)
            # Sparse BM25 embeddings (fastembed)
            sparse_b = _sparse_embed_batch(texts_b)

            points = []
            for k_idx in range(len(ids_b)):
                payload = dict(metas_b[k_idx])
                payload["_text"] = texts_b[k_idx]  # store document text in payload
                payload["_doc_id"] = ids_b[k_idx]   # store original doc id

                points.append(
                    models.PointStruct(
                        id=k_idx + i,   # temporary; will use named IDs below
                        vector={
                            "dense": dense_b[k_idx],
                            "bm25": sparse_b[k_idx],
                        },
                        payload=payload,
                    )
                )

            # Use doc IDs as point IDs via the string-hash approach.
            # Qdrant supports unsigned 64-bit int IDs or UUID strings.
            # Map string doc IDs to deterministic UUIDs for stable upserts.
            for k_idx, pt in enumerate(points):
                pt.id = str(_uuid.uuid5(_uuid.NAMESPACE_URL, ids_b[k_idx]))

            # Retry-on-transient-proxy-error.  See module-level
            # ``_is_transient_proxy_error`` for the classification rule.
            # Attempt 0 is the initial try; attempts 1..max_retries are
            # retries with exponential backoff (1s, 2s, 4s, ...).
            last_exc: BaseException | None = None
            for attempt in range(max_retries + 1):
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                    last_exc = None
                    break
                except UnexpectedResponse as exc:
                    last_exc = exc
                    if attempt >= max_retries or not _is_transient_proxy_error(exc):
                        raise
                    sleep_s = _retry_backoff_seconds(attempt)
                    code = getattr(exc, "status_code", "???")
                    log.warning(
                        "Qdrant upsert got HTTP %s on batch %d-%d "
                        "(attempt %d/%d); retrying in %.1fs",
                        code, i, min(j, len(ids)), attempt + 1,
                        max_retries, sleep_s,
                    )
                    time.sleep(sleep_s)
            if last_exc is not None:
                # Defensive: should already have raised inside the loop.
                raise last_exc

        log.info("Upserted %d documents into '%s'.", len(ids), self.collection_name)

    # ------------------------------------------------------------------
    # Query — hybrid dense + BM25 with RRF fusion
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search: dense + BM25 fused via Reciprocal Rank Fusion.

        Args:
            query_text: Natural-language query.
            k: Number of hits to return.
            where: Optional Qdrant payload filter.

        Returns:
            List of hit dicts with keys: id, text, metadata, distance (score).
        """
        k = k or settings.CMAP_AGENT_KB_TOP_K
        prefetch_k = k * max(1, int(settings.QDRANT_PREFETCH_FACTOR))

        # Dense query embedding
        dense_qemb = get_embedder().embed([query_text])[0]
        # Sparse BM25 query embedding
        sparse_qemb = _sparse_embed_query(query_text)

        # Build optional filter
        qfilter = None
        if where:
            must_conditions = []
            for wk, wv in where.items():
                must_conditions.append(_build_field_condition(wk, wv))
            qfilter = models.Filter(must=must_conditions)

        # Fusion method selection.  See settings.QDRANT_FUSION docs for
        # when to prefer each; RRF is the safe default.  The choice is
        # query-time only; changing it does not require a rebuild.
        fusion_name = (settings.QDRANT_FUSION or "rrf").lower().strip()
        if fusion_name == "dbsf":
            fusion = models.Fusion.DBSF
        elif fusion_name == "rrf":
            fusion = models.Fusion.RRF
        else:
            log.warning(
                "Unknown QDRANT_FUSION=%r; falling back to RRF. "
                "Valid values: 'rrf', 'dbsf'.",
                settings.QDRANT_FUSION,
            )
            fusion = models.Fusion.RRF

        # Hybrid query with configured fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_qemb,
                    using="dense",
                    limit=prefetch_k,
                    filter=qfilter,
                ),
                models.Prefetch(
                    query=sparse_qemb,
                    using="bm25",
                    limit=prefetch_k,
                    filter=qfilter,
                ),
            ],
            query=models.FusionQuery(fusion=fusion),
            limit=k,
            with_payload=True,
        )

        out: list[dict[str, Any]] = []
        for pt in results.points:
            payload = pt.payload or {}
            # Extract stored text and doc_id; remaining payload is metadata
            text = payload.pop("_text", "")
            doc_id = payload.pop("_doc_id", str(pt.id))
            # Normalize score -> distance (lower = better) for parity with
            # ChromaKB.  Qdrant returns a higher-is-better fusion score; we
            # map it to a distance so that downstream consumers (notably
            # catalog_tools._kb_semantic_table_scores) can apply a single
            # distance-based transform like 1/(1+d) consistently across
            # backends.  Prior to v217, consumers were inverting the
            # meaning of Qdrant scores: a higher-ranked hit produced a
            # smaller s value because the raw pt.score was being fed into
            # 1/(1+d) as though it were a distance.  See handoff for
            # detail.
            score = pt.score if pt.score is not None else 0.0
            out.append({
                "id": doc_id,
                "text": text,
                "metadata": payload,
                "distance": 1.0 - float(score),
            })

        return out

    # ------------------------------------------------------------------
    # ID management
    # ------------------------------------------------------------------

    def all_ids(self, batch_size: int = 256) -> list[str]:
        """Return all document IDs in the collection."""
        ids: list[str] = []
        offset = None
        while True:
            scroll_kwargs: dict[str, Any] = {
                "collection_name": self.collection_name,
                "limit": batch_size,
                "with_payload": ["_doc_id"],
            }
            if offset is not None:
                scroll_kwargs["offset"] = offset

            points, next_offset = self.client.scroll(**scroll_kwargs)
            for pt in points:
                doc_id = (pt.payload or {}).get("_doc_id", str(pt.id))
                ids.append(doc_id)

            if next_offset is None or not points:
                break
            offset = next_offset

        return ids

    def delete_ids(self, ids: list[str]) -> None:
        """Delete documents by their string doc IDs."""
        if not ids:
            return
        point_ids = [
            str(_uuid.uuid5(_uuid.NAMESPACE_URL, doc_id))
            for doc_id in ids
        ]
        # Qdrant delete supports up to ~1000 IDs at a time
        for i in range(0, len(point_ids), 500):
            batch = point_ids[i:i + 500]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=batch),
            )
