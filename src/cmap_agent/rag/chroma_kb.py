from __future__ import annotations
from typing import Any, Iterable, List

import datetime as _dt
import json

import chromadb
from chromadb.api.models.Collection import Collection

from cmap_agent.config.settings import settings
from cmap_agent.rag.embedder import get_embedder


def _to_chroma_scalar(value: Any) -> str | int | float | bool | None:
    """Coerce arbitrary values into Chroma-acceptable scalar types.

    Chroma metadata values must be: str, int, float, bool, SparseVector, or None.
    We don't use SparseVector here. This helper makes the pipeline robust against
    pandas/numpy scalars, datetimes, and accidental list/dict values.
    """
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
    # lists/tuples/sets => compact string
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(x) for x in list(value)[:50])
    # dict => json string
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)
    # fallback
    return str(value)


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _to_chroma_scalar(v) for k, v in (meta or {}).items()}

class ChromaKB:
    def __init__(self, persist_dir: str | None = None, collection: str | None = None):
        self.persist_dir = persist_dir or settings.CMAP_AGENT_CHROMA_DIR
        self.collection_name = collection or settings.CMAP_AGENT_KB_COLLECTION
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection: Collection = self.client.get_or_create_collection(name=self.collection_name)


    @classmethod
    def from_settings(
        cls,
        *,
        persist_dir: str | None = None,
        collection: str | None = None,
    ) -> "ChromaKB":
        """Create a ChromaKB instance using app settings (with optional overrides).

        This is a small convenience used by tools so they don't need to know
        where the Chroma configuration lives.
        """
        pd = persist_dir or settings.CMAP_AGENT_CHROMA_DIR
        coll = collection or settings.CMAP_AGENT_KB_COLLECTION
        return cls(persist_dir=pd, collection=coll)

    def upsert(self, *, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]):
        if not ids:
            return
        if not (len(ids) == len(texts) == len(metadatas)):
            raise ValueError("ids/texts/metadatas length mismatch")

        chunk_size = int(getattr(settings, "CMAP_AGENT_KB_UPSERT_CHUNK", 256))
        embdr = get_embedder()

        for i in range(0, len(ids), chunk_size):
            j = i + chunk_size
            ids_b = ids[i:j]
            texts_b = texts[i:j]
            metas_b = [_sanitize_metadata(m) for m in metadatas[i:j]]
            emb_b = embdr.embed(texts_b)
            self.collection.upsert(ids=ids_b, documents=texts_b, embeddings=emb_b, metadatas=metas_b)


    def all_ids(self, batch_size: int = 5000) -> list[str]:
        """Return all document IDs in the collection.

        Chroma's Collection.get supports paging via limit/offset. IDs are always
        returned, so we can request include=[] to keep responses small.
        """
        ids: list[str] = []
        offset = 0
        while True:
            try:
                res = self.collection.get(limit=batch_size, offset=offset, include=[])
            except TypeError:
                # Older chromadb versions may not accept `include=`.
                res = self.collection.get(limit=batch_size, offset=offset)
            batch = res.get("ids") or []
            if not batch:
                break
            ids.extend(batch)
            offset += len(batch)
            if len(batch) < batch_size:
                break
        return ids

    def delete_ids(self, ids: list[str]) -> None:
        """Delete a batch of IDs from the collection."""
        if not ids:
            return
        self.collection.delete(ids=ids)

    def query(
        self,
        query_text: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over the KB.

        Args:
            query_text: Natural-language query.
            k: Number of hits.
            where: Optional Chroma metadata filter (e.g., {"doc_type": "dataset"}).
        """
        k = k or settings.CMAP_AGENT_KB_TOP_K
        qemb = get_embedder().embed([query_text])[0]
        kwargs: dict[str, Any] = {
            "query_embeddings": [qemb],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)

        out: list[dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append({"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]})
        return out
