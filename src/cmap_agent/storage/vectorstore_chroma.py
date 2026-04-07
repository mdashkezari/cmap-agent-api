from __future__ import annotations

from typing import Any

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:  # pragma: no cover
    chromadb = None

from cmap_agent.storage.vectorstore_base import VectorStore

class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str, collection: str = "cmap_agent"):
        if chromadb is None:
            raise RuntimeError("chromadb is not installed. Install with: pip install -e '.[retrieval]'")
        client = chromadb.Client(ChromaSettings(persist_directory=persist_dir, anonymized_telemetry=False))
        self.col = client.get_or_create_collection(collection)

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]], embeddings: list[list[float]] | None = None) -> None:
        self.col.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, embedding: list[float], top_k: int = 10, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        res = self.col.query(query_embeddings=[embedding], n_results=top_k, where=where)
        out = []
        for i in range(len(res.get("ids", [[]])[0])):
            out.append({
                "id": res["ids"][0][i],
                "score": res.get("distances", [[]])[0][i] if res.get("distances") else None,
                "document": res.get("documents", [[]])[0][i] if res.get("documents") else None,
                "metadata": res.get("metadatas", [[]])[0][i] if res.get("metadatas") else None,
            })
        return out
