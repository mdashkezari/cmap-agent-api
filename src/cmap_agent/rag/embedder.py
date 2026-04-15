from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

from openai import OpenAI

from ..config.settings import settings


def _estimate_tokens(text: str) -> int:
    """Very conservative estimate of token count.

    We use UTF-8 byte length as a safe upper bound on token count.
    Each token encodes at least one byte, so: tokens <= len(utf8_bytes).

    This makes batching and truncation robust even for extremely "token dense"
    strings (lots of punctuation, separators, or short identifiers) that can
    otherwise exceed 8k tokens with only ~15k characters.
    """
    if not text:
        return 0
    return len(text.encode("utf-8"))


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate a string to at most max_bytes (UTF-8), keeping valid UTF-8."""
    if max_bytes <= 0:
        return ""
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    return b[:max_bytes].decode("utf-8", errors="ignore")


@dataclass
class OpenAIEmbedder:
    client: OpenAI
    model: str

    # Conservative limits to avoid "max context length" errors.
    # Many embedding models enforce a ~8192 token max for the *entire request*.
    # Keep both per-item and per-request totals comfortably below that.
    max_item_tokens: int = 7200
    max_batch_tokens: int = 7200
    max_batch_items: int = 96           # also cap number of strings

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)

        out_embs: List[List[float]] = []
        batch: List[str] = []
        batch_tok = 0

        def flush() -> None:
            nonlocal batch, batch_tok, out_embs
            if not batch:
                return
            resp = self.client.embeddings.create(model=self.model, input=batch)
            # The API returns embeddings aligned with inputs
            out_embs.extend([d.embedding for d in resp.data])
            batch = []
            batch_tok = 0

        def _sanitize(s: str) -> str:
            """Remove characters that break JSON serialization."""
            # Remove null bytes and other control chars that invalidate JSON
            import re
            s = s.replace("\x00", "")
            # Replace other problematic control characters (except \n \t \r)
            s = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
            # Ensure valid UTF-8 by encode/decode round-trip
            s = s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            return s.strip() or " "

        for t in texts:
            if t is None:
                t = ""
            if not isinstance(t, str):
                t = str(t)
            t = _sanitize(t)

            tok = _estimate_tokens(t)

            # If an item is too large, truncate it in UTF-8 bytes.
            # Using bytes as an upper bound on tokens keeps us under embedding
            # model context limits even for very token-dense text.
            if tok > self.max_item_tokens:
                t = _truncate_utf8(t, self.max_item_tokens)
                tok = _estimate_tokens(t)

            # If adding this item would exceed batch limits, flush first.
            if batch and (
                batch_tok + tok > self.max_batch_tokens
                or len(batch) >= self.max_batch_items
            ):
                flush()

            batch.append(t)
            batch_tok += tok

        flush()

        # Sanity: preserve 1:1 alignment
        if len(out_embs) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(out_embs)} embeddings for {len(texts)} texts"
            )

        return out_embs


_embedder_singleton: OpenAIEmbedder | None = None


def _make_openai_client() -> OpenAI:
    """Create an OpenAI client using config + env.

    - Prefer settings.OPENAI_API_KEY (loaded from .env / environment by pydantic)
    - Fall back to OPENAI_API_KEY env var
    - Optionally honor settings.OPENAI_BASE_URL
    """
    # Settings may not define all optional fields; use getattr for compatibility.
    api_key = getattr(settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
    base_url = getattr(settings, "OPENAI_BASE_URL", None) or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        # Raise an explicit, actionable error rather than letting the OpenAI client throw.
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "Set it in your shell (export OPENAI_API_KEY=...) or put it in a .env file, e.g.:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "  CMAP_EMBEDDINGS_MODEL=text-embedding-3-small\n"
        )

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_embedder(model: str | None = None) -> OpenAIEmbedder:
    global _embedder_singleton
    desired_model = model or settings.CMAP_EMBEDDINGS_MODEL or "text-embedding-3-small"
    if _embedder_singleton is None:
        _embedder_singleton = OpenAIEmbedder(client=_make_openai_client(), model=desired_model)
    else:
        _embedder_singleton.model = desired_model
    return _embedder_singleton
