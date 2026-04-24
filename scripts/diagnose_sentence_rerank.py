#!/usr/bin/env python3
"""diagnose_sentence_rerank.py — would sentence re-ranking help?

Non-destructive diagnostic. Measures, for each harness query, what rank
the best answer-containing sentence would achieve if the retrieval
pipeline scored at sentence granularity instead of chunk granularity.

Hypothesis being tested
-----------------------
For queries that currently miss (target substring at chunk rank > TOP_K),
one of two things is true:

  A. The chunk containing the answer is in the initial retrieval window
     (SCAN_K) but ranks below TOP_K because its OVERALL embedding points
     at a different topic than the answer sentence itself. In this case,
     re-ranking by sentence-level similarity should promote the correct
     sentence near the top across-all-chunks, and sentence re-ranking
     would rescue the query.

  B. The chunk containing the answer is nowhere in the initial
     retrieval window. No amount of within-chunk re-ranking helps;
     the chunk was never fetched. In this case, sentence re-ranking
     is not the right intervention.

The output identifies which case applies to each close-miss query,
telling us whether implementing sentence re-ranking is worth the effort.

For each query the script reports:

  chunk_rank        Rank of first chunk containing an any_of substring
                    (from Qdrant hybrid retrieval). `None` if not in SCAN_K.
  sent_rank         Rank of best-matching sentence across all sentences
                    of all retrieved chunks (combined sentence pool,
                    re-ranked by query-sentence cosine similarity).
                    `None` if no sentence contains an any_of substring.
  best_sent_in_top  Whether the answer sentence would be in the top-N
                    sentences (configurable N). This is the proxy for
                    "would the LLM see the answer if the prompt were
                    filled with top-N sentences instead of top-K chunks."

Usage
-----
    python scripts/diagnose_sentence_rerank.py
    python scripts/diagnose_sentence_rerank.py --scan-k 20 --top-n-sents 24
    python scripts/diagnose_sentence_rerank.py --only P101_M,P104_M
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "scripts"))


DEFAULT_QUERIES = _ROOT / "scripts" / "retrieval_eval_queries.json"


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Naive but adequate sentence splitter.  PDF text is messy and full of
# abbreviations; we don't need perfect sentence segmentation — we need
# spans that are small enough for dense embedding to distinguish one
# topic from another within a chunk.  A ~1-3 line span is about right.
_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])"  # terminator + whitespace + new-sentence start
)


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentence-like spans.

    Collapses whitespace runs within each span but preserves intra-span
    punctuation.  Empty spans and trivially short spans (< 8 chars) are
    dropped because they rarely carry answer content and their embeddings
    are noisy.
    """
    if not text:
        return []
    # Normalize newlines + collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 8:
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Cosine + scoring
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equally-dimensioned float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _best_matching_sentence_rank(
    sentences: list[str],
    sentence_scores: list[float],
    any_of_norm: list[str],
) -> int | None:
    """Rank (1-based) of the best-scoring sentence that contains any target.

    Sentences are ranked by *sentence_scores* (higher = better match to
    query). We return the rank of the top-scoring sentence whose text
    contains any normalized any_of substring.
    """
    from eval_text_match import normalize_for_match

    # Sort sentence indices by score descending.
    order = sorted(range(len(sentences)), key=lambda i: -sentence_scores[i])
    for pos, i in enumerate(order, start=1):
        text = normalize_for_match(sentences[i])
        if any(n and n in text for n in any_of_norm):
            return pos
    return None


def _chunk_rank_of_target(
    chunks: list[dict],
    any_of_norm: list[str],
) -> int | None:
    """Rank of first chunk (1-based) whose text contains any any_of substring."""
    from eval_text_match import normalize_for_match
    for i, ch in enumerate(chunks, start=1):
        text = normalize_for_match(ch.get("text") or "")
        if any(n and n in text for n in any_of_norm):
            return i
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
    )
    ap.add_argument(
        "--scan-k",
        type=int,
        default=20,
        help="Number of chunks to retrieve from Qdrant per query. "
             "Smaller is more conservative (simulates tighter real-time "
             "budgets). Default: 20.",
    )
    ap.add_argument(
        "--top-n-sents",
        type=int,
        default=24,
        help="Sentence-level top-N threshold for the 'would it be visible' "
             "metric. Default: 24 (~ equivalent to a TOP_K=12 chunk window "
             "at 2 sentences per chunk average).",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of entry IDs to run (debugging).",
    )
    args = ap.parse_args()

    # Late imports so `-h` works without OpenAI keys.
    import eval_retrieval as ev
    from cmap_agent.rag.qdrant_kb import QdrantKB
    from cmap_agent.rag.embedder import get_embedder
    from cmap_agent.config.settings import settings
    from eval_text_match import normalize_for_match

    queries = ev._load_queries(args.queries)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        queries = [q for q in queries if q["id"] in wanted]
    if not queries:
        print("No queries to score.")
        return 0

    print(f"[setup] {len(queries)} queries  SCAN_K={args.scan_k}  "
          f"top-N sentences={args.top_n_sents}")

    kb = QdrantKB(collection=settings.CMAP_AGENT_KB_COLLECTION)
    embedder = get_embedder()

    n_total = 0
    n_chunk_hit_top12 = 0
    n_sent_hit_topN = 0
    n_chunk_miss_sent_rescue = 0  # interesting: would sentence rerank help?
    n_chunk_hit_sent_miss = 0     # anti-interesting: would sentence rerank hurt?
    rows: list[dict] = []

    for q in queries:
        any_of = q["any_of"]
        any_of_norm = [normalize_for_match(s) for s in any_of if s]
        any_of_norm = [n for n in any_of_norm if n]

        # 1. Fetch top-SCAN_K chunks.
        hits = kb.query(q["question"], k=args.scan_k)
        chunk_rank = _chunk_rank_of_target(hits, any_of_norm)

        # 2. Split each chunk into sentences; tag each sentence with its
        #    source chunk index so we can see which chunks' sentences dominate.
        sentences: list[str] = []
        sent_chunk_idx: list[int] = []
        for ci, h in enumerate(hits):
            for s in split_sentences(h.get("text") or ""):
                sentences.append(s)
                sent_chunk_idx.append(ci)

        # 3. Embed query + all sentences in one batch.
        if not sentences:
            sent_rank = None
        else:
            texts = [q["question"]] + sentences
            embs = embedder.embed(texts)
            q_emb = embs[0]
            sent_scores = [_cosine(q_emb, s_emb) for s_emb in embs[1:]]
            sent_rank = _best_matching_sentence_rank(
                sentences, sent_scores, any_of_norm
            )

        # Accounting.
        n_total += 1
        chunk_ok = chunk_rank is not None and chunk_rank <= 12
        sent_ok = sent_rank is not None and sent_rank <= args.top_n_sents
        if chunk_ok:
            n_chunk_hit_top12 += 1
        if sent_ok:
            n_sent_hit_topN += 1
        if not chunk_ok and sent_ok:
            n_chunk_miss_sent_rescue += 1
        if chunk_ok and not sent_ok:
            n_chunk_hit_sent_miss += 1

        rows.append({
            "id": q["id"],
            "shape": q["shape"],
            "chunk_rank": chunk_rank,
            "sent_rank": sent_rank,
            "n_sent": len(sentences),
            "would_rescue": (not chunk_ok) and sent_ok,
            "would_regress": chunk_ok and not sent_ok,
        })

        marker = "+" if chunk_ok else " "
        smarker = "+" if sent_ok else " "
        print(
            f"  [{marker}/{smarker}] {q['id']:<14} {q['shape']:<22} "
            f"chunk_rank={str(chunk_rank):<6}  "
            f"sent_rank={str(sent_rank):<6}  "
            f"n_sent={len(sentences)}"
        )

    # Summary.
    print("\n" + "=" * 70)
    print("Sentence-rerank diagnostic summary")
    print("=" * 70)
    print(f"  queries scored:              {n_total}")
    print(f"  chunk_rank <= 12:            {n_chunk_hit_top12}/{n_total}")
    print(f"  sent_rank <= {args.top_n_sents}:             "
          f"{n_sent_hit_topN}/{n_total}")
    print(f"  rescue candidates:           {n_chunk_miss_sent_rescue}/{n_total}  "
          f"(chunk miss, sentence hit)")
    print(f"  regression candidates:       {n_chunk_hit_sent_miss}/{n_total}  "
          f"(chunk hit, sentence miss)")
    print()
    print("Interpretation:")
    print("  If `rescue candidates` > `regression candidates`, sentence")
    print("  re-ranking is a net win on this harness and worth implementing.")
    print("  If they are roughly equal or regressions dominate, sentence")
    print("  re-ranking would not help — skip the implementation.")

    # Detail for rescues and regressions.
    rescues = [r for r in rows if r["would_rescue"]]
    regressions = [r for r in rows if r["would_regress"]]
    if rescues:
        print(f"\nRescue candidates (chunk rank > 12, sentence rank <= "
              f"{args.top_n_sents}):")
        for r in rescues:
            print(f"  {r['id']:<14} chunk={r['chunk_rank']}  "
                  f"sent={r['sent_rank']}  shape={r['shape']}")
    if regressions:
        print(f"\nRegression candidates (chunk rank <= 12, sentence rank > "
              f"{args.top_n_sents}):")
        for r in regressions:
            print(f"  {r['id']:<14} chunk={r['chunk_rank']}  "
                  f"sent={r['sent_rank']}  shape={r['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
