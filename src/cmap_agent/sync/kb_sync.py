"""kb_sync — build/refresh the KB directly from CMAP metadata tables.

Data is sourced from:
  - udfCatalog()         — all variables with dataset metadata, keywords, coverage
  - tblDataset_References — citations/references per dataset

No intermediate agent.Catalog* tables are used. The script can be run
at any time to pick up new datasets or updated metadata without requiring
a separate catalog sync step first.

Supports two backends (controlled by CMAP_AGENT_KB_BACKEND or --target):
  - chroma: legacy ChromaDB (dense-only, local file-based)
  - qdrant: Qdrant with hybrid dense + BM25 search

The --target flag allows directing sync output to a specific backend,
overriding the default from settings.  This is useful for running local
validation before pushing to a production Qdrant Cloud instance.

Entry point: cmap-agent-sync-kb  (see pyproject.toml [project.scripts])
"""
from __future__ import annotations

import argparse
import re
from typing import Any

import pandas as pd
from sqlalchemy import text
from pathlib import Path
import hashlib
import re as _re


def _sanitize_text(s: str) -> str:
    """Strip characters that break JSON/embedding API calls."""
    if not s:
        return " "
    s = s.replace("\x00", "")
    s = _re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
    s = s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return s.strip() or " "

from cmap_agent.storage.sqlserver import SQLServerStore
from cmap_agent.rag.retrieval import get_kb
from cmap_agent.config.settings import settings

URL_RE = re.compile(r"https?://\S+", re.I)
DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s]+)", re.I)

# ---------------------------------------------------------------------------
# udfCatalog column → kb_sync internal key mapping
# udfCatalog columns (after our SELECT aliases):
#   Variable, Table_Name, Long_Name, Unit, Make, Sensor, Process_Level,
#   Study_Domain, Temporal_Resolution, Spatial_Resolution,
#   Time_Min, Time_Max, Lat_Min, Lat_Max, Lon_Min, Lon_Max, Depth_Min, Depth_Max,
#   Dataset_Name (long title), Dataset_Short_Name (short name),
#   Data_Source, Distributor, Dataset_Description, Acknowledgement,
#   Dataset_ID, ID (var id), Keywords
# ---------------------------------------------------------------------------

_UDF_SQL = """
SELECT
    Table_Name          AS TableName,
    Dataset_ID          AS DatasetId,
    Dataset_Name        AS DatasetName,
    Dataset_Short_Name  AS ShortName,
    Make                AS Make,
    Sensor              AS Sensor,
    Process_Level       AS ProcessLevel,
    Study_Domain        AS StudyDomain,
    Temporal_Resolution AS TemporalResolution,
    Spatial_Resolution  AS SpatialResolution,
    Time_Min            AS TimeMin,
    Time_Max            AS TimeMax,
    Lat_Min             AS LatMin,
    Lat_Max             AS LatMax,
    Lon_Min             AS LonMin,
    Lon_Max             AS LonMax,
    Depth_Min           AS DepthMin,
    Depth_Max           AS DepthMax,
    Variable            AS VarName,
    Long_Name           AS LongName,
    Unit                AS Unit,
    Keywords            AS Keywords,
    Data_Source         AS DataSource,
    Distributor         AS Distributor,
    Dataset_Description AS Description,
    Acknowledgement     AS Acknowledgement
FROM udfCatalog()
"""


# ---------------------------------------------------------------------------
# Text splitting (unchanged)
# ---------------------------------------------------------------------------

def _split_text(text: str, max_chars: int = 20_000):
    """Yield chunks of `text` no longer than `max_chars`.

    Prefers paragraph boundaries, then sentence boundaries, then words.
    Keeps embedding inputs under typical model context limits (~8k tokens).
    """
    if text is None:
        return
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return

    max_chars = int(max(1_000, min(max_chars, 20_000)))

    if len(text) <= max_chars:
        yield text
        return

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            yield buf.strip()
        buf = ""

    for p in paras:
        if len(p) > max_chars:
            parts = re.split(r"(?<=[\.!\?])\s+", p)
            if len(parts) == 1:
                parts = p.split()
            cur = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                sep = " " if cur else ""
                candidate = f"{cur}{sep}{part}"
                if len(candidate) <= max_chars:
                    cur = candidate
                else:
                    if cur:
                        yield cur.strip()
                        cur = part
                    else:
                        for i in range(0, len(part), max_chars):
                            yield part[i:i+max_chars].strip()
                        cur = ""
            if cur.strip():
                yield cur.strip()
            continue

        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            yield from flush()
            buf = p

    if buf.strip():
        yield buf.strip()


# ---------------------------------------------------------------------------
# Metadata helpers (unchanged)
# ---------------------------------------------------------------------------

def _join_list(values: list[str], limit: int = 30) -> str | None:
    """Return a compact, de-duplicated string for list-like metadata.

    ChromaDB metadata values must be scalar (no list/dict). Order-stable
    joins ensure the same input produces the same stored value.
    """
    if not values:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return "; ".join(out) if out else None


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def _dataset_doc(row: dict[str, Any], refs: list[str], vars_: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Build the ChromaDB document text and metadata for a dataset."""
    title = row.get("DatasetName") or row.get("ShortName") or row.get("TableName")
    parts = []

    def add(k, v):
        if v is None:
            return
        s = str(v).strip()
        if not s:
            return
        parts.append(f"{k}: {s}")

    add("Dataset", title)
    add("ShortName", row.get("ShortName"))
    add("TableName", row.get("TableName"))
    add("DatasetId", row.get("DatasetId"))
    add("Description", row.get("Description"))
    add("Keywords", row.get("Keywords"))
    add("DataSource", row.get("DataSource"))
    add("Distributor", row.get("Distributor"))
    add("Acknowledgement", row.get("Acknowledgement"))
    add("Make", row.get("Make"))
    add("Sensor", row.get("Sensor"))
    add("ProcessLevel", row.get("ProcessLevel"))
    add("StudyDomain", row.get("StudyDomain"))
    add("TemporalResolution", row.get("TemporalResolution"))
    add("SpatialResolution", row.get("SpatialResolution"))
    add("TimeMin", row.get("TimeMin"))
    add("TimeMax", row.get("TimeMax"))
    add("LatMin", row.get("LatMin"))
    add("LatMax", row.get("LatMax"))
    add("LonMin", row.get("LonMin"))
    add("LonMax", row.get("LonMax"))
    add("DepthMin", row.get("DepthMin"))
    add("DepthMax", row.get("DepthMax"))

    if vars_:
        parts.append("Variables:")
        for v in vars_[:80]:
            vn = v.get("VarName")
            ln = v.get("LongName")
            unit = v.get("Unit")
            kw = v.get("Keywords")
            line = f"  - {vn} | {ln or ''} | {unit or ''}"
            if kw:
                line += f" | {kw}"
            parts.append(line.strip())

    urls = []
    dois = []
    if refs:
        parts.append("References:")
        for r in refs[:50]:
            parts.append(f"  - {r}")
            urls += URL_RE.findall(r)
            dois += DOI_RE.findall(r)

    meta = {
        "doc_type": "dataset",
        "table": row.get("TableName"),
        "dataset_id": row.get("DatasetId"),
        "dataset_name": title,
        "short_name": row.get("ShortName"),
        "make": row.get("Make") or "",
        "sensor": row.get("Sensor") or "",
        "reference_urls": _join_list(urls, limit=30),
        "reference_dois": _join_list(dois, limit=30),
        "reference_url_count": len({u for u in urls if u}),
        "source": "udfCatalog",
        "title": title,
    }
    return "\n".join(parts), meta


def _reference_docs(table: str, dataset_id: int | None, refs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
    """Build individual ChromaDB documents for each dataset reference/citation."""
    out = []
    for r in refs:
        rid = r.get("Reference_ID")
        txt = (r.get("Reference") or "").strip()
        if not txt:
            continue
        urls = URL_RE.findall(txt)
        dois = DOI_RE.findall(txt)
        doc_id = f"ref:{table}:{rid}"
        meta = {
            "doc_type": "dataset_reference",
            "table": table,
            "dataset_id": dataset_id,
            "reference_id": rid,
            "reference_urls": _join_list(urls, limit=10),
            "reference_dois": _join_list(dois, limit=10),
            "source": "tblDataset_References",
            "title": f"Reference {rid} for {table}",
        }
        out.append((doc_id, meta, txt))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reference bank ingestion
# ---------------------------------------------------------------------------

def _default_bank_dir() -> Path:
    """Return notrack/reference_bank relative to project root."""
    here = Path(__file__).resolve()
    project_root = here.parents[3]
    return project_root / "notrack" / "reference_bank"


def _file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


# Re-exported as ``_fix_pdf_number_breaks`` for backward compatibility with
# any external callers or older regression scripts.  The canonical
# implementation lives in ``sync.text_fixes`` so it is dependency-free and
# independently unit-testable.
from cmap_agent.sync.text_fixes import fix_pdf_number_breaks as _fix_pdf_number_breaks  # noqa: E402


def _extract_text_from_file(path: Path) -> str | None:
    """Extract plain text from PDF, HTML, Markdown, or plain text files.

    For PDFs, uses PyMuPDF block-level extraction.  Repeated journal page
    headers (e.g. "2 Scientific Data | (2025) 12:1078 | https://doi.org/...")
    are suppressed by dropping blocks in the top 10 % of the page that match
    a journal-metadata pattern.  Footer zone is intentionally NOT filtered —
    methods text in two-column layouts can appear at low y-positions and
    false-positive footer drops cause content loss.

    Post-processing fixes garbled numbers from two-column PDF line breaks.
    For example, "ranged from 1 – 1, 250, 359" (MuPDF artefact from a soft
    hyphen or en-dash at a line break inside the number 1,250,359) is
    normalised back to "ranged from 1 to 1,250,359".
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            import fitz  # pymupdf
            import re as _re

            doc = fitz.open(str(path))

            # Matches clear journal header metadata — kept intentionally narrow
            # to avoid dropping legitimate content.
            _JOURNAL_HEADER_PAT = _re.compile(
                r"(scientific\s+data|nature\s+communications|nature\s+methods"
                r"|plos\s+one|frontiers\s+in\s+\w|molecular\s+ecology\s+resources"
                r"|doi\.org/10\.\d{4}|www\.nature\.com|www\.frontiersin\.org)",
                _re.I,
            )

            all_parts: list[str] = []

            for page in doc:
                page_h = page.rect.height
                blocks = page.get_text("blocks")
                # Sort top-to-bottom, left-to-right for reading order
                blocks.sort(key=lambda b: (b[1], b[0]))

                page_parts: list[str] = []
                for b in blocks:
                    if b[6] != 0:  # skip image blocks
                        continue
                    text = b[4].strip()
                    if not text:
                        continue
                    y0 = b[1]
                    # Drop only header-zone blocks that clearly are journal metadata
                    if y0 < page_h * 0.10 and _JOURNAL_HEADER_PAT.search(text):
                        continue
                    page_parts.append(text)

                if page_parts:
                    all_parts.append("\n".join(page_parts))

            doc.close()
            raw = "\n\n".join(all_parts).strip()
            if raw:
                raw = _fix_pdf_number_breaks(raw)
            return raw or None

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("PDF extract failed %s: %s", path.name, e)
            return None

    if suffix in (".html", ".htm"):
        try:
            from html.parser import HTMLParser
            class _Strip(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._parts = []
                    self._skip = False
                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style"):
                        self._skip = True
                def handle_endtag(self, tag):
                    if tag in ("script", "style"):
                        self._skip = False
                def handle_data(self, data):
                    if not self._skip:
                        self._parts.append(data)
            p = _Strip()
            p.feed(path.read_text(errors="replace"))
            return " ".join(p._parts).strip() or None
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("HTML extract failed %s: %s", path.name, e)
            return None

    # Markdown / plain text
    if suffix in (".md", ".txt", ".rst", ""):
        try:
            return path.read_text(errors="replace").strip() or None
        except Exception:
            return None

    return None


def _ingest_reference_bank(
    bank_dir: Path,
    dataset_rows: dict,          # TableName → row dict (for dataset_id lookup)
    short_name_to_table: dict,   # ShortName → TableName
    ids: list,
    texts: list,
    metas: list,
) -> int:
    """Scan the reference bank and add paper-chunk documents to the KB lists.

    Returns the number of chunk documents added.
    """
    import logging
    log = logging.getLogger(__name__)

    if not bank_dir.exists():
        log.info("Reference bank not found at %s — skipping.", bank_dir)
        return 0

    added = 0
    for dataset_dir in sorted(bank_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        short_name = dataset_dir.name
        table_name = short_name_to_table.get(short_name)
        dataset_id = None
        if table_name and table_name in dataset_rows:
            dataset_id = dataset_rows[table_name].get("DatasetId")

        files = [f for f in sorted(dataset_dir.iterdir())
                 if f.is_file() and not f.name.startswith(".")]
        if not files:
            continue

        log.info("Reference bank: %s (%d file(s))", short_name, len(files))

        for fpath in files:
            text_content = _extract_text_from_file(fpath)
            if not text_content or len(text_content) < 50:
                log.debug("  Skipping %s (empty or unreadable)", fpath.name)
                continue

            fhash = _file_hash(fpath)
            base_id = f"refbank:{short_name}:{fpath.stem}:{fhash}"
            meta_base = {
                "doc_type": "paper_chunk",
                "short_name": short_name,
                "table": table_name or "",
                "dataset_id": dataset_id,
                "filename": fpath.name,
                "source": "reference_bank",
                "title": f"{short_name} — {fpath.stem}",
            }

            for idx, chunk in enumerate(_split_text(text_content, max_chars=settings.CMAP_AGENT_KB_REFBANK_CHUNK_SIZE), start=1):
                cid = base_id if idx == 1 else f"{base_id}#chunk{idx}"
                cmeta = dict(meta_base)
                if idx != 1:
                    cmeta["chunk_index"] = idx
                ids.append(cid)
                texts.append(_sanitize_text(chunk))
                metas.append(cmeta)
                added += 1

    return added


def _build_kb(target: str | None, collection: str | None = None):
    """Instantiate the appropriate KB backend.

    If *target* is given it overrides the ``CMAP_AGENT_KB_BACKEND`` setting.
    Accepted values: ``chroma``, ``qdrant``.
    """
    import os
    from cmap_agent.config.settings import settings

    effective = (target or settings.CMAP_AGENT_KB_BACKEND).lower().strip()

    if effective == "qdrant":
        from cmap_agent.rag.qdrant_kb import QdrantKB
        kb = QdrantKB(
            url=os.environ.get("QDRANT_URL") or settings.QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY") or settings.QDRANT_API_KEY,
            collection=collection or settings.CMAP_AGENT_KB_COLLECTION,
        )
        kb.ensure_collection()
        return kb

    # Default: chroma
    from cmap_agent.rag.chroma_kb import ChromaKB
    return ChromaKB(
        persist_dir=settings.CMAP_AGENT_CHROMA_DIR,
        collection=collection or settings.CMAP_AGENT_KB_COLLECTION,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Build/refresh the KB directly from CMAP metadata tables."
    )
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete all existing KB docs before re-indexing.")
    ap.add_argument("--delete-stale", action="store_true",
                    help="Delete KB docs not present in the current catalog.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Index only this many datasets (useful for dev/testing).")
    ap.add_argument("--bank-dir", default=None,
                    help="Path to reference bank root. Default: notrack/reference_bank/.")
    ap.add_argument("--skip-bank", action="store_true",
                    help="Skip reference bank ingestion (catalog metadata only).")
    ap.add_argument(
        "--target",
        choices=["chroma", "qdrant"],
        default=None,
        help=(
            "KB backend to sync into: 'chroma' (legacy local) or 'qdrant' "
            "(hybrid search). Defaults to CMAP_AGENT_KB_BACKEND setting."
        ),
    )
    ap.add_argument(
        "--collection",
        default=None,
        help="Override the collection/index name (useful for A/B testing).",
    )
    args = ap.parse_args()

    store = SQLServerStore.from_env()
    kb = _build_kb(args.target, collection=args.collection)
    backend_label = args.target or settings.CMAP_AGENT_KB_BACKEND
    print(f"KB backend: {backend_label}")

    if args.rebuild:
        if hasattr(kb, "delete_collection"):
            # Qdrant: drop and recreate the collection for a clean rebuild
            kb.delete_collection()
            kb.ensure_collection()
            print("Deleted and recreated collection (full rebuild).")
        else:
            ids = kb.all_ids()
            kb.delete_ids(ids)
            print(f"Deleted {len(ids)} existing KB docs.")

    print("Loading catalog from udfCatalog()...")
    with store.engine.connect() as conn:
        catalog = pd.read_sql_query(text(_UDF_SQL), conn)

    print(f"  {len(catalog)} variable rows across {catalog['TableName'].nunique()} tables.")

    # Build per-dataset aggregates: one representative row + all variables
    # Group by TableName; keep first row for dataset-level metadata (same for all vars in a table)
    dataset_rows = (
        catalog.drop_duplicates(subset=["TableName"])
        .set_index("TableName")
        .to_dict(orient="index")
    )
    vars_by_table: dict[str, list[dict]] = {}
    for _, r in catalog.iterrows():
        tbl = r["TableName"]
        vars_by_table.setdefault(tbl, []).append({
            "VarName": r["VarName"],
            "LongName": r["LongName"],
            "Unit": r["Unit"],
            "Keywords": r["Keywords"],
        })

    # Load references directly from tblDataset_References
    print("Loading references from tblDataset_References...")
    with store.engine.connect() as conn:
        refs_df = pd.read_sql_query(
            text("SELECT Dataset_ID, Reference_ID, Reference, Data_DOI FROM dbo.tblDataset_References"),
            conn,
        )
    # Map DatasetId → list of reference dicts
    refs_by_dataset_id: dict[int, list[dict]] = {}
    for _, r in refs_df.iterrows():
        did = int(r["Dataset_ID"])
        refs_by_dataset_id.setdefault(did, []).append(r.to_dict())

    # Apply dataset limit if requested
    tables = list(dataset_rows.keys())
    if args.limit and args.limit > 0:
        tables = tables[:args.limit]

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []

    for table in tables:
        row = dataset_rows[table]
        did = row.get("DatasetId")
        refs_list_raw = refs_by_dataset_id.get(int(did) if did is not None else -1, [])
        refs_strs = [x.get("Reference", "") for x in refs_list_raw if x.get("Reference")]
        vars_list = vars_by_table.get(table, [])

        # Dataset document
        doc_text, meta = _dataset_doc(row, refs_strs, vars_list)
        doc_id = f"ds:{table}"
        for idx, chunk in enumerate(_split_text(doc_text, max_chars=settings.CMAP_AGENT_KB_CATALOG_CHUNK_SIZE), start=1):
            cid = doc_id if idx == 1 else f"{doc_id}#chunk{idx}"
            cmeta = dict(meta)
            if idx != 1:
                cmeta["chunk_index"] = idx
            ids.append(cid)
            texts.append(_sanitize_text(chunk))
            metas.append(cmeta)

        # Reference documents (one per citation)
        for doc_id2, meta2, txt in _reference_docs(table, did, refs_list_raw):
            for idx, chunk in enumerate(_split_text(txt, max_chars=settings.CMAP_AGENT_KB_CATALOG_CHUNK_SIZE), start=1):
                cid = doc_id2 if idx == 1 else f"{doc_id2}#chunk{idx}"
                cmeta = dict(meta2)
                if idx != 1:
                    cmeta["chunk_index"] = idx
                ids.append(cid)
                texts.append(_sanitize_text(chunk))
                metas.append(cmeta)

        # Variable documents (one per variable, includes keywords)
        for v in vars_list:
            vid = f"var:{table}:{v.get('VarName')}"
            vtxt = "\n".join([
                f"Variable: {v.get('VarName')}",
                f"TableName: {table}",
                f"LongName: {v.get('LongName') or ''}",
                f"Unit: {v.get('Unit') or ''}",
                f"Keywords: {v.get('Keywords') or ''}",
            ])
            vmeta = {
                "doc_type": "variable",
                "table": table,
                "dataset_id": did,
                "var_name": v.get("VarName"),
                "title": f"{v.get('VarName')} ({table})",
                "source": "udfCatalog",
            }
            for idx, chunk in enumerate(_split_text(vtxt, max_chars=settings.CMAP_AGENT_KB_CATALOG_CHUNK_SIZE), start=1):
                cid = vid if idx == 1 else f"{vid}#chunk{idx}"
                cmeta = dict(vmeta)
                if idx != 1:
                    cmeta["chunk_index"] = idx
                ids.append(cid)
                texts.append(_sanitize_text(chunk))
                metas.append(cmeta)

    kb.upsert(ids=ids, texts=texts, metadatas=metas)

    # Reference bank ingestion
    if not args.skip_bank:
        bank_dir = Path(args.bank_dir) if args.bank_dir else _default_bank_dir()
        # Build short_name → table_name lookup
        short_name_to_table = {
            row.get("ShortName"): tname
            for tname, row in dataset_rows.items()
            if row.get("ShortName")
        }
        bank_ids: list[str] = []
        bank_texts: list[str] = []
        bank_metas: list[dict] = []
        n_bank = _ingest_reference_bank(
            bank_dir, dataset_rows, short_name_to_table,
            bank_ids, bank_texts, bank_metas,
        )
        if n_bank:
            kb.upsert(ids=bank_ids, texts=bank_texts, metadatas=bank_metas)
            ids.extend(bank_ids)
            print(f"Indexed {n_bank} reference bank chunks from {bank_dir}.")
        else:
            print("No reference bank documents found.")

    if args.delete_stale:
        current = set(ids)
        existing = set(kb.all_ids())
        stale = sorted(existing - current)
        if stale:
            kb.delete_ids(stale)
            print(f"Deleted {len(stale)} stale KB docs.")

    # Summary
    dest = getattr(kb, "url", None) or getattr(kb, "persist_dir", backend_label)
    print(
        f"Done. Indexed {len(ids)} KB docs across {len(tables)} datasets "
        f"into collection '{kb.collection_name}' ({backend_label} → {dest})."
    )


if __name__ == "__main__":
    main()
