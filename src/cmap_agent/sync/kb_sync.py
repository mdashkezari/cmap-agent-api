"""kb_sync — build/refresh the Chroma KB directly from CMAP metadata tables.

Data is sourced from:
  - udfCatalog()         — all variables with dataset metadata, keywords, coverage
  - tblDataset_References — citations/references per dataset

No intermediate agent.Catalog* tables are used. The script can be run
at any time to pick up new datasets or updated metadata without requiring
a separate catalog sync step first.

Entry point: cmap-agent-sync-kb  (see pyproject.toml [project.scripts])
"""
from __future__ import annotations

import argparse
import re
from typing import Any

import pandas as pd
from sqlalchemy import text

from cmap_agent.storage.sqlserver import SQLServerStore
from cmap_agent.rag.chroma_kb import ChromaKB

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

def main():
    ap = argparse.ArgumentParser(
        description="Build/refresh the Chroma KB directly from CMAP metadata tables."
    )
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete all existing KB docs before re-indexing.")
    ap.add_argument("--delete-stale", action="store_true",
                    help="Delete KB docs not present in the current catalog.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Index only this many datasets (useful for dev/testing).")
    args = ap.parse_args()

    store = SQLServerStore.from_env()
    kb = ChromaKB()

    if args.rebuild:
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
        for idx, chunk in enumerate(_split_text(doc_text, max_chars=7_000), start=1):
            cid = doc_id if idx == 1 else f"{doc_id}#chunk{idx}"
            cmeta = dict(meta)
            if idx != 1:
                cmeta["chunk_index"] = idx
            ids.append(cid)
            texts.append(chunk)
            metas.append(cmeta)

        # Reference documents (one per citation)
        for doc_id2, meta2, txt in _reference_docs(table, did, refs_list_raw):
            for idx, chunk in enumerate(_split_text(txt, max_chars=7_000), start=1):
                cid = doc_id2 if idx == 1 else f"{doc_id2}#chunk{idx}"
                cmeta = dict(meta2)
                if idx != 1:
                    cmeta["chunk_index"] = idx
                ids.append(cid)
                texts.append(chunk)
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
            for idx, chunk in enumerate(_split_text(vtxt, max_chars=7_000), start=1):
                cid = vid if idx == 1 else f"{vid}#chunk{idx}"
                cmeta = dict(vmeta)
                if idx != 1:
                    cmeta["chunk_index"] = idx
                ids.append(cid)
                texts.append(chunk)
                metas.append(cmeta)

    kb.upsert(ids=ids, texts=texts, metadatas=metas)

    if args.delete_stale:
        current = set(ids)
        existing = set(kb.all_ids())
        stale = sorted(existing - current)
        if stale:
            kb.delete_ids(stale)
            print(f"Deleted {len(stale)} stale KB docs.")

    print(
        f"Indexed {len(ids)} KB docs across {len(tables)} datasets "
        f"into collection '{kb.collection_name}' at '{kb.persist_dir}'."
    )


if __name__ == "__main__":
    main()
