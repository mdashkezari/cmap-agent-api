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



def _split_text(text: str, max_chars: int = 20_000):
    """Yield chunks of `text` no longer than `max_chars`.

    This is a conservative, dependency-free splitter designed to keep
    embedding inputs under typical model context limits (~8k tokens).
    It prefers paragraph boundaries, then sentence boundaries, then words.
    """
    if text is None:
        return
    # Normalize whitespace a bit but keep paragraph structure
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return

    # Be conservative even if caller passes a larger number
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
        # If a single paragraph is too large, split further.
        if len(p) > max_chars:
            # Split on sentence-ish boundaries; if that still fails, split on words.
            parts = re.split(r"(?<=[\.!\?])\s+", p)
            if len(parts) == 1:
                parts = p.split()

            cur = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # For word-based splitting, add a space
                sep = " " if cur else ""
                candidate = f"{cur}{sep}{part}"
                if len(candidate) <= max_chars:
                    cur = candidate
                else:
                    if cur:
                        yield cur.strip()
                        cur = part
                    else:
                        # extreme fallback: hard cut
                        for i in range(0, len(part), max_chars):
                            yield part[i:i+max_chars].strip()
                        cur = ""
            if cur.strip():
                yield cur.strip()
            continue

        # Normal paragraph accumulation
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            yield from flush()
            buf = p

    if buf.strip():
        yield buf.strip()


def _join_list(values: list[str], limit: int = 30) -> str | None:
    """Return a compact, de-duplicated string for list-like metadata.

    Chroma metadata values must be scalar (no list/dict). We keep order-stable
    joins so the same input produces the same stored value.
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

def _dataset_doc(row: dict[str, Any], refs: list[str], vars_: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    title = row.get("DatasetName") or row.get("ShortName") or row.get("TableName")
    parts=[]
    def add(k,v):
        if v is None: return
        s=str(v).strip()
        if not s: return
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
    add("Units", row.get("Units"))
    add("Comments", row.get("Comments"))
    add("Regions", row.get("Regions"))
    # Coverage
    add("TimeMin", row.get("TimeMin"))
    add("TimeMax", row.get("TimeMax"))
    add("LatMin", row.get("LatMin"))
    add("LatMax", row.get("LatMax"))
    add("LonMin", row.get("LonMin"))
    add("LonMax", row.get("LonMax"))
    add("DepthMin", row.get("DepthMin"))
    add("DepthMax", row.get("DepthMax"))

    # Variables (compact)
    if vars_:
        parts.append("Variables:")
        for v in vars_[:80]:
            vn=v.get("VarName")
            ln=v.get("LongName")
            unit=v.get("Unit")
            parts.append(f"  - {vn} | {ln or ''} | {unit or ''}".strip())

    # References
    urls=[]
    dois=[]
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
        # Chroma metadata values must be scalar (no list/dict).
        "reference_urls": _join_list(urls, limit=30),
        "reference_dois": _join_list(dois, limit=30),
        "reference_url_count": len({u for u in urls if u}),
        "source": "cmap_sql_cache",
        "title": title,
    }
    return "\n".join(parts), meta

def _reference_docs(table: str, dataset_id: int | None, refs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
    out=[]
    for r in refs:
        rid = r.get("ReferenceId")
        txt = (r.get("Reference") or "").strip()
        if not txt:
            continue
        urls=URL_RE.findall(txt)
        dois=DOI_RE.findall(txt)
        doc_id=f"ref:{table}:{rid}"
        meta={
            "doc_type":"dataset_reference",
            "table": table,
            "dataset_id": dataset_id,
            "reference_id": rid,
            # Chroma metadata values must be scalar.
            "reference_urls": _join_list(urls, limit=10),
            "reference_dois": _join_list(dois, limit=10),
            "source":"tblDataset_References",
            "title": f"Reference {rid} for {table}",
        }
        out.append((doc_id, meta, txt))
    return out

def main():
    ap=argparse.ArgumentParser(description="Build/refresh the Chroma KB from agent catalog cache tables.")
    ap.add_argument("--rebuild", action="store_true", help="Delete all existing KB docs before re-indexing")
    ap.add_argument("--delete-stale", action="store_true", help="Delete KB docs not present in current catalog cache")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit number of datasets to index (dev)")
    args=ap.parse_args()

    store=SQLServerStore.from_env()
    kb=ChromaKB()

    if args.rebuild:
        # delete everything
        ids=kb.all_ids()
        kb.delete_ids(ids)

    with store.engine.begin() as conn:
        ds = pd.read_sql_query(text("SELECT * FROM agent.CatalogDatasets ORDER BY TableName"), conn)
        vars_ = pd.read_sql_query(text("SELECT * FROM agent.CatalogVariables"), conn)
        refs = pd.read_sql_query(text("SELECT * FROM agent.CatalogDatasetReferences"), conn)

    if args.limit and args.limit>0:
        ds = ds.head(args.limit)

    # Build maps
    vars_by_table={}
    for _, r in vars_.iterrows():
        vars_by_table.setdefault(r["TableName"], []).append(r.to_dict())

    refs_by_table={}
    for _, r in refs.iterrows():
        refs_by_table.setdefault(r["TableName"], []).append(r.to_dict())

    ids=[]
    texts=[]
    metas=[]

    for _, r in ds.iterrows():
        row=r.to_dict()
        table=row.get("TableName")
        if not table:
            continue
        did=row.get("DatasetId")
        refs_list=[x.get("Reference","") for x in refs_by_table.get(table, []) if x.get("Reference")]
        vars_list=vars_by_table.get(table, [])
        doc_text, meta = _dataset_doc(row, refs_list, vars_list)
        doc_id = f"ds:{table}"
        # Keep chunks small enough to stay well under common embedding
        # context limits (e.g., 8k tokens). We err on the conservative
        # side because some catalog fields can be extremely token-dense.
        for idx, chunk in enumerate(_split_text(doc_text, max_chars=7_000), start=1):
            cid = doc_id if idx == 1 else f"{doc_id}#chunk{idx}"
            cmeta = dict(meta)
            if idx != 1:
                cmeta["chunk_index"] = idx
            ids.append(cid)
            texts.append(chunk)
            metas.append(cmeta)

        # Reference docs
        for doc_id2, meta2, txt in _reference_docs(table, did, refs_by_table.get(table, [])):
            for idx, chunk in enumerate(_split_text(txt, max_chars=7_000), start=1):
                cid = doc_id2 if idx == 1 else f"{doc_id2}#chunk{idx}"
                cmeta = dict(meta2)
                if idx != 1:
                    cmeta["chunk_index"] = idx
                ids.append(cid)
                texts.append(chunk)
                metas.append(cmeta)

        # Variable docs (per variable)
        for v in vars_list:
            vid=f"var:{table}:{v.get('VarName')}"
            vtxt = "\n".join([
                f"Variable: {v.get('VarName')}",
                f"TableName: {table}",
                f"LongName: {v.get('LongName') or ''}",
                f"Unit: {v.get('Unit') or ''}",
                f"Keywords: {v.get('Keywords') or ''}",
            ])
            vmeta={
                "doc_type":"variable",
                "table": table,
                "dataset_id": did,
                "var_name": v.get("VarName"),
                "title": f"{v.get('VarName')} ({table})",
                "source":"agent.CatalogVariables",
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
        current=set(ids)
        existing=set(kb.all_ids())
        stale=sorted(existing-current)
        kb.delete_ids(stale)

    print(f"Indexed {len(ids)} KB docs into collection '{kb.collection_name}' at '{kb.persist_dir}'.")

if __name__=="__main__":
    main()

