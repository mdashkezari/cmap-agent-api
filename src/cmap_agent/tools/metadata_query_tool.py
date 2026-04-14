"""catalog.query_metadata — read-only SQL tool for CMAP metadata tables.

Allows the agent to answer structural/relational metadata questions that
udfCatalog() doesn't cover: cruise scientists, programs, regions, collections,
dataset references, server assignments, etc.

Safety guarantees:
  - Only SELECT statements are permitted (no DDL/DML).
  - Only whitelisted metadata tables may be referenced.
  - Data tables (tblSST_*, tblModis_*, etc.) are explicitly blocked.
  - Every query is capped at TOP 200 rows.
  - Query timeout: 15 seconds.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from cmap_agent.storage.sqlserver import SQLServerStore
from cmap_agent.tools.catalog_tools import _catalog_cache

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whitelisted metadata tables (dbo schema only)
# ---------------------------------------------------------------------------

_ALLOWED_TABLES: frozenset[str] = frozenset({
    "tblDatasets",
    "tblVariables",
    "tblKeywords",
    "tblDataset_Stats",
    "tblDataset_References",
    "tblDataset_Cruises",
    "tblDataset_Programs",
    "tblDataset_Regions",
    "tblDataset_Servers",
    "tblCruise",
    "tblCruise_Keywords",
    "tblCruise_Scientists",
    "tblCruise_Series",
    "tblCruise_Regions",
    "tblPrograms",
    "tblRegions",
    "tblCollections",
    "tblCollection_Datasets",
    "tblNews",
    "tblNews_Datasets",
    "tblMakes",
    "tblSensors",
    "tblProcess_Stages",
    "tblTemporal_Resolutions",
    "tblSpatial_Resolutions",
    "tblStudy_Domains",
})

# Prefixes that identify data tables — blocked regardless of whitelist
_DATA_TABLE_PREFIXES = (
    "tblSSTd", "tblSST_", "tblModis", "tblERA", "tblWOA",
    "tblPisces", "tblDarwin", "tblArgo", "tblSeaFlow",
    "tblGLODAP", "tblGEBCO", "tblInflux", "tblKOK", "tblKM",
    "tblMGL", "tblTN", "tblHOT", "tblFalkor", "tblGradients",
)

_MAX_ROWS = 200
_QUERY_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Schema context injected into the system prompt / tool description
# ---------------------------------------------------------------------------

SCHEMA_SUMMARY = """
CMAP metadata tables (dbo schema, SQL Server):

Core catalog:
  tblDatasets(ID, Dataset_Name[short], Dataset_Long_Name[title], Description,
              Data_Source, Distributor, Acknowledgement, Climatology[bit],
              Doc_URL, Contact_Email, Dataset_Version, Dataset_Release_Date)
  tblVariables(ID, Dataset_ID→tblDatasets.ID, Table_Name, Short_Name,
               Long_Name, Unit, Has_Depth[bit], Visualize[bit],
               Make_ID→tblMakes, Sensor_ID→tblSensors,
               Temporal_Res_ID→tblTemporal_Resolutions,
               Spatial_Res_ID→tblSpatial_Resolutions)
  tblKeywords(var_ID→tblVariables.ID, keywords)  -- one row per keyword per var
  tblDataset_Stats(Dataset_ID, JSON_stats)        -- bbox/time/stats as JSON
  tblDataset_References(Reference_ID, Dataset_ID, Reference, Data_DOI[bit])

Cruises:
  tblCruise(ID, Nickname[human-readable name e.g. 'Gradients_1'], Name[cruise# e.g. 'KOK1606'],
            Ship_Name, Start_Time, End_Time,
            Lat_Min, Lat_Max, Lon_Min, Lon_Max, Chief_Name,
            Cruise_Series[int FK→tblCruise_Series.ID — NOT a text field, do not LIKE-search it])
  tblCruise_Scientists(Cruise_ID, First_Name, Last_Name, Email, Chief[bit])
  tblCruise_Keywords(cruise_ID, keywords)
  tblCruise_Series(ID, Series)  -- text name of the cruise series
  tblCruise_Regions(Cruise_ID, Region_ID→tblRegions)
  tblDataset_Cruises(Dataset_ID, Cruise_ID)  -- links datasets to cruises

  IMPORTANT: To search cruises by name use Nickname (e.g. WHERE Nickname LIKE '%Gradients%').
  Do NOT use Cruise_Series for text search — it is an integer FK.
  Chief_Name is stored directly on tblCruise; tblCruise_Scientists has fuller per-person records.

Programs & regions:
  tblPrograms(Program_ID, Program_Name)
  tblDataset_Programs(Dataset_ID, Program_ID)
  tblRegions(Region_ID, Region_Name)
  tblDataset_Regions(Dataset_ID, Region_ID)

Collections (public only — NEVER expose rows where Private=1):
  tblCollections(Collection_ID, User_ID, Collection_Name, Private[bit: 0=public, 1=private],
                 Description, Downloads, Views, Copies, Created_At, Modified_At)
  tblCollection_Datasets(Collection_ID→tblCollections.Collection_ID, Dataset_Short_Name)
  tblCollection_Follows(User_ID, Collection_ID→tblCollections.Collection_ID, Follow_Date)

News announcements:
  tblNews(ID, headline, body, link, date[nvarchar], publish_date, rank,
          view_status[IMPORTANT: always filter WHERE view_status = 3 for published news],
          Status_ID, Label, create_date, modify_date, UserID)
  tblNews_Datasets(News_ID→tblNews.ID, Dataset_ID→tblDatasets.ID)
  NOTE: ALWAYS filter tblNews with WHERE view_status = 3 — other values are drafts/unpublished.
        Order by publish_date DESC for latest news.

Lookup tables:
  tblMakes(ID, Make)                     -- e.g. Observation, Model, Assimilation
  tblSensors(ID, Sensor)                 -- e.g. Satellite, CTD, Flow Cytometer
  tblProcess_Stages(ID, Process_Stage, Process_Stage_Long)
  tblTemporal_Resolutions(ID, Temporal_Resolution)
  tblSpatial_Resolutions(ID, Spatial_Resolution)
  tblDataset_Servers(Dataset_ID, Server_Alias)  -- e.g. rainier, rossby

Notes:
  - tblDatasets.Dataset_Name is the SHORT name; Dataset_Long_Name is the title.
  - tblVariables.Table_Name is the actual SQL table name (tblXXX).
  - Use udfCatalog() (via catalog.search tools) for variable/dataset search.
  - This tool is for structural questions: cruises, programs, regions, scientists,
    collections, news. Always use dbo. prefix. Always include TOP N. SQL Server only.
  - For "datasets in each collection", query count + a few sample datasets per
    collection. Use NVARCHAR(MAX) cast in STRING_AGG to avoid the 8000-byte limit,
    and limit the sample to avoid very large results. Example:
      SELECT TOP 50 c.Collection_Name,
             COUNT(cd.Dataset_Short_Name) AS Dataset_Count,
             (SELECT TOP 5 STRING_AGG(CAST(cd2.Dataset_Short_Name AS NVARCHAR(MAX)), ', ')
              FROM dbo.tblCollection_Datasets cd2
              WHERE cd2.Collection_ID = c.Collection_ID) AS Sample_Datasets
      FROM dbo.tblCollections c
      JOIN dbo.tblCollection_Datasets cd ON c.Collection_ID = cd.Collection_ID
      WHERE c.Private = 0
      GROUP BY c.Collection_ID, c.Collection_Name
      ORDER BY c.Collection_Name
  - NEVER use STRING_AGG without CAST(... AS NVARCHAR(MAX)) — bare STRING_AGG on
    NVARCHAR columns exceeds 8000 bytes for large collections and causes an error.
  - If a result has many rows, summarize by grouping rather than returning raw flat data.
"""


# ---------------------------------------------------------------------------
# Arg model
# ---------------------------------------------------------------------------

class QueryMetadataArgs(BaseModel):
    sql: str = Field(
        ...,
        description=(
            "A read-only SELECT query against CMAP metadata tables. "
            "Must reference only whitelisted metadata tables (not data tables). "
            "Must include TOP N (max 200). SQL Server dialect. "
            "Always prefix tables with dbo. (e.g. SELECT TOP 20 * FROM dbo.tblCruise). "
            "Use this for questions about cruises, scientists, programs, regions, "
            "collections, references — NOT for variable data retrieval."
        ),
    )
    intent: str = Field(
        "",
        description="One-sentence description of what this query answers (for logging).",
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _extract_table_names(sql: str) -> list[str]:
    """Extract table names referenced in a SQL query (rough heuristic)."""
    # Match FROM/JOIN followed by optional schema prefix and table name
    pattern = re.compile(
        r'\b(?:FROM|JOIN)\s+(?:dbo\.)?(\w+)',
        re.IGNORECASE,
    )
    return [m.group(1) for m in pattern.finditer(sql)]


def _validate_sql(sql: str, data_table_names: frozenset[str]) -> str | None:
    """Return an error string if the SQL is not permitted, else None."""
    sql_stripped = sql.strip()

    # Must be a SELECT statement
    if not re.match(r'^\s*SELECT\b', sql_stripped, re.IGNORECASE):
        return "Only SELECT statements are permitted."

    # Block dangerous keywords
    blocked = re.compile(
        r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|EXECUTE|'
        r'TRUNCATE|MERGE|GRANT|REVOKE|xp_|sp_)\b',
        re.IGNORECASE,
    )
    if blocked.search(sql_stripped):
        return "Query contains prohibited SQL keywords."

    # Check all referenced tables are whitelisted metadata tables
    referenced = _extract_table_names(sql_stripped)
    for tbl in referenced:
        # Check if it's a known data table
        if tbl in data_table_names:
            return (
                f"'{tbl}' is a data table. Use pycmap / cmap.space_time to query data. "
                f"This tool is for metadata tables only."
            )
        if any(tbl.lower().startswith(p.lower()) for p in _DATA_TABLE_PREFIXES):
            return (
                f"'{tbl}' appears to be a data table. "
                f"This tool is for metadata tables only."
            )
        if tbl not in _ALLOWED_TABLES:
            return (
                f"Table '{tbl}' is not in the allowed metadata table list. "
                f"Allowed: {', '.join(sorted(_ALLOWED_TABLES))}."
            )

    # Require TOP N
    if not re.search(r'\bTOP\s+\d+\b', sql_stripped, re.IGNORECASE):
        return "Query must include TOP N (e.g. TOP 50) to limit result size."

    # Cap TOP N at _MAX_ROWS
    top_match = re.search(r'\bTOP\s+(\d+)\b', sql_stripped, re.IGNORECASE)
    if top_match and int(top_match.group(1)) > _MAX_ROWS:
        return f"TOP N cannot exceed {_MAX_ROWS}. Please reduce the limit."

    return None  # valid


def _get_data_table_names(store: SQLServerStore) -> frozenset[str]:
    """Get the set of data table names from the catalog cache (fast, no SQL)."""
    cache = _catalog_cache
    if cache.rows:
        return frozenset(
            str(r.get("table_name") or "")
            for r in cache.rows
            if r.get("table_name")
        )
    # Fallback: query directly if cache not loaded
    try:
        with store.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT Table_Name FROM dbo.tblVariables WHERE Table_Name IS NOT NULL")
            ).fetchall()
        return frozenset(r[0] for r in rows if r[0])
    except Exception:
        return frozenset()


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

def query_metadata(args: QueryMetadataArgs, ctx: dict) -> dict:
    """Execute a read-only SQL query against CMAP metadata tables."""
    store = ctx.get("store")
    if not isinstance(store, SQLServerStore):
        try:
            store = SQLServerStore.from_env()
        except Exception as e:
            return {"error": f"No database connection available: {e}"}

    sql = (args.sql or "").strip()
    if not sql:
        return {"error": "No SQL query provided."}

    # Get data table names for validation (from cache — free)
    data_tables = _get_data_table_names(store)

    # Validate
    err = _validate_sql(sql, data_tables)
    if err:
        return {"error": err, "sql": sql}

    log.info("catalog.query_metadata: %s | SQL: %.200s", args.intent or "?", sql)

    try:
        with store.engine.connect() as conn:
            conn = conn.execution_options(timeout=_QUERY_TIMEOUT)
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "sql": sql,
        }

    except Exception as e:
        log.warning("catalog.query_metadata error: %s", e)
        return {"error": str(e), "sql": sql}
