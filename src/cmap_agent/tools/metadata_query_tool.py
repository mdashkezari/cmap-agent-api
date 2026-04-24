"""catalog.query_metadata — read-only SQL tool for CMAP metadata tables.

Allows the agent to answer structural/relational metadata questions that
udfCatalog() does not cover: cruise scientists, programs, regions,
collections, dataset references, server assignments, news, and so on.

Safety guarantees enforced in code (see ``tools.sql_validator``):
  - Only SELECT statements are permitted (no DDL/DML).
  - Only whitelisted metadata tables/functions may be referenced.
  - Data tables (tblSST_*, tblModis_*, etc.) are explicitly blocked.
  - Every query is capped at TOP N (``MAX_ROWS`` = 200).
  - ``SELECT *`` and ``<alias>.*`` are rejected; explicit column lists are
    required so that silent column-set drift cannot leak new fields.
  - ``User_ID`` / ``UserID`` may appear in WHERE / JOIN / GROUP BY but not
    in the SELECT projection.
  - Queries over tblCollections / tblCollection_Datasets /
    tblCollection_Follows must include a ``Private = 0`` predicate.
  - Queries over tblNews / tblNews_Datasets must include a
    ``view_status = 3`` predicate.

Query timeout: 15 seconds.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from cmap_agent.storage.sqlserver import SQLServerStore
from cmap_agent.tools.catalog_tools import _catalog_cache
from cmap_agent.tools.sql_validator import (
    ALLOWED_TABLES,
    MAX_ROWS,
    validate_sql,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema summary — injected into the system prompt / tool description.
# Kept in sync with the CMAP Database Metadata Tables bundle.
# ---------------------------------------------------------------------------

SCHEMA_SUMMARY = """
CMAP metadata tables (dbo schema, SQL Server).  Always prefix tables with
dbo., always include TOP N (≤ 200), and ALWAYS specify column lists —
SELECT * is refused.

Core catalog:
  tblDatasets(ID, DB, Dataset_Name[short], Dataset_Long_Name[title],
              Variables, Data_Source, Distributor, Description,
              Climatology[bit], Acknowledgement, Doc_URL, Icon_URL,
              Contact_Email, Dataset_Version, Dataset_Release_Date,
              Dataset_History)
  tblVariables(ID, DB, Dataset_ID→tblDatasets.ID, Table_Name,
               Short_Name, Long_Name, Unit,
               Temporal_Res_ID→tblTemporal_Resolutions.ID,
               Spatial_Res_ID→tblSpatial_Resolutions.ID,
               Temporal_Coverage_Begin, Temporal_Coverage_End,
               Lat_Coverage_Begin, Lat_Coverage_End,
               Lon_Coverage_Begin, Lon_Coverage_End,
               Grid_Mapping, Make_ID→tblMakes.ID, Sensor_ID→tblSensors.ID,
               Process_ID→tblProcess_Stages.ID,
               Study_Domain_ID→tblStudy_Domains.ID,
               Comment, Visualize[bit], Data_Type,
               Org_ID, Conversion_Coefficient)
  tblKeywords(var_ID→tblVariables.ID, keywords)       -- one row per keyword
  tblDataset_Stats(Dataset_ID, JSON_stats)            -- bbox/time/stats JSON
  tblDataset_References(Reference_ID, Dataset_ID, Reference, Data_DOI[bit])
  tblDatasets_JSON_Metadata(Dataset_ID, JSON_Metadata)
  tblVariables_JSON_Metadata(Var_ID, JSON_Metadata)

Cruises:
  tblCruise(ID, Nickname[human name e.g. 'Gradients_1'],
            Name[cruise# e.g. 'KOK1606'], Ship_Name,
            Start_Time, End_Time, Lat_Min, Lat_Max, Lon_Min, Lon_Max,
            Chief_Name,
            Cruise_Series[int FK→tblCruise_Series.ID — NOT a text field])
  tblCruise_Scientists(Cruise_ID, First_Name, Last_Name, Email, Chief[bit])
  tblCruise_Keywords(cruise_ID, keywords)
  tblCruise_Series(ID, Series)                        -- text name
  tblCruise_Regions(Cruise_ID, Region_ID→tblRegions.Region_ID)
  tblCruise_Trajectory(Cruise_ID, time, lat, lon)     -- dense lat/lon tracks
  tblDataset_Cruises(Dataset_ID, Cruise_ID)

  IMPORTANT: to search cruises by name use Nickname (e.g. WHERE Nickname
  LIKE '%Gradients%').  Do NOT LIKE-search Cruise_Series (integer FK).
  tblCruise_Trajectory can be very large — always keep TOP N small and
  filter by Cruise_ID first.

Programs & regions:
  tblPrograms(Program_ID, Program_Name)
  tblDataset_Programs(Dataset_ID, Program_ID)
  tblRegions(Region_ID, Region_Name)
  tblDataset_Regions(Dataset_ID, Region_ID)

Collections (PRIVACY RULE — enforced in code):
  tblCollections(Collection_ID, User_ID, Collection_Name,
                 Private[bit: 0=public, 1=private],
                 Description, Downloads, Views, Copies,
                 Created_At, Modified_At)
  tblCollection_Datasets(Collection_ID→tblCollections.Collection_ID,
                         Dataset_Short_Name)
  tblCollection_Follows(User_ID, Collection_ID→tblCollections.Collection_ID,
                        Follow_Date)

  Any query that references any of the three tables above MUST include a
  Private = 0 predicate (joining tblCollections if needed).  User_ID may
  be used in WHERE / JOIN / GROUP BY but MUST NOT appear in the SELECT
  projection — output it aggregated (e.g. COUNT(DISTINCT User_ID)) if the
  question calls for follower counts.

News (PUBLICATION RULE — enforced in code):
  tblNews(ID, headline, link, body, date[nvarchar], rank, view_status,
          create_date, modify_date, publish_date, UserID, Status_ID, Label)
  tblNews_Datasets(News_ID→tblNews.ID, Dataset_ID→tblDatasets.ID)

  Any query that references tblNews or tblNews_Datasets MUST include a
  view_status = 3 predicate (other values are drafts/unpublished).
  Order by publish_date DESC for "latest".  UserID follows the same no-
  SELECT-output rule as User_ID.

Lookup dimensions:
  tblMakes(ID, Make)                        -- Observation | Model | Assimilation
  tblSensors(ID, Sensor)                    -- Satellite | CTD | Flow Cytometer | …
  tblProcess_Stages(ID, Process_Stage, Process_Stage_Long)
  tblTemporal_Resolutions(ID, Temporal_Resolution)
  tblSpatial_Resolutions(ID, Spatial_Resolution)
  tblStudy_Domains(ID, Study_Domain)
  tblDataset_Servers(Dataset_ID, Server_Alias)   -- e.g. rainier, rossby

Functions:
  udfCatalog()  -- everything-in-one-view: variable × dataset × resolution ×
                   make × sensor × process × study domain, plus JSON stats.
                   Query as: FROM dbo.udfCatalog()  (no arguments).

General rules:
  - tblDatasets.Dataset_Name is the SHORT name; Dataset_Long_Name is the title.
  - tblVariables.Table_Name is the actual SQL table name (tblXXX).
  - Use udfCatalog() (or the catalog.search tools) for variable/dataset
    text search; use this tool for structural / relational questions.
  - Grouped results beat flat results when row counts are large — GROUP
    BY collection / program / region and return COUNTs plus a small
    representative sample.
  - Wrap STRING_AGG(...) in CAST(... AS NVARCHAR(MAX)) to avoid the
    8000-byte limit on NVARCHAR text.
  - Example (allowed):
      SELECT TOP 50 c.Collection_Name,
             COUNT(cd.Dataset_Short_Name) AS Dataset_Count
      FROM dbo.tblCollections c
      JOIN dbo.tblCollection_Datasets cd ON c.Collection_ID = cd.Collection_ID
      WHERE c.Private = 0
      GROUP BY c.Collection_Name
      ORDER BY c.Collection_Name
"""


# ---------------------------------------------------------------------------
# Arg model
# ---------------------------------------------------------------------------

class QueryMetadataArgs(BaseModel):
    sql: str = Field(
        ...,
        description=(
            "A read-only SELECT query against CMAP metadata tables. "
            "Must reference only whitelisted metadata tables (not data "
            "tables).  Must include TOP N (max 200).  Must list columns "
            "explicitly — SELECT * and alias.* are refused.  SQL Server "
            "dialect.  Always prefix tables with dbo. (e.g. SELECT TOP 20 "
            "Name FROM dbo.tblCruise).  Use this for questions about "
            "cruises, scientists, programs, regions, collections, "
            "references — NOT for variable data retrieval."
        ),
    )
    intent: str = Field(
        "",
        description="One-sentence description of what this query answers "
                    "(used for logging).",
    )


# ---------------------------------------------------------------------------
# Data-table lookup (from the in-memory catalog cache)
# ---------------------------------------------------------------------------

_QUERY_TIMEOUT = 15  # seconds


def _get_data_table_names(store: SQLServerStore) -> frozenset[str]:
    """Return the set of data-table names from the catalog cache.

    The cache is loaded at startup so this is free.  If the cache is empty
    for any reason (fresh process, cache miss), a quick distinct-query
    against tblVariables.Table_Name is the fallback.
    """
    cache = _catalog_cache
    if cache.rows:
        return frozenset(
            str(r.get("table_name") or "")
            for r in cache.rows
            if r.get("table_name")
        )
    try:
        with store.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT Table_Name FROM dbo.tblVariables "
                    "WHERE Table_Name IS NOT NULL"
                )
            ).fetchall()
        return frozenset(r[0] for r in rows if r[0])
    except Exception:
        return frozenset()


# ---------------------------------------------------------------------------
# Back-compat private aliases — retained so that any code path or older
# regression script that imported these names keeps working.
# ---------------------------------------------------------------------------

_ALLOWED_TABLES = ALLOWED_TABLES
_MAX_ROWS = MAX_ROWS


def _validate_sql(sql: str, data_table_names: frozenset[str]) -> str | None:
    """Thin shim around ``sql_validator.validate_sql`` for back-compat."""
    return validate_sql(sql, data_table_names)


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

    data_tables = _get_data_table_names(store)

    err = validate_sql(sql, data_tables)
    if err:
        return {"error": err, "sql": sql}

    log.info(
        "catalog.query_metadata: %s | SQL: %.200s",
        args.intent or "?",
        sql,
    )

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
