from __future__ import annotations

import argparse
import json
import math
import re
from typing import Any

import pandas as pd
from sqlalchemy import text

MAX_TABLE_NAME = 128
MAX_DATASET_SHORTNAME = 450
MAX_VAR_NAME = 200

def _truncate_str(value, max_len: int):
    """Safely truncate values before inserting into SQL Server fixed-length columns."""
    if value is None:
        return None
    s = str(value)
    return s if len(s) <= max_len else s[:max_len]


def _split_tokens(value: str) -> list[str]:
    """Turn a raw keyword field into a list of tokens.

    In Opedia's tblKeywords, each row is typically a single keyword/phrase. We keep the
    logic conservative (no aggressive splitting) to avoid accidentally exploding
    long free-text fields into "keywords".
    """
    if value is None:
        return []
    s = str(value).strip()
    return [s] if s else []



def _sql_value(v):
    """Convert pandas/numpy values to DB-safe Python values for pyodbc.

    - None/NaN/NaT -> None
    - pandas Timestamp -> python datetime (tz-naive UTC)
    - pandas Series (duplicate column access) -> first element (recursively cleaned)
    - numpy scalar -> python scalar
    """
    if v is None:
        return None

    # If a DataFrame has duplicate column names, row["col"] can return a Series.
    if isinstance(v, pd.Series):
        if v.empty:
            return None
        return _sql_value(v.iloc[0])

    # NaN/NaT handling
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # Timestamp handling
    if isinstance(v, pd.Timestamp):
        try:
            if v.tzinfo is not None:
                v = v.tz_convert("UTC").tz_localize(None)
        except Exception:
            # already tz-naive
            pass
        try:
            return v.to_pydatetime()
        except Exception:
            return None

    # numpy scalar -> python scalar
    try:
        import numpy as np  # type: ignore
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass

    return v


def _sql_params(d: dict) -> dict:
    return {k: _sql_value(v) for k, v in d.items()}


def _ensure_col(df: pd.DataFrame, target: str, candidates: list[str] | None = None) -> None:
    """Ensure df[target] exists. If any candidate column exists, copy it; otherwise set to None."""
    if target in df.columns:
        return
    candidates = candidates or []
    # exact matches
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            return
    # fuzzy contains match
    tl = target.lower()
    for c in df.columns:
        cl = c.lower()
        if tl in cl:
            df[target] = df[c]
            return
    df[target] = None

from cmap_agent.storage.sqlserver import SQLServerStore
def _fetch_catalog(store: SQLServerStore) -> pd.DataFrame:
    """For consistency, fetch the CMAP catalog from SQL Server directly, instead of using pycmap."""
    df = pd.DataFrame()
    with store.engine.begin() as conn:
        q = """
        SELECT RTRIM(LTRIM(Short_Name)) AS Variable,
        [tblVariables].Table_Name AS [Table_Name],
        RTRIM(LTRIM(Long_Name)) AS [Long_Name],
        RTRIM(LTRIM(Unit)) AS [Unit],
        RTRIM(LTRIM(Make)) AS [Make],
        RTRIM(LTRIM(Sensor)) AS [Sensor],
        RTRIM(LTRIM(Process_Stage_Long)) AS [Process_Level],
        RTRIM(LTRIM(Study_Domain)) AS [Study_Domain],
        RTRIM(LTRIM(Temporal_Resolution)) AS [Temporal_Resolution],
        RTRIM(LTRIM(Spatial_Resolution)) AS [Spatial_Resolution],
        JSON_VALUE(JSON_stats,'$.time.min') AS [Time_Min],
        JSON_VALUE(JSON_stats,'$.time.max') AS [Time_Max],
        CAST(JSON_VALUE(JSON_stats,'$.lat.min') AS float) AS [Lat_Min],
        CAST(JSON_VALUE(JSON_stats,'$.lat.max') AS float) AS [Lat_Max],
        CAST(JSON_VALUE(JSON_stats,'$.lon.min') AS float) AS [Lon_Min],
        CAST(JSON_VALUE(JSON_stats,'$.lon.max') AS float) AS [Lon_Max],
        CAST(JSON_VALUE(JSON_stats,'$.depth.min') AS float) AS [Depth_Min],
        CAST(JSON_VALUE(JSON_stats,'$.depth.max') AS float) AS [Depth_Max],
        CAST(JSON_VALUE(JSON_stats,'$."'+[Short_Name]+'"."25%"') AS float) AS [Variable_25th],
        CAST(JSON_VALUE(JSON_stats,'$."'+[Short_Name]+'"."50%"') AS float) AS [Variable_50th],
        CAST(JSON_VALUE(JSON_stats,'$."'+[Short_Name]+'"."75%"') AS float) AS [Variable_75th],
        CAST(JSON_VALUE(JSON_stats,'$."'+[Short_Name]+'".count') AS float) AS [Variable_Count],
        CAST(JSON_VALUE(JSON_stats,'$."'+[Short_Name]+'".mean') AS float) AS [Variable_Mean],
        CAST(JSON_VALUE(JSON_stats,'$."'+[Short_Name]+'".std') AS float) AS [Variable_Std],
        CAST(JSON_VALUE(JSON_stats,'$."'+[Short_Name]+'".min') AS float) AS [Variable_Min],
        CAST(JSON_VALUE(JSON_stats,'$."'+[Short_Name]+'".max') AS float) AS [Variable_Max],
        RTRIM(LTRIM(Comment)) AS [Comment],
        RTRIM(LTRIM(Dataset_Long_Name)) AS [Dataset_Name],
        RTRIM(LTRIM(Dataset_Name)) AS [Dataset_Short_Name],
        RTRIM(LTRIM([Data_Source])) AS [Data_Source],
        RTRIM(LTRIM(Distributor)) AS [Distributor],
        RTRIM(LTRIM([Description])) AS [Dataset_Description],
        RTRIM(LTRIM([Acknowledgement])) AS [Acknowledgement],
        [tblVariables].Dataset_ID AS [Dataset_ID],
        [tblVariables].ID AS [ID],
        [tblVariables].Visualize AS [Visualize],
        [keywords_agg].Keywords AS [Keywords],
        [Dataset_Metadata].Unstructured_Dataset_Metadata as [Unstructured_Dataset_Metadata],
        [Variable_Metadata].Unstructured_Variable_Metadata as [Unstructured_Variable_Metadata]
        FROM tblVariables
        JOIN tblDataset_Stats ON [tblVariables].Dataset_ID = [tblDataset_Stats].Dataset_ID
        JOIN tblDatasets ON [tblVariables].Dataset_ID=[tblDatasets].ID
        JOIN tblTemporal_Resolutions ON [tblVariables].Temporal_Res_ID=[tblTemporal_Resolutions].ID
        JOIN tblSpatial_Resolutions ON [tblVariables].Spatial_Res_ID=[tblSpatial_Resolutions].ID
        JOIN tblMakes ON [tblVariables].Make_ID=[tblMakes].ID
        JOIN tblSensors ON [tblVariables].Sensor_ID=[tblSensors].ID
        JOIN tblProcess_Stages ON [tblVariables].Process_ID=[tblProcess_Stages].ID
        JOIN tblStudy_Domains ON [tblVariables].Study_Domain_ID=[tblStudy_Domains].ID
        JOIN (SELECT var_ID, STRING_AGG (CAST(keywords as NVARCHAR(MAX)), ', ') AS Keywords FROM tblVariables var_table
        JOIN tblKeywords key_table ON [var_table].ID = [key_table].var_ID GROUP BY var_ID)
        AS keywords_agg ON [keywords_agg].var_ID = [tblVariables].ID
        LEFT JOIN (SELECT Dataset_ID, STRING_AGG (CAST(JSON_Metadata as NVARCHAR(MAX)), ', ') AS Unstructured_Dataset_Metadata FROM tblDatasets dataset_table
        JOIN tblDatasets_JSON_Metadata meta_table ON [dataset_table].ID = [meta_table].Dataset_ID GROUP BY Dataset_ID)
        AS Dataset_Metadata ON [Dataset_Metadata].Dataset_ID = [tblDatasets].ID
        LEFT JOIN (SELECT Var_ID, STRING_AGG (CAST(JSON_Metadata as NVARCHAR(MAX)), ', ') AS Unstructured_Variable_Metadata FROM tblVariables var_meta_table
        JOIN tblVariables_JSON_Metadata meta_table ON [var_meta_table].ID = [meta_table].Var_ID GROUP BY Var_ID)
        AS Variable_Metadata ON [Variable_Metadata].Var_ID = [tblVariables].ID
        """
        df = pd.read_sql_query(text(q), conn)
    return df


def _chunked(seq: list[int], n: int) -> list[list[int]]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]

def _fetch_stats(store: SQLServerStore, dataset_ids: list[int]) -> pd.DataFrame:
    if not dataset_ids:
        return pd.DataFrame()
    rows=[]
    with store.engine.begin() as conn:
        for chunk in _chunked(dataset_ids, 900):
            ph = ",".join(str(int(x)) for x in chunk)
            q = f"""
            SELECT
                Dataset_ID,
                JSON_VALUE(JSON_stats,'$.time.min') AS [Time_Min],
                JSON_VALUE(JSON_stats,'$.time.max') AS [Time_Max],
                CAST(JSON_VALUE(JSON_stats,'$.lat.min') AS float) AS [Lat_Min],
                CAST(JSON_VALUE(JSON_stats,'$.lat.max') AS float) AS [Lat_Max],
                CAST(JSON_VALUE(JSON_stats,'$.lon.min') AS float) AS [Lon_Min],
                CAST(JSON_VALUE(JSON_stats,'$.lon.max') AS float) AS [Lon_Max],
                CAST(JSON_VALUE(JSON_stats,'$.depth.min') AS float) AS [Depth_Min],
                CAST(JSON_VALUE(JSON_stats,'$.depth.max') AS float) AS [Depth_Max]
            FROM tblDataset_Stats
            WHERE Dataset_ID IN ({ph})
            """
            df = pd.read_sql_query(text(q), conn)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def _fetch_references(store: SQLServerStore, dataset_ids: list[int]) -> pd.DataFrame:
    if not dataset_ids:
        return pd.DataFrame()
    rows=[]
    with store.engine.begin() as conn:
        for chunk in _chunked(dataset_ids, 900):
            ph=",".join(str(int(x)) for x in chunk)
            q=f"""
            SELECT Reference_ID, Dataset_ID, Reference
            FROM tblDataset_References
            WHERE Dataset_ID IN ({ph})
            """
            df=pd.read_sql_query(text(q), conn)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fetch_dataset_descriptions(store: SQLServerStore, dataset_ids: list[int]) -> dict[int, str]:
    """Fetch the long-form dataset Description from Opedia.tblDatasets.

    Some catalog views do not always include the full dataset description, so we source it directly from tblDatasets.
    """
    if not dataset_ids:
        return {}

    frames: list[pd.DataFrame] = []
    with store.engine.begin() as conn:
        for chunk in _chunked(dataset_ids, 900):
            ph = ",".join(str(int(x)) for x in chunk)
            q = f"""
            SELECT ID AS Dataset_ID, Description
            FROM tblDatasets
            WHERE ID IN ({ph})
            """
            frames.append(pd.read_sql_query(text(q), conn))

    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    out: dict[int, str] = {}
    for _, r in df.iterrows():
        did = r.get("Dataset_ID")
        desc = r.get("Description")
        if did is None:
            continue
        try:
            did_i = int(did)
        except Exception:
            continue
        if isinstance(desc, str) and desc.strip():
            out[did_i] = desc
    return out

def _upsert_catalog(store: SQLServerStore, ds_df: pd.DataFrame, var_df: pd.DataFrame, ref_df: pd.DataFrame) -> dict[str,int]:
    # Datasets
    ds_count=0
    var_count=0
    ref_count=0

    # Convert pandas NaT/NaN to None for DB inserts (pyodbc + datetime2 are strict)
    ds_df = ds_df.where(pd.notna(ds_df), None)
    var_df = var_df.where(pd.notna(var_df), None)
    ref_df = ref_df.where(pd.notna(ref_df), None)

    with store.engine.begin() as conn:
        # Upsert datasets
        total_ds = len(ds_df)
        for i, (_, r) in enumerate(ds_df.iterrows(), start=1):
            if i == 1:
                print(f"[sync] upserting datasets: {total_ds}")
            if i % 200 == 0:
                print(f"[sync] datasets {i}/{total_ds}")
            conn.execute(text("""
            MERGE agent.CatalogDatasets AS tgt
            USING (SELECT :TableName AS TableName) AS src
            ON tgt.TableName = src.TableName
            WHEN MATCHED THEN UPDATE SET
                DatasetId = :DatasetId,
                ShortName = :ShortName,
                DatasetName = :DatasetName,
                Description = :Description,
                Keywords = :Keywords,
                DataSource = :DataSource,
                Distributor = :Distributor,
                Acknowledgement = :Acknowledgement,
                Make = :Make,
                Sensor = :Sensor,
                ProcessLevel = :ProcessLevel,
                StudyDomain = :StudyDomain,
                TemporalResolution = :TemporalResolution,
                SpatialResolution = :SpatialResolution,
                Units = :Units,
                Comments = :Comments,
                Regions = :Regions,
                TimeMin = :TimeMin,
                TimeMax = :TimeMax,
                LatMin = :LatMin,
                LatMax = :LatMax,
                LonMin = :LonMin,
                LonMax = :LonMax,
                DepthMin = :DepthMin,
                DepthMax = :DepthMax,
                HasDepth = :HasDepth,
                UpdatedAt = SYSUTCDATETIME()
            WHEN NOT MATCHED THEN INSERT (
                TableName, DatasetId, ShortName, DatasetName, Description, Keywords,
                DataSource, Distributor, Acknowledgement,
                Make, Sensor, ProcessLevel, StudyDomain, TemporalResolution, SpatialResolution,
                Units, Comments, Regions,
                TimeMin, TimeMax, LatMin, LatMax, LonMin, LonMax, DepthMin, DepthMax, HasDepth
            ) VALUES (
                :TableName, :DatasetId, :ShortName, :DatasetName, :Description, :Keywords,
                :DataSource, :Distributor, :Acknowledgement,
                :Make, :Sensor, :ProcessLevel, :StudyDomain, :TemporalResolution, :SpatialResolution,
                :Units, :Comments, :Regions,
                :TimeMin, :TimeMax, :LatMin, :LatMax, :LonMin, :LonMax, :DepthMin, :DepthMax, :HasDepth
            );
            """), _sql_params(dict(r)))
            ds_count += 1

        # Upsert variables
        total_var = len(var_df)
        for i, (_, r) in enumerate(var_df.iterrows(), start=1):
            if i == 1:
                print(f"[sync] upserting variables: {total_var}")
            if i % 1000 == 0:
                print(f"[sync] variables {i}/{total_var}")
            conn.execute(text("""
            MERGE agent.CatalogVariables AS tgt
            USING (SELECT :TableName AS TableName, :VarName AS VarName) AS src
            ON tgt.TableName = src.TableName AND tgt.VarName = src.VarName
            WHEN MATCHED THEN UPDATE SET
                LongName = :LongName,
                Unit = :Unit,
                Keywords = :Keywords,
                UpdatedAt = SYSUTCDATETIME()
            WHEN NOT MATCHED THEN INSERT (
                TableName, VarName, LongName, Unit, Keywords
            ) VALUES (
                :TableName, :VarName, :LongName, :Unit, :Keywords
            );
            """), _sql_params(dict(r)))
            var_count += 1

        # Rebuild references for affected datasets (simple: delete + insert)
        if len(ref_df):
            tables = sorted(set(ds_df["TableName"].dropna().astype(str)))
            # delete
            for t in tables:
                conn.execute(text("DELETE FROM agent.CatalogDatasetReferences WHERE TableName=:t"), {"t": t})
            # insert
            total_ref = len(ref_df)
            for i, (_, r) in enumerate(ref_df.iterrows(), start=1):
                if i == 1:
                    print(f"[sync] inserting references: {total_ref}")
                if i % 1000 == 0:
                    print(f"[sync] references {i}/{total_ref}")
                conn.execute(text("""
                    INSERT INTO agent.CatalogDatasetReferences(TableName, DatasetId, ReferenceId, Reference)
                    VALUES (:TableName, :DatasetId, :ReferenceId, :Reference)
                """), _sql_params(dict(r)))
                ref_count += 1

    return {"datasets": ds_count, "variables": var_count, "references": ref_count}


def _fetch_keywords(store: Store, dataset_ids: list[int]) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Fetch dataset- and variable-level keywords from Opedia tables.

    Notes about Opedia schema (as of Jan 2026, per user-provided DDL):
    - dbo.tblDatasets: primary key is [ID] (NOT Dataset_ID / Table_Name)
    - dbo.tblVariables: has [Dataset_ID], [Table_Name], [Short_Name], primary key [ID]
    - dbo.tblKeywords: has [var_ID], [keywords]
    We therefore join tblKeywords -> tblVariables on var_ID = tblVariables.ID and filter on tblVariables.Dataset_ID.
    """
    if not dataset_ids:
        return {}, {}

    ds_kw: dict[str, set[str]] = {}
    var_kw: dict[tuple[str, str], set[str]] = {}

    chunks = list(_chunked(dataset_ids, 800))
    with store.engine.begin() as conn:
        frames: list[pd.DataFrame] = []
        for ids in chunks:
            ids_sql = ",".join(str(int(i)) for i in ids)
            q = f"""
            SELECT
                COALESCE(v.Table_Name, v.DB) AS TableName,
                v.Short_Name AS VarName,
                k.keywords AS Keywords
            FROM tblVariables v
            LEFT JOIN tblKeywords k ON k.var_ID = v.ID
            WHERE v.Dataset_ID IN ({ids_sql})
            """
            frames.append(pd.read_sql_query(text(q), conn))

    if not frames:
        return {}, {}

    df = pd.concat(frames, ignore_index=True)

    # Normalize / split keyword strings (supports ';' ',' '|' etc.)
    for _, row in df.iterrows():
        table = row.get("TableName")
        var = row.get("VarName")
        kws = row.get("Keywords")
        if not isinstance(table, str) or not table:
            continue
        if not isinstance(var, str) or not var:
            # Some rows may not have a Short_Name; skip var-level keywords in that case
            var = ""

        tokens = _split_tokens(kws) if isinstance(kws, str) else []
        if not tokens:
            continue

        # dataset-level aggregation by TableName
        ds_kw.setdefault(table, set()).update(tokens)

        # variable-level mapping by (TableName, VarName)
        if var:
            var_kw.setdefault((table, var), set()).update(tokens)

    # flatten to strings for storage
    ds_kw_s: dict[str, str] = {k: ", ".join(sorted(v)) for k, v in ds_kw.items()}
    var_kw_s: dict[tuple[str, str], str] = {k: ", ".join(sorted(v)) for k, v in var_kw.items()}
    return ds_kw_s, var_kw_s

def build_variable_rows(cat: pd.DataFrame) -> pd.DataFrame:
    df=cat.copy()
    if "Table_Name" in df.columns:
        df["TableName"]=df["Table_Name"]
    if "Variable" in df.columns:
        df["VarName"]=df["Variable"]
    if "Long_Name" in df.columns:
        df["LongName"]=df["Long_Name"]
    if "Unit" in df.columns:
        df["Unit"]=df["Unit"]
    # Collect variable keywords if present
    if "Keywords" in df.columns:
        # dataset keywords; still useful
        df["VarKeywords"]=df["Keywords"]
    else:
        df["VarKeywords"]=None
    out=df[["TableName","VarName","LongName","Unit","VarKeywords"]].dropna(subset=["TableName","VarName"])
    out=out.rename(columns={"VarKeywords":"Keywords"})
    out=out.drop_duplicates(subset=["TableName","VarName"])
    # enforce agent table lengths to avoid SQL truncation errors
    out["TableName"] = out["TableName"].apply(lambda v: _truncate_str(v, MAX_TABLE_NAME))
    out["VarName"] = out["VarName"].apply(lambda v: _truncate_str(v, MAX_VAR_NAME))
    return out

def build_dataset_rows(cat: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per dataset for agent.CatalogDatasets.

    Notes:
    - `cat` is the CMAP catalog view (variable-level rows) fetched from SQL Server (Opedia tables).
    - Keywords are filled later via `_fetch_keywords()`.
    - Spatial/temporal extents are filled later via `_fetch_stats()`.
    """
    if cat is None or len(cat) == 0:
        return pd.DataFrame(columns=[
            "TableName","DatasetId","ShortName","DatasetName","Description","Keywords",
            "DataSource","Distributor","Acknowledgement",
            "Make","Sensor","ProcessLevel","StudyDomain","TemporalResolution","SpatialResolution",
            "Units","Comments","Regions","HasDepth",
        ])

    df = cat.copy()

    # normalize column names
    if "Table_Name" in df.columns and "TableName" not in df.columns:
        df = df.rename(columns={"Table_Name": "TableName"})
    if "Dataset_ID" in df.columns and "DatasetId" not in df.columns:
        df = df.rename(columns={"Dataset_ID": "DatasetId"})
    if "Dataset_Name" in df.columns and "ShortName" not in df.columns:
        # in CMAP DB, Dataset_Name is typically a short name (e.g., "ALOHA_O2toAr")
        df = df.rename(columns={"Dataset_Name": "ShortName"})
    if "Dataset_Long_Name" in df.columns and "DatasetName" not in df.columns:
        df = df.rename(columns={"Dataset_Long_Name": "DatasetName"})
    if "Data_Source" in df.columns and "DataSource" not in df.columns:
        df = df.rename(columns={"Data_Source": "DataSource"})
    if "Process_Level" in df.columns and "ProcessLevel" not in df.columns:
        df = df.rename(columns={"Process_Level": "ProcessLevel"})
    if "Study_Domain" in df.columns and "StudyDomain" not in df.columns:
        df = df.rename(columns={"Study_Domain": "StudyDomain"})
    if "Temporal_Resolution" in df.columns and "TemporalResolution" not in df.columns:
        df = df.rename(columns={"Temporal_Resolution": "TemporalResolution"})
    if "Spatial_Resolution" in df.columns and "SpatialResolution" not in df.columns:
        df = df.rename(columns={"Spatial_Resolution": "SpatialResolution"})

    # fallbacks
    if "DatasetName" not in df.columns and "Dataset_Long_Name" in df.columns:
        df["DatasetName"] = df["Dataset_Long_Name"]
    if "ShortName" not in df.columns:
        # fallback: strip leading "tbl" when it looks like a table name
        df["ShortName"] = df.get("TableName", "").astype(str).str.replace(r"^tbl", "", regex=True)

    # Helper to aggregate distinct non-null strings.
    def _agg_unique(series: pd.Series) -> str | None:
        vals = [str(v).strip() for v in series.dropna().tolist() if str(v).strip() and str(v).strip().lower() != "nan"]
        if not vals:
            return None
        # preserve stable order
        seen = set()
        out = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return ", ".join(out)

    # Build one row per dataset based on (TableName, DatasetId)
    group_cols = []
    if "TableName" in df.columns:
        group_cols.append("TableName")
    if "DatasetId" in df.columns:
        group_cols.append("DatasetId")
    if not group_cols:
        # can't group reliably; return empty
        return pd.DataFrame()

    g = df.groupby(group_cols, dropna=False)

    out = pd.DataFrame({
        "ShortName": g["ShortName"].apply(lambda s: next((x for x in s.dropna().tolist() if str(x).strip()), None)) if "ShortName" in df.columns else None,
        "DatasetName": g["DatasetName"].apply(lambda s: next((x for x in s.dropna().tolist() if str(x).strip()), None)) if "DatasetName" in df.columns else None,
        "Description": g["Description"].apply(lambda s: next((x for x in s.dropna().tolist() if str(x).strip()), None)) if "Description" in df.columns else None,
        "DataSource": g["DataSource"].apply(lambda s: next((x for x in s.dropna().tolist() if str(x).strip()), None)) if "DataSource" in df.columns else None,
        "Distributor": g["Distributor"].apply(lambda s: next((x for x in s.dropna().tolist() if str(x).strip()), None)) if "Distributor" in df.columns else None,
        "Acknowledgement": g["Acknowledgement"].apply(lambda s: next((x for x in s.dropna().tolist() if str(x).strip()), None)) if "Acknowledgement" in df.columns else None,
        "Make": g["Make"].apply(_agg_unique) if "Make" in df.columns else None,
        "Sensor": g["Sensor"].apply(_agg_unique) if "Sensor" in df.columns else None,
        "ProcessLevel": g["ProcessLevel"].apply(_agg_unique) if "ProcessLevel" in df.columns else None,
        "StudyDomain": g["StudyDomain"].apply(_agg_unique) if "StudyDomain" in df.columns else None,
        "TemporalResolution": g["TemporalResolution"].apply(_agg_unique) if "TemporalResolution" in df.columns else None,
        "SpatialResolution": g["SpatialResolution"].apply(_agg_unique) if "SpatialResolution" in df.columns else None,
        "Units": g["Unit"].apply(_agg_unique) if "Unit" in df.columns else None,
        "Comments": None,
        "Regions": None,
        # keywords filled later
        "Keywords": None,
    }).reset_index()

    # HasDepth: prefer variable metadata if present; otherwise leave null and let stats merge handle it.
    if "Has_Depth" in df.columns:
        has_depth = g["Has_Depth"].apply(lambda s: bool(pd.Series(s).fillna(False).astype(bool).any())).reset_index(drop=True)
        out["HasDepth"] = has_depth
    elif "HasDepth" in df.columns:
        has_depth = g["HasDepth"].apply(lambda s: bool(pd.Series(s).fillna(False).astype(bool).any())).reset_index(drop=True)
        out["HasDepth"] = has_depth
    else:
        out["HasDepth"] = None

    # Merge extents from stats (time/space/depth) if available
    if stats is not None and len(stats) > 0:
        st = stats.copy()
        st = st.rename(
            columns={
                "Dataset_ID": "DatasetId",
                "Time_Min": "TimeMin",
                "Time_Max": "TimeMax",
                "Lat_Min": "LatMin",
                "Lat_Max": "LatMax",
                "Lon_Min": "LonMin",
                "Lon_Max": "LonMax",
                "Depth_Min": "DepthMin",
                "Depth_Max": "DepthMax",
            }
        )

        # Parse datetimes -> tz-naive UTC (pyodbc/FreeTDS often dislikes tz-aware values)
        for c in ["TimeMin", "TimeMax"]:
            if c in st.columns:
                ts = pd.to_datetime(st[c], errors="coerce", utc=True)
                st[c] = ts.dt.tz_convert(None)

        # Parse numeric extents
        for c in ["LatMin", "LatMax", "LonMin", "LonMax", "DepthMin", "DepthMax"]:
            if c in st.columns:
                st[c] = pd.to_numeric(st[c], errors="coerce")

        if "HasDepth" not in st.columns:
            st["HasDepth"] = st[["DepthMin", "DepthMax"]].notna().any(axis=1)

        st = st[
            [
                "DatasetId",
                "TimeMin",
                "TimeMax",
                "LatMin",
                "LatMax",
                "LonMin",
                "LonMax",
                "DepthMin",
                "DepthMax",
                "HasDepth",
            ]
        ].drop_duplicates(subset=["DatasetId"])

        out = out.merge(st, on="DatasetId", how="left", suffixes=("_Cat", ""))

        # Prefer stats-derived HasDepth; fall back to catalog-derived
        if "HasDepth_Cat" in out.columns:
            out["HasDepth"] = out["HasDepth"].combine_first(out["HasDepth_Cat"])
            out = out.drop(columns=["HasDepth_Cat"])

    # Ensure ALL required columns exist (even if None) so bindparams are always satisfied
    required_cols = [
        "TableName",
        "DatasetId",
        "ShortName",
        "DatasetName",
        "Description",
        "Keywords",
        "DataSource",
        "Distributor",
        "Acknowledgement",
        "Make",
        "Sensor",
        "ProcessLevel",
        "StudyDomain",
        "TemporalResolution",
        "SpatialResolution",
        "Units",
        "Comments",
        "Regions",
        "TimeMin",
        "TimeMax",
        "LatMin",
        "LatMax",
        "LonMin",
        "LonMax",
        "DepthMin",
        "DepthMax",
        "HasDepth",
    ]
    for c in required_cols:
        if c not in out.columns:
            out[c] = None

    # Normalize types (avoid pandas NA leaking into ODBC bindings later)
    if "DatasetId" in out.columns:
        out["DatasetId"] = pd.to_numeric(out["DatasetId"], errors="coerce").astype("Int64")
    if "HasDepth" in out.columns:
        out["HasDepth"] = out["HasDepth"].astype("boolean")

    # enforce agent table lengths to avoid SQL truncation errors
    out["TableName"] = out["TableName"].apply(lambda v: _truncate_str(v, MAX_TABLE_NAME))
    out["ShortName"] = out["ShortName"].apply(lambda v: _truncate_str(v, MAX_DATASET_SHORTNAME))

    return out


def build_reference_rows(ds_rows: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    if refs is None or len(refs)==0:
        return pd.DataFrame(columns=["TableName","DatasetId","ReferenceId","Reference"])
    # Map dataset id -> table
    m = ds_rows[["DatasetId","TableName"]].dropna().drop_duplicates()
    refs = refs.merge(m, left_on="Dataset_ID", right_on="DatasetId", how="left")
    refs = refs.rename(columns={"Reference_ID":"ReferenceId","Dataset_ID":"DatasetId","Reference":"Reference"})
    refs = refs.dropna(subset=["TableName"])
    return refs[["TableName","DatasetId","ReferenceId","Reference"]]

def main():
    ap = argparse.ArgumentParser(
        description="Sync CMAP catalog metadata into agent cache tables (source: SQL Server / Opedia tables)."
    )
    # Backward-compat CLI flags kept for compatibility with older scripts; catalog is sourced from SQL Server.
    ap.add_argument("--base-url", type=str, default="https://simonscmap.dev", help="DEPRECATED (ignored).")
    ap.add_argument("--api-key", type=str, default=None, help="DEPRECATED (ignored).")
    args = ap.parse_args()

    store = SQLServerStore.from_env()

    cat = _fetch_catalog(store)
    if cat is None or len(cat) == 0:
        raise SystemExit(
            "Catalog query returned 0 rows from SQL Server. "
            "Check the SQL connection (agent DB env vars) and ensure Opedia tables are accessible "
            "(tblVariables, tblDataset_Stats, tblDatasets, etc.)."
        )

    dataset_ids = (
        sorted(set(int(x) for x in cat["Dataset_ID"].dropna().unique()))
        if "Dataset_ID" in cat.columns
        else []
    )
    stats = _fetch_stats(store, dataset_ids)
    ds_desc = _fetch_dataset_descriptions(store, dataset_ids)
    refs = _fetch_references(store, dataset_ids)

    ds_rows = build_dataset_rows(cat, stats)
    if ds_desc and "DatasetId" in ds_rows.columns:
        # Prefer the authoritative long-form dataset description from tblDatasets
        ds_rows["Description"] = ds_rows["DatasetId"].map(ds_desc).fillna(ds_rows["Description"])
    var_rows = build_variable_rows(cat)

    # Prefer keywords from Opedia.tblKeywords to avoid confusing/misleading keyword fields in the catalog view.
    ds_kw_map, var_kw_map = _fetch_keywords(store, dataset_ids)
    if ds_kw_map and "TableName" in ds_rows.columns:
        ds_rows["Keywords"] = ds_rows["TableName"].map(ds_kw_map)
    if var_kw_map and all(c in var_rows.columns for c in ("TableName", "VarName")):
        keys = list(zip(var_rows["TableName"].astype(str), var_rows["VarName"].astype(str)))
        var_rows["Keywords"] = [var_kw_map.get(k) for k in keys]

    ref_rows = build_reference_rows(ds_rows, refs)

    counts = _upsert_catalog(store, ds_rows, var_rows, ref_rows)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    import json
    main()