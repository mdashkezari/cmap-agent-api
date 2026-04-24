"""sql_validator — pure, dependency-free validation for catalog.query_metadata.

Keeping this module free of SQLAlchemy / pydantic / network imports means
the validator can be exhaustively unit-tested in isolation.  The enforcement
rules implemented here are stronger than the v202/v203 version:

  1.  SELECT-only; no DDL/DML keywords.
  2.  Table references must be in ``ALLOWED_TABLES`` or ``ALLOWED_FUNCTIONS``.
  3.  Known data-table names and data-table prefixes are blocked.
  4.  TOP N (≤ ``MAX_ROWS``) must be present.
  5.  ``SELECT *`` and ``<alias>.*`` projections are rejected — explicit
      column lists are required.  ``COUNT(*)`` and similar aggregates are
      unaffected.
  6.  ``User_ID`` / ``UserID`` columns may appear in WHERE / JOIN / GROUP BY
      but MUST NOT appear in SELECT output lists.  Rejected if found.
  7.  Queries that reference ``tblCollections`` / ``tblCollection_Datasets``
      / ``tblCollection_Follows`` must contain a ``Private = 0`` predicate.
  8.  Queries that reference ``tblNews`` / ``tblNews_Datasets`` must
      contain a ``view_status = 3`` predicate.

Rules 5–8 were prompt-only before v204; they are now refused server-side.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Whitelists
# ---------------------------------------------------------------------------

# Authoritative inventory of metadata tables (dbo schema), derived from the
# CMAP "Database Metadata Tables" bundle plus the tables directly referenced
# by the udfCatalog() function.  Not included by policy:
#   - tblUsers      — PII (user IDs, emails).
#   - tblOrganism   — supporting lookup table whose DDL is not documented in
#                     the authoritative bundle; skip until schema is supplied.
ALLOWED_TABLES: frozenset[str] = frozenset({
    # Core catalog
    "tblDatasets",
    "tblVariables",
    "tblKeywords",
    "tblDataset_Stats",
    "tblDataset_References",
    "tblDataset_Cruises",
    "tblDataset_Programs",
    "tblDataset_Regions",
    "tblDataset_Servers",
    # Unstructured JSON metadata (joined by udfCatalog)
    "tblDatasets_JSON_Metadata",
    "tblVariables_JSON_Metadata",
    # Cruises
    "tblCruise",
    "tblCruise_Keywords",
    "tblCruise_Scientists",
    "tblCruise_Series",
    "tblCruise_Regions",
    "tblCruise_Trajectory",
    # Programs & regions
    "tblPrograms",
    "tblRegions",
    # Collections
    "tblCollections",
    "tblCollection_Datasets",
    "tblCollection_Follows",
    # News
    "tblNews",
    "tblNews_Datasets",
    # Lookup dimensions
    "tblMakes",
    "tblSensors",
    "tblProcess_Stages",
    "tblTemporal_Resolutions",
    "tblSpatial_Resolutions",
    "tblStudy_Domains",
})

# Functions that may also appear after FROM/JOIN.
ALLOWED_FUNCTIONS: frozenset[str] = frozenset({
    "udfCatalog",
})

# Prefixes that identify data tables — blocked regardless of the whitelist.
DATA_TABLE_PREFIXES: tuple[str, ...] = (
    "tblSSTd", "tblSST_", "tblModis", "tblERA", "tblWOA",
    "tblPisces", "tblDarwin", "tblArgo", "tblSeaFlow",
    "tblGLODAP", "tblGEBCO", "tblInflux", "tblKOK", "tblKM",
    "tblMGL", "tblTN", "tblHOT", "tblFalkor", "tblGradients",
)

# Tables carrying a Private column that must be filtered.
PRIVATE_TABLES: frozenset[str] = frozenset({
    "tblCollections",
    "tblCollection_Datasets",
    "tblCollection_Follows",
})

# Tables carrying view_status that must be filtered to published rows.
NEWS_TABLES: frozenset[str] = frozenset({
    "tblNews",
    "tblNews_Datasets",
})

MAX_ROWS: int = 200


# ---------------------------------------------------------------------------
# Regexes — compiled once at import time
# ---------------------------------------------------------------------------

_SELECT_START = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

_BLOCKED_KEYWORDS = re.compile(
    # DDL/DML verbs — strict word-boundary match.
    r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|EXECUTE|"
    r"TRUNCATE|MERGE|GRANT|REVOKE)\b"
    # Extended / system stored procedure prefixes — must be followed by a
    # name character, since the `\b` after `_` does not trigger (`_` is a
    # word char), which would otherwise silently miss e.g. ``xp_cmdshell``.
    r"|\b(?:xp_|sp_)\w+",
    re.IGNORECASE,
)

# Captures the identifier after FROM/JOIN, optional dbo. prefix, allowing
# bracketed names and optional trailing "(" for function invocations.
_TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN)\s+(?:\[?dbo\]?\.)?\[?(\w+)\]?",
    re.IGNORECASE,
)

_TOP_N = re.compile(r"\bTOP\s+(\d+)\b", re.IGNORECASE)

# Wildcard projections.
_ALIAS_STAR = re.compile(r"\b\w+\s*\.\s*\*")
_WILDCARD_START_OR_COMMA = re.compile(r"(?:^|,)\s*\*")

# User identity columns that must not appear in a SELECT projection.
# Matches UserID or User_ID (with or without bracket quoting).
_USERID_TOKEN = re.compile(r"\bUser_?ID\b", re.IGNORECASE)

# Strip a leading TOP N clause from a projection body.  The extractor in
# ``extract_select_projections`` returns text starting immediately after the
# ``SELECT`` keyword, which may include ``TOP N`` before the column list.
_TOP_N_PREFIX = re.compile(r"^\s*TOP\s+\d+\s*", re.IGNORECASE)

# Matches a single function-call expression whose body contains no further
# parens.  Used iteratively to peel nested function calls (``MAX(COUNT(*))``)
# without also removing naked subquery parens such as ``(SELECT ...)``, which
# are NOT preceded by a bareword identifier.
_FUNC_CALL = re.compile(r"\b\w+\s*\([^()]*\)")


def _strip_function_calls(s: str) -> str:
    """Remove all function-call expressions from *s*, leaving subqueries."""
    prev = None
    while s != prev:
        prev = s
        s = _FUNC_CALL.sub("", s)
    return s


def _projection_body(proj: str) -> str:
    """Return the column-list portion of a SELECT projection.

    Strips the optional ``TOP N`` prefix, then peels every function-call
    expression so that aggregates like ``COUNT(DISTINCT User_ID)`` do not
    masquerade as User_ID projections.  Naked subquery parentheses — which
    would expose a scalar result to the caller — are deliberately retained.
    """
    s = _TOP_N_PREFIX.sub("", proj)
    return _strip_function_calls(s)

# Privacy / publication predicates.  Tolerant of optional bracket quoting
# around the column name.  They need only appear somewhere in the SQL body.
_PRIVATE_FILTER = re.compile(r"\bPrivate\s*\]?\s*=\s*0\b", re.IGNORECASE)
_NEWS_FILTER = re.compile(r"\bview_status\s*\]?\s*=\s*3\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_table_names(sql: str) -> list[str]:
    """Return all identifiers that follow FROM/JOIN in *sql*.

    Includes function-call targets such as ``udfCatalog``.  De-duplication
    is NOT performed — callers that care about duplicates should coerce the
    result to a set.
    """
    return [m.group(1) for m in _TABLE_REF.finditer(sql)]


def extract_select_projections(sql: str) -> list[str]:
    """Return every top-level ``SELECT ... FROM`` projection substring in *sql*.

    The extractor walks character by character to track parenthesis depth,
    so subqueries, CTEs, and ``UNION`` branches are each surfaced as their
    own projection segment.  If no matching FROM is found for a SELECT, the
    remainder of the string is returned (best effort).
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        m = re.search(r"\bSELECT\b", sql[i:], re.IGNORECASE)
        if not m:
            break
        start = i + m.end()
        depth = 0
        j = start
        proj_end = -1
        while j < n:
            c = sql[j]
            if c == "(":
                depth += 1
                j += 1
                continue
            if c == ")":
                # This ")" may close an outer scope — stop the scan; the
                # projection ends here even without an explicit FROM.
                if depth == 0:
                    proj_end = j
                    break
                depth -= 1
                j += 1
                continue
            if depth == 0:
                fm = re.match(r"FROM\b", sql[j:], re.IGNORECASE)
                if fm:
                    proj_end = j
                    break
            j += 1
        if proj_end == -1:
            proj_end = n
        out.append(sql[start:proj_end])
        i = proj_end + 1
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_sql(sql: str, data_table_names: frozenset[str]) -> str | None:
    """Validate *sql* against the v204 policy.

    Returns ``None`` when the query is permitted, or a human-readable error
    string explaining why it was rejected.  The error strings are intended
    to be returned verbatim to the LLM so that it can amend its next query.

    Args:
        sql: The raw SQL submitted by the LLM.
        data_table_names: Names of data tables (from the catalog cache) that
            must be blocked even if they appear to match a metadata pattern.
    """
    sql_stripped = sql.strip()

    # 1. SELECT-only.
    if not _SELECT_START.match(sql_stripped):
        return "Only SELECT statements are permitted."

    # 2. Blocked keywords (DDL/DML/exec).
    if _BLOCKED_KEYWORDS.search(sql_stripped):
        return "Query contains prohibited SQL keywords."

    # 3. Table whitelisting.
    referenced = extract_table_names(sql_stripped)
    for tbl in referenced:
        if tbl in data_table_names:
            return (
                f"'{tbl}' is a data table. Use pycmap / cmap.space_time to "
                f"query data. This tool is for metadata tables only."
            )
        if any(tbl.lower().startswith(p.lower()) for p in DATA_TABLE_PREFIXES):
            return (
                f"'{tbl}' appears to be a data table. "
                f"This tool is for metadata tables only."
            )
        if tbl in ALLOWED_FUNCTIONS:
            continue
        if tbl not in ALLOWED_TABLES:
            return (
                f"Table '{tbl}' is not in the allowed metadata table list. "
                f"Allowed: {', '.join(sorted(ALLOWED_TABLES))}."
            )

    # 4. TOP N required and bounded.
    top = _TOP_N.search(sql_stripped)
    if not top:
        return "Query must include TOP N (e.g. TOP 50) to limit result size."
    if int(top.group(1)) > MAX_ROWS:
        return f"TOP N cannot exceed {MAX_ROWS}. Please reduce the limit."

    # 5 & 6. Per-projection checks.
    projections = extract_select_projections(sql_stripped)
    for proj in projections:
        body = _projection_body(proj)
        if _WILDCARD_START_OR_COMMA.search(body):
            return (
                "SELECT * is not permitted — list the columns explicitly. "
                "Aggregates like COUNT(*) are fine."
            )
        if _ALIAS_STAR.search(body):
            return (
                "Alias.* projections (e.g. t.*) are not permitted — list "
                "the columns explicitly."
            )
        if _USERID_TOKEN.search(body):
            return (
                "User_ID / UserID columns may appear in WHERE or JOIN "
                "clauses but must not be selected into output. Remove them "
                "from the SELECT list (aggregating inside COUNT / DISTINCT "
                "is allowed)."
            )

    # 7. Private=0 when collection tables are referenced.
    ref_set = {t for t in referenced}
    if ref_set & PRIVATE_TABLES:
        if not _PRIVATE_FILTER.search(sql_stripped):
            return (
                "Queries that reference tblCollections, tblCollection_Datasets, "
                "or tblCollection_Follows must include a Private = 0 "
                "predicate (join to tblCollections if necessary). "
                "Private = 1 rows may not be exposed."
            )

    # 8. view_status=3 when news tables are referenced.
    if ref_set & NEWS_TABLES:
        if not _NEWS_FILTER.search(sql_stripped):
            return (
                "Queries that reference tblNews or tblNews_Datasets must "
                "include a view_status = 3 predicate (join to tblNews if "
                "necessary). Other view_status values are drafts."
            )

    return None
