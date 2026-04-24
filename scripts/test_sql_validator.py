#!/usr/bin/env python3
"""test_sql_validator.py — unit tests for the v204 query_metadata validator.

These tests exercise ``cmap_agent.tools.sql_validator.validate_sql``
directly.  The validator module has no external dependencies, so the
tests can run in any environment (no SQL Server driver, no OpenAI key,
no Qdrant).

Covered behaviours
------------------
  1. Whitelist acceptance — every authoritative metadata table is allowed.
  2. Whitelist rejection — disallowed tables (tblUsers, invented tables).
  3. Data-table prefix rejection — tblSSTd_* etc.
  4. TOP N enforcement — required, bounded, numeric.
  5. SELECT * and alias.* refusal; COUNT(*) allowed.
  6. User_ID / UserID refused in SELECT but allowed elsewhere.
  7. Private = 0 enforcement on tblCollections / linked tables.
  8. view_status = 3 enforcement on tblNews / tblNews_Datasets.
  9. udfCatalog() is queryable.

Usage
-----
    python scripts/test_sql_validator.py
"""
from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Runner scaffolding
# ---------------------------------------------------------------------------

def _run(cases: list[tuple[str, str, bool, str]]) -> tuple[int, int]:
    """Run a list of (description, sql, should_pass, fragment) cases."""
    from cmap_agent.tools.sql_validator import validate_sql

    data_tables = frozenset({"tblChl_OC_CCI", "tblSST_AVHRR_OI_NRT"})

    passed = 0
    failed = 0
    for desc, sql, should_pass, fragment in cases:
        err = validate_sql(sql, data_tables)
        got_pass = err is None
        ok = got_pass == should_pass
        if ok and not should_pass and fragment:
            # Verify the error mentions the expected fragment — guards
            # against drift in error messages sending misleading signals.
            if fragment.lower() not in (err or "").lower():
                ok = False
                print(f"  [FAIL] {desc}")
                print(f"         expected error to mention {fragment!r}")
                print(f"         got error: {err!r}")
                failed += 1
                continue
        if ok:
            print(f"  [PASS] {desc}")
            passed += 1
        else:
            print(f"  [FAIL] {desc}")
            print(f"         sql: {sql}")
            print(f"         expected_pass={should_pass}, got_pass={got_pass}")
            print(f"         err: {err!r}")
            failed += 1
    return passed, failed


# ---------------------------------------------------------------------------
# Test cases — (description, sql, should_pass, expected_error_fragment)
# ---------------------------------------------------------------------------

WHITELIST_CASES: list[tuple[str, str, bool, str]] = [
    (
        "tblDatasets accepted",
        "SELECT TOP 10 Dataset_Name FROM dbo.tblDatasets",
        True, "",
    ),
    (
        "tblCollection_Follows accepted (v204 addition)",
        "SELECT TOP 10 Follow_Date FROM dbo.tblCollection_Follows f "
        "JOIN dbo.tblCollections c ON f.Collection_ID = c.Collection_ID "
        "WHERE c.Private = 0",
        True, "",
    ),
    (
        "tblCruise_Trajectory accepted (v204 addition)",
        "SELECT TOP 10 time, lat, lon FROM dbo.tblCruise_Trajectory "
        "WHERE Cruise_ID = 42",
        True, "",
    ),
    (
        "tblDatasets_JSON_Metadata accepted (v204 addition)",
        "SELECT TOP 5 Dataset_ID FROM dbo.tblDatasets_JSON_Metadata",
        True, "",
    ),
    (
        "tblStudy_Domains accepted (kept from v203)",
        "SELECT TOP 20 Study_Domain FROM dbo.tblStudy_Domains",
        True, "",
    ),
    (
        "udfCatalog() accepted as a function source",
        "SELECT TOP 5 Variable, Table_Name FROM dbo.udfCatalog()",
        True, "",
    ),
    (
        "tblUsers rejected",
        "SELECT TOP 5 Email FROM dbo.tblUsers",
        False, "not in the allowed",
    ),
    (
        "invented table rejected",
        "SELECT TOP 5 foo FROM dbo.tblSomethingMadeUp",
        False, "not in the allowed",
    ),
    (
        "data-table prefix (tblSSTd_*) rejected",
        "SELECT TOP 5 lat, lon, time, sst FROM dbo.tblSSTd_test",
        False, "data table",
    ),
    (
        "data-table from catalog cache rejected",
        "SELECT TOP 5 lat FROM dbo.tblChl_OC_CCI",
        False, "data table",
    ),
]


TOPN_CASES: list[tuple[str, str, bool, str]] = [
    (
        "TOP N required",
        "SELECT Dataset_Name FROM dbo.tblDatasets",
        False, "TOP",
    ),
    (
        "TOP N cannot exceed MAX_ROWS",
        "SELECT TOP 5000 Dataset_Name FROM dbo.tblDatasets",
        False, "TOP",
    ),
    (
        "TOP 200 (boundary) accepted",
        "SELECT TOP 200 Dataset_Name FROM dbo.tblDatasets",
        True, "",
    ),
]


STAR_CASES: list[tuple[str, str, bool, str]] = [
    (
        "SELECT * rejected",
        "SELECT TOP 10 * FROM dbo.tblDatasets",
        False, "SELECT *",
    ),
    (
        "SELECT t.* rejected",
        "SELECT TOP 10 t.* FROM dbo.tblDatasets t",
        False, "Alias.*",
    ),
    (
        "SELECT COUNT(*) accepted",
        "SELECT TOP 10 COUNT(*) AS n FROM dbo.tblDatasets",
        True, "",
    ),
    (
        "SELECT a * b (multiplication) accepted — not a wildcard",
        "SELECT TOP 10 Downloads * Views AS product "
        "FROM dbo.tblCollections WHERE Private = 0",
        True, "",
    ),
    (
        "SELECT COUNT(DISTINCT User_ID) accepted (inside aggregate)",
        "SELECT TOP 10 c.Collection_Name, COUNT(DISTINCT f.User_ID) AS followers "
        "FROM dbo.tblCollection_Follows f "
        "JOIN dbo.tblCollections c ON f.Collection_ID = c.Collection_ID "
        "WHERE c.Private = 0 GROUP BY c.Collection_Name",
        True, "",
    ),
]


USERID_CASES: list[tuple[str, str, bool, str]] = [
    (
        "User_ID in SELECT list rejected",
        "SELECT TOP 10 User_ID, Collection_Name FROM dbo.tblCollections "
        "WHERE Private = 0",
        False, "User_ID",
    ),
    (
        "UserID (no underscore, tblNews) in SELECT list rejected",
        "SELECT TOP 10 UserID, headline FROM dbo.tblNews "
        "WHERE view_status = 3",
        False, "User",
    ),
    (
        "bracketed [User_ID] in SELECT list rejected",
        "SELECT TOP 10 [User_ID], Collection_Name FROM dbo.tblCollections "
        "WHERE Private = 0",
        False, "User_ID",
    ),
    (
        "User_ID in WHERE (not SELECT) accepted",
        "SELECT TOP 10 Collection_Name FROM dbo.tblCollections "
        "WHERE Private = 0 AND User_ID = 42",
        True, "",
    ),
    (
        "User_ID in JOIN (not SELECT) accepted",
        "SELECT TOP 10 f.Follow_Date, c.Collection_Name "
        "FROM dbo.tblCollection_Follows f "
        "JOIN dbo.tblCollections c ON f.User_ID = c.User_ID "
        "WHERE c.Private = 0",
        True, "",
    ),
    (
        "User_ID in GROUP BY (not SELECT) accepted",
        "SELECT TOP 10 COUNT(*) AS n FROM dbo.tblCollection_Follows f "
        "JOIN dbo.tblCollections c ON f.Collection_ID = c.Collection_ID "
        "WHERE c.Private = 0 GROUP BY f.User_ID",
        True, "",
    ),
]


PRIVACY_CASES: list[tuple[str, str, bool, str]] = [
    (
        "tblCollections without Private=0 rejected",
        "SELECT TOP 10 Collection_Name FROM dbo.tblCollections",
        False, "Private = 0",
    ),
    (
        "tblCollection_Datasets without Private=0 rejected",
        "SELECT TOP 10 Dataset_Short_Name FROM dbo.tblCollection_Datasets",
        False, "Private = 0",
    ),
    (
        "tblCollection_Follows without Private=0 rejected",
        "SELECT TOP 10 Follow_Date FROM dbo.tblCollection_Follows",
        False, "Private = 0",
    ),
    (
        "tblCollections with Private=0 accepted",
        "SELECT TOP 10 Collection_Name FROM dbo.tblCollections WHERE Private = 0",
        True, "",
    ),
    (
        "tblCollection_Datasets with joined Private=0 accepted",
        "SELECT TOP 10 cd.Dataset_Short_Name "
        "FROM dbo.tblCollection_Datasets cd "
        "JOIN dbo.tblCollections c ON cd.Collection_ID = c.Collection_ID "
        "WHERE c.Private = 0",
        True, "",
    ),
    (
        "bracketed [Private] = 0 accepted",
        "SELECT TOP 10 Collection_Name FROM dbo.tblCollections "
        "WHERE [Private] = 0",
        True, "",
    ),
    (
        "tblPrograms (no privacy column) does not require Private=0",
        "SELECT TOP 10 Program_Name FROM dbo.tblPrograms",
        True, "",
    ),
]


NEWS_CASES: list[tuple[str, str, bool, str]] = [
    (
        "tblNews without view_status=3 rejected",
        "SELECT TOP 10 headline FROM dbo.tblNews",
        False, "view_status = 3",
    ),
    (
        "tblNews_Datasets without view_status=3 rejected",
        "SELECT TOP 10 News_ID, Dataset_ID FROM dbo.tblNews_Datasets",
        False, "view_status = 3",
    ),
    (
        "tblNews with view_status=3 accepted",
        "SELECT TOP 10 headline, publish_date FROM dbo.tblNews "
        "WHERE view_status = 3 ORDER BY publish_date DESC",
        True, "",
    ),
    (
        "tblNews_Datasets joined with tblNews + view_status=3 accepted",
        "SELECT TOP 10 n.headline, nd.Dataset_ID "
        "FROM dbo.tblNews_Datasets nd "
        "JOIN dbo.tblNews n ON nd.News_ID = n.ID "
        "WHERE n.view_status = 3",
        True, "",
    ),
]


BASIC_CASES: list[tuple[str, str, bool, str]] = [
    (
        "non-SELECT rejected",
        "INSERT INTO dbo.tblDatasets (Dataset_Name) VALUES ('x')",
        False, "SELECT",
    ),
    (
        "blocked keyword (DROP) rejected",
        "SELECT TOP 10 Dataset_Name FROM dbo.tblDatasets; DROP TABLE foo",
        False, "prohibited",
    ),
    (
        "blocked keyword (xp_) rejected",
        "SELECT TOP 10 Dataset_Name FROM dbo.tblDatasets WHERE xp_cmdshell()",
        False, "prohibited",
    ),
]


SUITES: list[tuple[str, list]] = [
    ("basic",      BASIC_CASES),
    ("whitelist",  WHITELIST_CASES),
    ("top n",      TOPN_CASES),
    ("wildcards",  STAR_CASES),
    ("user id",    USERID_CASES),
    ("privacy",    PRIVACY_CASES),
    ("news",       NEWS_CASES),
]


def main() -> int:
    total_p, total_f = 0, 0
    for name, cases in SUITES:
        print(f"\n== {name} ==")
        p, f = _run(cases)
        total_p += p
        total_f += f
    print(f"\nSummary: {total_p} passed, {total_f} failed")
    return 0 if total_f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
