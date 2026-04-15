"""reference_download — download scientific references into the reference bank.

For each dataset in tblDataset_References, attempts to fetch the full text of
each reference (PDF, HTML, GitHub README) and saves it to:

    notrack/reference_bank/{Dataset_Short_Name}/{slug}.{ext}

The reference bank is a human-editable staging area. Documents can also be
placed there manually — the KB sync will ingest whatever it finds, regardless
of origin.

Sources attempted (in order):
  1. DOI → Zenodo API (for zenodo DOIs — fetches actual files or description)
  2. DOI → Unpaywall API to find open-access PDF URL
  3. DOI → Europe PMC (strong coverage for marine/environmental science journals)
  4. DOI → direct resolve (catches remaining open PDFs)
  5. Explicit URLs in reference text
  6. GitHub URLs → fetch README.md and other root .md files

Entry point: cmap-agent-download-refs  (see pyproject.toml [project.scripts])

Usage:
    cmap-agent-download-refs
    cmap-agent-download-refs --dataset GRUMP
    cmap-agent-download-refs --dataset GRUMP --dataset HOT_Bottle_ALOHA
    cmap-agent-download-refs --limit 10
    cmap-agent-download-refs --dry-run
"""
from __future__ import annotations

import argparse
import logging
import re
import time
import unicodedata
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from urllib.parse import urlparse

import requests
from sqlalchemy import text

from cmap_agent.storage.sqlserver import SQLServerStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNPAYWALL_EMAIL  = "mdehghan@uw.edu"
UNPAYWALL_BASE   = "https://api.unpaywall.org/v2"
EUROPEPMC_BASE   = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ZENODO_API_BASE  = "https://zenodo.org/api/records"

REQUEST_TIMEOUT     = 30
RETRY_SLEEP         = 2
MAX_RETRIES         = 2
INTER_REQUEST_SLEEP = 1.2
MAX_FILE_BYTES      = 50_000_000  # 50 MB

GITHUB_RAW_BASE = "https://raw.githubusercontent.com"

# DOI_RE: allows dots inside the path (needed for zenodo.NNNNN),
# strips trailing punctuation in post-processing
DOI_RE    = re.compile(r'\b(10\.\d{4,9}/[^\s\)\]>,\"\']+)', re.I)
URL_RE    = re.compile(r'https?://[^\s\)\]>,\"\']+', re.I)
GITHUB_RE = re.compile(
    r'https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s\)\]>,\"\']+)', re.I
)
# Matches both zenodo.org URLs and zenodo in DOI paths (10.5281/zenodo.NNNN)
ZENODO_RE = re.compile(r'zenodo', re.I)

# Publisher domains where HTML responses are likely paywalled
_PUBLISHER_DOMAINS = {
    "nature.com", "springer.com", "wiley.com", "elsevier.com",
    "tandfonline.com", "oup.com", "science.org", "cell.com",
    "sagepub.com", "pnas.org", "acs.org", "rsc.org",
    "annualreviews.org", "royalsocietypublishing.org",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 80) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s\-\.]", "", text)
    text = re.sub(r"[\s\-]+", "_", text).strip("_.")
    return text[:max_len] or "ref"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "CMAP-Agent-RefDownloader/1.0 (mailto:mdehghan@uw.edu)"
    })
    return s


def _get(session: requests.Session, url: str, stream: bool = False,
         params: dict | None = None) -> requests.Response | None:
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT, stream=stream,
                            allow_redirects=True, params=params)
            if r.status_code == 200:
                return r
            if r.status_code in (404, 410):
                log.debug("  404/410 at %s", url)
                return None
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 15))
                log.warning("  429 rate limit — sleeping %ds", wait)
                time.sleep(wait)
            else:
                log.debug("  HTTP %d at %s", r.status_code, url)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP)
        except requests.RequestException as e:
            log.debug("  Request error at %s: %s", url, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
    return None


def _save(dest: Path, content: bytes, overwrite: bool = False) -> bool:
    if dest.exists() and not overwrite:
        log.debug("  Already exists: %s", dest.name)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    log.info("  Saved %s (%d KB)", dest.name, len(content) // 1024)
    return True


def _fetch_bytes(session: requests.Session, url: str) -> bytes | None:
    r = _get(session, url, stream=True)
    if r is None:
        return None
    chunks, total = [], 0
    for chunk in r.iter_content(chunk_size=65536):
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            log.warning("  Truncated at %d MB: %s", MAX_FILE_BYTES // 1_000_000, url)
            break
    return b"".join(chunks) or None


def _is_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"


def _is_publisher_url(url: str) -> bool:
    host = urlparse(url).netloc.lower().lstrip("www.")
    return any(host == d or host.endswith("." + d) for d in _PUBLISHER_DOMAINS)


def _is_useful_html(content: bytes, url: str = "", min_bytes: int = 3000) -> bool:
    """True if HTML content is worth keeping.

    Applies strict paywall detection only for known publisher domains.
    For open repositories and institutional pages, only checks minimum size.
    """
    if len(content) < min_bytes:
        return False
    if _is_publisher_url(url):
        text = content[:4000].decode("utf-8", errors="replace").lower()
        for signal in ("access denied", "subscribe to read", "purchase access",
                        "log in to read", "institutional access", "paywall",
                        "access this article", "buy article"):
            if signal in text:
                log.debug("  Paywall signal at %s", url)
                return False
    return True


# ---------------------------------------------------------------------------
# Unpaywall
# ---------------------------------------------------------------------------

def _unpaywall_pdf_url(doi: str, session: requests.Session) -> str | None:
    r = _get(session, f"{UNPAYWALL_BASE}/{doi}?email={UNPAYWALL_EMAIL}")
    if r is None:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    if not data.get("is_oa"):
        return None
    best = data.get("best_oa_location") or {}
    pdf_url = best.get("url_for_pdf") or best.get("url")
    if pdf_url:
        return pdf_url
    for loc in data.get("oa_locations") or []:
        if loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    return None


# ---------------------------------------------------------------------------
# Europe PMC
# ---------------------------------------------------------------------------

def _europepmc_pdf_url(doi: str, session: requests.Session) -> str | None:
    r = _get(session, EUROPEPMC_BASE, params={
        "query": f"DOI:{doi}",
        "format": "json",
        "resultType": "core",
        "pageSize": 1,
    })
    if r is None:
        return None
    try:
        results = r.json().get("resultList", {}).get("result", [])
    except Exception:
        return None
    if not results or results[0].get("isOpenAccess") != "Y":
        return None
    pmcid = results[0].get("pmcid")
    if pmcid:
        return f"https://europepmc.org/articles/{pmcid}?pdf=render"
    return None


# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------

def _fetch_zenodo(doi: str, dest_dir: Path, stem: str,
                  session: requests.Session, overwrite: bool) -> list[Path]:
    """Fetch files from a Zenodo record. Falls back to description text if no docs."""
    m = re.search(r'zenodo\.(\d+)', doi, re.I)
    if not m:
        return []
    record_id = m.group(1)

    r = _get(session, f"{ZENODO_API_BASE}/{record_id}")
    if r is None:
        return []
    try:
        data = r.json()
    except Exception:
        return []

    saved: list[Path] = []
    for f in data.get("files") or []:
        fname = f.get("key", "")
        furl  = (f.get("links") or {}).get("self") or f.get("url")
        if not furl:
            continue
        ext = Path(fname).suffix.lower()
        if ext not in (".pdf", ".txt", ".md", ".rst", ".html"):
            log.debug("  Zenodo: skipping data file %s", fname)
            continue
        content = _fetch_bytes(session, furl)
        if not content:
            continue
        dest = dest_dir / f"{stem}_{_slugify(fname, 40)}"
        if _save(dest, content, overwrite=overwrite):
            saved.append(dest)
        time.sleep(INTER_REQUEST_SLEEP)

    # Fallback: save record title + description as plain text
    if not saved:
        meta  = data.get("metadata") or {}
        title = meta.get("title", f"Zenodo record {record_id}")
        desc  = meta.get("description", "")
        if desc:
            txt = f"Title: {title}\n\nDescription:\n{desc}"
            dest = dest_dir / f"{stem}_description.txt"
            if _save(dest, txt.encode("utf-8"), overwrite=overwrite):
                saved.append(dest)

    return saved


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def _fetch_github(match: re.Match, dest_dir: Path, session: requests.Session,
                  overwrite: bool) -> list[Path]:
    owner, repo = match.group(1), match.group(2).rstrip("/")
    saved: list[Path] = []
    for branch in ("main", "master"):
        r = _get(session, f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}")
        if r is None:
            continue
        try:
            tree = r.json().get("tree", [])
        except Exception:
            continue
        for item in tree:
            name: str = item.get("path", "")
            if item.get("type") != "blob":
                continue
            if (name.lower() in ("readme.md", "readme.rst", "readme.txt")
                    or (name.endswith(".md") and "/" not in name)):
                raw_url = f"{GITHUB_RAW_BASE}/{owner}/{repo}/{branch}/{name}"
                r2 = _get(session, raw_url)
                if r2 is None:
                    continue
                dest = dest_dir / f"{_slugify(f'github_{owner}_{repo}_{name}')}.md"
                if _save(dest, r2.content, overwrite=overwrite):
                    saved.append(dest)
                time.sleep(INTER_REQUEST_SLEEP)
        if saved:
            break
    return saved


# ---------------------------------------------------------------------------
# Per-reference download
# ---------------------------------------------------------------------------

def _download_reference(
    ref_text: str,
    ref_id: int,
    dest_dir: Path,
    session: requests.Session,
    overwrite: bool,
) -> list[Path]:
    saved: list[Path] = []
    attempted_urls: set[str] = set()

    # Extract DOIs — strip trailing punctuation after extraction
    dois = [re.sub(r'[.,;:]+$', '', d) for d in DOI_RE.findall(ref_text)]
    urls = URL_RE.findall(ref_text)

    def _try_url(url: str, stem: str) -> bool:
        """Fetch url and save. Returns True if something was saved."""
        if url in attempted_urls:
            return False
        attempted_urls.add(url)
        # GitHub: special handling
        m = GITHUB_RE.match(url)
        if m:
            paths = _fetch_github(m, dest_dir, session, overwrite)
            saved.extend(paths)
            return bool(paths)
        content = _fetch_bytes(session, url)
        if not content:
            return False
        if _is_pdf(content):
            dest = dest_dir / f"{stem}.pdf"
        else:
            if not _is_useful_html(content, url=url):
                log.debug("  Skipping unusable HTML at %s", url)
                return False
            dest = dest_dir / f"{stem}.html"
        if _save(dest, content, overwrite=overwrite):
            saved.append(dest)
            return True
        return False

    for doi in dois:
        stem = _slugify(doi, 60) + f"_{ref_id}"

        # 1. Zenodo — use API (not the HTML landing page)
        if ZENODO_RE.search(doi):
            log.debug("  Zenodo DOI: %s", doi)
            paths = _fetch_zenodo(doi, dest_dir, stem, session, overwrite)
            if paths:
                saved.extend(paths)
                time.sleep(INTER_REQUEST_SLEEP)
                continue

        # 2. Unpaywall
        log.debug("  Unpaywall: %s", doi)
        pdf_url = _unpaywall_pdf_url(doi, session)
        if pdf_url and _try_url(pdf_url, stem):
            time.sleep(INTER_REQUEST_SLEEP)
            continue

        # 3. Europe PMC
        log.debug("  EuropePMC: %s", doi)
        epmc_url = _europepmc_pdf_url(doi, session)
        if epmc_url and _try_url(epmc_url, stem):
            time.sleep(INTER_REQUEST_SLEEP)
            continue

        # 4. Direct DOI resolve
        doi_url = f"https://doi.org/{doi}"
        log.debug("  Direct DOI: %s", doi_url)
        _try_url(doi_url, stem)
        time.sleep(INTER_REQUEST_SLEEP)

    # 5. Explicit URLs (skip doi.org already handled above)
    for url in urls:
        if "doi.org" in url:
            continue
        m = GITHUB_RE.match(url)
        if m:
            paths = _fetch_github(m, dest_dir, session, overwrite)
            saved.extend(paths)
            time.sleep(INTER_REQUEST_SLEEP)
            continue
        stem = _slugify(url, 60) + f"_{ref_id}"
        _try_url(url, stem)
        time.sleep(INTER_REQUEST_SLEEP)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_references(store: SQLServerStore,
                     datasets: list[str] | None,
                     limit: int) -> list[dict]:
    sql = """
    SELECT
        d.Dataset_Name  AS ShortName,
        r.Dataset_ID    AS DatasetId,
        r.Reference_ID  AS ReferenceId,
        r.Reference     AS Reference,
        r.Data_DOI      AS DataDOI
    FROM dbo.tblDataset_References r
    JOIN dbo.tblDatasets d ON d.ID = r.Dataset_ID
    WHERE r.Reference IS NOT NULL AND LEN(TRIM(r.Reference)) > 0
    ORDER BY d.Dataset_Name, r.Reference_ID
    """
    import pandas as pd
    with store.engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)
    if datasets:
        df = df[df["ShortName"].isin(datasets)]
    if limit and limit > 0:
        unique_ds = df["ShortName"].unique()[:limit]
        df = df[df["ShortName"].isin(unique_ds)]
    return df.to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download reference documents into the CMAP reference bank."
    )
    ap.add_argument("--bank-dir", default=None,
                    help="Path to reference bank root. "
                         "Default: notrack/reference_bank/ relative to project root.")
    ap.add_argument("--dataset", action="append", dest="datasets", metavar="SHORT_NAME",
                    help="Only download references for this dataset (repeatable).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most this many datasets (0 = all).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download files that already exist in the bank.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be downloaded without fetching anything.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.bank_dir:
        bank_root = Path(args.bank_dir)
    else:
        here = Path(__file__).resolve()
        project_root = here.parents[3]
        bank_root = project_root / "notrack" / "reference_bank"

    bank_root.mkdir(parents=True, exist_ok=True)
    log.info("Reference bank: %s", bank_root)

    store = SQLServerStore.from_env()
    refs = _load_references(store, args.datasets, args.limit)
    log.info("Loaded %d references across %d datasets.",
             len(refs), len({r["ShortName"] for r in refs}))

    if args.dry_run:
        for r in refs:
            dois = [re.sub(r'[.,;:]+$', '', d) for d in DOI_RE.findall(r["Reference"])]
            urls = URL_RE.findall(r["Reference"])
            print(f"[{r['ShortName']}] ref_id={r['ReferenceId']} "
                  f"dois={dois} urls={urls[:3]}")
        return

    session = _session()
    stats = {"attempted": 0, "saved": 0, "failed": 0}

    refs_sorted = sorted(refs, key=itemgetter("ShortName", "ReferenceId"))
    for short_name, group in groupby(refs_sorted, key=itemgetter("ShortName")):
        dest_dir = bank_root / short_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        log.info("Dataset: %s", short_name)

        for ref in group:
            ref_id   = ref["ReferenceId"]
            ref_text = (ref["Reference"] or "").strip()
            if not ref_text:
                continue
            stats["attempted"] += 1
            log.info("  [ref %d] %s", ref_id, ref_text[:120])
            saved = _download_reference(
                ref_text=ref_text,
                ref_id=ref_id,
                dest_dir=dest_dir,
                session=session,
                overwrite=args.overwrite,
            )
            if saved:
                stats["saved"] += len(saved)
            else:
                stats["failed"] += 1
                log.info("  [ref %d] nothing downloaded", ref_id)

    log.info(
        "Done. %d references attempted, %d files saved, %d with nothing downloaded.",
        stats["attempted"], stats["saved"], stats["failed"],
    )


if __name__ == "__main__":
    main()
