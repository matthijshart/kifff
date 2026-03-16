#!/usr/bin/env python3
"""
KIFID Uitspraken Scraper
========================
Downloads KIFID decisions (PDFs) from kifid.nl using the KIFID Search API.

The KIFID website is a React SPA backed by an IIS/ASP.NET API at:
    https://www.kifid.nl/api/Search/SearchDecision/

This scraper paginates through the API, collects PDF URLs (statementLink),
and downloads the PDFs to data/pdfs/.

Usage:
    python3 scripts/kifid_scraper.py              # Run full scraper
    python3 scripts/kifid_scraper.py --discover    # Only discover URLs, don't download
    python3 scripts/kifid_scraper.py --download    # Only download from urls.json
    python3 scripts/kifid_scraper.py --limit 50    # Limit number of downloads
    python3 scripts/kifid_scraper.py --category Verzekeringen  # Filter by category
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://www.kifid.nl"
API_URL = f"{BASE_URL}/api/Search/SearchDecision/"

USER_AGENT = "KIFID-Predictor-Research/1.0 (+https://github.com/kifid-predictor)"
REQUEST_DELAY = 2.0  # seconds between requests
REQUEST_TIMEOUT = 30  # seconds
PAGE_SIZE = 100  # max items per API page

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
URLS_FILE = DATA_DIR / "urls.json"
LOG_FILE = DATA_DIR / "download_log.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kifid-scraper")

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    return session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_uitspraak_nr(url_or_name: str) -> Optional[str]:
    """Extract uitspraaknummer like '2025-0448' from a URL or name."""
    m = re.search(r"(\d{4}-\d{3,4})", url_or_name)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# KIFID API Discovery
# ---------------------------------------------------------------------------


def discover_from_api(
    session: requests.Session,
    category: str = "",
    max_items: int = 0,
) -> List[dict]:
    """Paginate through the KIFID Search API and collect uitspraak metadata."""
    log.info("Discovering uitspraken via KIFID API...")

    all_items = []  # type: List[dict]
    page = 1
    total_items = None

    while True:
        params = {
            "searchTerm": "",
            "category": category,
            "authority": "",
            "targetGroup": "",
            "startDate": 0,
            "endDate": 0,
            "page": page,
            "pageSize": PAGE_SIZE,
            "sort": "",
        }

        log.info("  Fetching page %d ...", page)
        time.sleep(REQUEST_DELAY)

        try:
            resp = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error("  API request failed on page %d: %s", page, e)
            break

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            log.error("  Failed to parse API response on page %d: %s", page, e)
            break

        decision = data.get("decision", {})
        items = decision.get("decisionItemsList", [])

        if total_items is None:
            total_items = data.get("totalItems", 0)
            log.info("  Total items available: %s", total_items)

        if not items:
            log.info("  No more items on page %d, stopping.", page)
            break

        for item in items:
            pdf_url = item.get("statementLink", "")
            name = item.get("name", "") or item.get("title", "")
            nr = extract_uitspraak_nr(name) or extract_uitspraak_nr(pdf_url)

            if not nr or not pdf_url:
                log.debug("  Skipping item without nr/pdf: %s", name)
                continue

            all_items.append({
                "uitspraaknr": nr,
                "pdf_url": pdf_url,
                "title": name,
                "category": item.get("category", ""),
                "authority": item.get("authority", ""),
                "defendant": item.get("defendant", ""),
                "summary": item.get("summary", ""),
                "page_url": item.get("url", ""),
                "date_found": date.today().isoformat(),
                "downloaded": False,
            })

        log.info("  Page %d: collected %d items (total so far: %d)",
                 page, len(items), len(all_items))

        # Stop if we've hit the user-requested limit
        if max_items and len(all_items) >= max_items:
            all_items = all_items[:max_items]
            log.info("  Reached requested limit of %d items.", max_items)
            break

        # Stop if we've fetched all available items
        if total_items and len(all_items) >= total_items:
            break

        # Stop if page returned fewer items than page size (last page)
        if len(items) < PAGE_SIZE:
            break

        page += 1

    log.info("Discovery complete: %d uitspraken found.", len(all_items))
    return all_items


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_urls() -> List[dict]:
    if URLS_FILE.exists():
        return json.loads(URLS_FILE.read_text(encoding="utf-8"))
    return []


def save_urls(urls: List[dict]) -> None:
    URLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    URLS_FILE.write_text(json.dumps(urls, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d URLs to %s", len(urls), URLS_FILE)


def merge_urls(existing: List[dict], new: List[dict]) -> List[dict]:
    """Merge new URLs into existing, preserving download status."""
    by_nr = {}  # type: Dict[str, dict]
    for u in existing:
        by_nr[u["uitspraaknr"]] = u
    for item in new:
        nr = item["uitspraaknr"]
        if nr in by_nr:
            # Keep existing download status, update PDF URL if needed
            if not by_nr[nr].get("pdf_url") and item.get("pdf_url"):
                by_nr[nr]["pdf_url"] = item["pdf_url"]
        else:
            by_nr[nr] = item
    return sorted(by_nr.values(), key=lambda u: u["uitspraaknr"])


def load_download_log() -> dict:
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text(encoding="utf-8"))
    return {"downloaded": [], "failed": [], "last_run": None}


def save_download_log(log_data: dict) -> None:
    LOG_FILE.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_pdfs(session: requests.Session, urls: List[dict], limit: Optional[int] = None) -> List[dict]:
    """Download PDFs that haven't been downloaded yet."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    dl_log = load_download_log()

    to_download = [u for u in urls if not u.get("downloaded") and u.get("pdf_url")]
    if limit:
        to_download = to_download[:limit]

    log.info("Downloading %d PDFs (skipping %d already downloaded)...",
             len(to_download), len(urls) - len(to_download))

    downloaded_count = 0
    failed_count = 0

    for i, item in enumerate(to_download, 1):
        nr = item["uitspraaknr"]
        pdf_url = item["pdf_url"]

        # Determine filename from URL (preserve original name)
        url_filename = urlparse(pdf_url).path.split("/")[-1]
        local_filename = url_filename if url_filename.endswith(".pdf") else "uitspraak-%s.pdf" % nr
        local_path = PDF_DIR / local_filename

        # Skip if already on disk
        if local_path.exists() and local_path.stat().st_size > 1000:
            log.info("  [%d/%d] Skip (exists): %s", i, len(to_download), local_filename)
            item["downloaded"] = True
            item["local_file"] = str(local_path.relative_to(PROJECT_ROOT))
            downloaded_count += 1
            continue

        log.info("  [%d/%d] Downloading: %s", i, len(to_download), nr)
        time.sleep(REQUEST_DELAY)

        try:
            resp = session.get(pdf_url, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                log.warning("  Unexpected content-type for %s: %s", nr, content_type)

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = local_path.stat().st_size
            if file_size < 1000:
                log.warning("  File too small (%d bytes), likely not a valid PDF: %s", file_size, nr)
                local_path.unlink(missing_ok=True)
                dl_log["failed"].append({"uitspraaknr": nr, "reason": "file_too_small", "size": file_size})
                failed_count += 1
                continue

            item["downloaded"] = True
            item["local_file"] = str(local_path.relative_to(PROJECT_ROOT))
            dl_log["downloaded"].append(nr)
            downloaded_count += 1
            log.info("  OK: %s (%d KB)", local_filename, file_size // 1024)

        except requests.RequestException as e:
            log.error("  FAILED: %s — %s", nr, e)
            dl_log["failed"].append({"uitspraaknr": nr, "reason": str(e)})
            failed_count += 1

    dl_log["last_run"] = date.today().isoformat()
    dl_log["stats"] = {
        "total_urls": len(urls),
        "downloaded": downloaded_count,
        "failed": failed_count,
        "total_on_disk": len(list(PDF_DIR.glob("*.pdf"))),
    }
    save_download_log(dl_log)

    log.info("Download complete: %d succeeded, %d failed", downloaded_count, failed_count)
    return urls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="KIFID Uitspraken PDF Scraper")
    parser.add_argument("--discover", action="store_true", help="Only discover URLs, don't download")
    parser.add_argument("--download", action="store_true", help="Only download from existing urls.json")
    parser.add_argument("--limit", type=int, default=None, help="Max number of PDFs to download")
    parser.add_argument("--max-items", type=int, default=0, help="Max items to discover from API (0=all)")
    parser.add_argument("--category", type=str, default="", help="Filter by category (e.g. 'Verzekeringen')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    session = create_session()

    if args.download:
        # Only download from existing urls.json
        urls = load_urls()
        if not urls:
            log.error("No urls.json found. Run without --download first to discover URLs.")
            sys.exit(1)
        urls = download_pdfs(session, urls, limit=args.limit)
        save_urls(urls)
        return

    # Discovery phase
    log.info("=" * 60)
    log.info("KIFID Uitspraken Scraper — Discovery Phase")
    log.info("=" * 60)

    new_urls = discover_from_api(session, category=args.category, max_items=args.max_items)

    # Merge with existing
    existing = load_urls()
    all_urls = merge_urls(existing, new_urls)
    save_urls(all_urls)

    if args.discover:
        log.info("Discovery complete. Found %d unique uitspraken with PDF URLs.", len(all_urls))
        log.info("Run without --discover to download PDFs.")
        return

    # Download phase
    log.info("=" * 60)
    log.info("KIFID Uitspraken Scraper — Download Phase")
    log.info("=" * 60)

    all_urls = download_pdfs(session, all_urls, limit=args.limit)
    save_urls(all_urls)

    # Summary
    total = len(all_urls)
    downloaded = sum(1 for u in all_urls if u.get("downloaded"))
    on_disk = len(list(PDF_DIR.glob("*.pdf")))

    log.info("=" * 60)
    log.info("Summary:")
    log.info("  URLs discovered:  %d", total)
    log.info("  PDFs downloaded:  %d", downloaded)
    log.info("  Files on disk:    %d", on_disk)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
