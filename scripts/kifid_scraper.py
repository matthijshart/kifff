#!/usr/bin/env python3
"""
KIFID Uitspraken Scraper
========================
Downloads KIFID decisions (PDFs) from kifid.nl.

Strategies (tried in order):
1. Sitemap XML parsing - find PDF links in sitemaps
2. Uitspraken search page scraping - parse paginated results
3. Individual uitspraak page scraping - find PDF download links
4. Known URL patterns - try known PDF URL patterns as fallback

Usage:
    python3 scripts/kifid_scraper.py              # Run full scraper
    python3 scripts/kifid_scraper.py --discover    # Only discover URLs, don't download
    python3 scripts/kifid_scraper.py --download    # Only download from urls.json
    python3 scripts/kifid_scraper.py --limit 50    # Limit number of downloads
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://www.kifid.nl"
UITSPRAKEN_URL = f"{BASE_URL}/kifid-kennis-en-uitspraken/uitspraken"
SITEMAP_URLS = [
    f"{BASE_URL}/sitemap_index.xml",
    f"{BASE_URL}/sitemap.xml",
    f"{BASE_URL}/wp-sitemap.xml",
    f"{BASE_URL}/wp-sitemap-posts-uitspraak-1.xml",
    f"{BASE_URL}/wp-sitemap-posts-post-1.xml",
]

USER_AGENT = "KIFID-Predictor-Research/1.0 (+https://github.com/kifid-predictor)"
REQUEST_DELAY = 2.0  # seconds between requests
REQUEST_TIMEOUT = 30  # seconds

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
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "nl,en;q=0.5",
    })
    return session


def fetch(session: requests.Session, url: str, **kwargs) -> requests.Response | None:
    """Fetch a URL with rate limiting and error handling."""
    time.sleep(REQUEST_DELAY)
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        log.warning("Failed to fetch %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# URL Discovery Strategies
# ---------------------------------------------------------------------------


def extract_uitspraak_nr(url_or_filename: str) -> str | None:
    """Extract uitspraaknummer from a URL or filename like 'uitspraak-2025-0448'."""
    m = re.search(r"uitspraak-(\d{4}-\d{3,4})", url_or_filename)
    return m.group(1) if m else None


def is_kifid_pdf_url(url: str) -> bool:
    """Check if URL looks like a KIFID uitspraak PDF."""
    return bool(
        re.search(r"kifid\.nl/media/[a-z0-9]+/uitspraak-\d{4}-\d{3,4}.*\.pdf", url)
    )


def discover_from_sitemaps(session: requests.Session) -> list[dict]:
    """Strategy 1: Parse sitemaps for uitspraak pages/PDFs."""
    log.info("Strategy 1: Checking sitemaps...")
    found = []

    for sitemap_url in SITEMAP_URLS:
        resp = fetch(session, sitemap_url)
        if not resp:
            continue

        log.info("  Found sitemap: %s", sitemap_url)
        soup = BeautifulSoup(resp.content, "lxml-xml")

        # Check for sitemap index (contains other sitemaps)
        sub_sitemaps = [loc.text.strip() for loc in soup.find_all("loc")
                        if "sitemap" in loc.text.lower() and loc.text.strip().endswith(".xml")]

        # Check for direct URLs
        urls = [loc.text.strip() for loc in soup.find_all("loc")
                if "uitspraak" in loc.text.lower() or loc.text.strip().endswith(".pdf")]

        for url in urls:
            if url.endswith(".pdf") and is_kifid_pdf_url(url):
                nr = extract_uitspraak_nr(url)
                if nr:
                    found.append({"uitspraaknr": nr, "pdf_url": url, "source": "sitemap"})
            elif "uitspraak" in url.lower() and not url.endswith(".xml"):
                # This is a page URL, we'll need to scrape it for the PDF link
                found.append({"page_url": url, "source": "sitemap"})

        # Recurse into sub-sitemaps
        for sub_url in sub_sitemaps:
            if sub_url in SITEMAP_URLS:
                continue
            log.info("  Checking sub-sitemap: %s", sub_url)
            sub_resp = fetch(session, sub_url)
            if not sub_resp:
                continue
            sub_soup = BeautifulSoup(sub_resp.content, "lxml-xml")
            for loc in sub_soup.find_all("loc"):
                url = loc.text.strip()
                if url.endswith(".pdf") and is_kifid_pdf_url(url):
                    nr = extract_uitspraak_nr(url)
                    if nr:
                        found.append({"uitspraaknr": nr, "pdf_url": url, "source": "sitemap"})
                elif "uitspraak" in url.lower() and not url.endswith(".xml"):
                    found.append({"page_url": url, "source": "sitemap"})

    log.info("  Sitemap strategy found %d items", len(found))
    return found


def discover_from_search_pages(session: requests.Session, max_pages: int = 20) -> list[dict]:
    """Strategy 2: Scrape the uitspraken search/listing pages."""
    log.info("Strategy 2: Scraping uitspraken listing pages...")
    found = []

    # Try paginated listing
    page_urls_to_try = [
        UITSPRAKEN_URL,
        f"{BASE_URL}/uitspraken/",
        f"{BASE_URL}/category/uitspraak/",
    ]

    for base_url in page_urls_to_try:
        resp = fetch(session, base_url)
        if not resp:
            continue

        log.info("  Found listing at: %s", base_url)
        found.extend(_extract_from_html(resp.text, resp.url))

        # Try pagination
        for page_num in range(2, max_pages + 1):
            # Try multiple pagination patterns
            page_variants = [
                f"{base_url}?paged={page_num}",
                f"{base_url}page/{page_num}/",
                f"{base_url}?page={page_num}",
            ]
            page_found = False
            for page_url in page_variants:
                resp = fetch(session, page_url)
                if resp and resp.status_code == 200:
                    new_items = _extract_from_html(resp.text, resp.url)
                    if not new_items:
                        break
                    found.extend(new_items)
                    log.info("  Page %d: found %d items", page_num, len(new_items))
                    page_found = True
                    break
            if not page_found:
                break

        if found:
            break  # Found a working listing URL

    log.info("  Search page strategy found %d items", len(found))
    return found


def _extract_from_html(html: str, base_url: str) -> list[dict]:
    """Extract uitspraak links and PDF URLs from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    items = []

    # Look for direct PDF links
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if href.endswith(".pdf") and is_kifid_pdf_url(href):
            nr = extract_uitspraak_nr(href)
            if nr:
                title = a.get_text(strip=True) or None
                items.append({"uitspraaknr": nr, "pdf_url": href, "title": title, "source": "html"})

        # Look for links to individual uitspraak pages
        elif re.search(r"uitspraak.*\d{4}-\d{3,4}", href):
            nr = extract_uitspraak_nr(href)
            if nr:
                title = a.get_text(strip=True) or None
                items.append({"uitspraaknr": nr, "page_url": href, "title": title, "source": "html"})

    return items


def discover_from_ajax(session: requests.Session, max_pages: int = 20) -> list[dict]:
    """Strategy 3: Try AJAX/REST endpoints for dynamic content loading."""
    log.info("Strategy 3: Trying AJAX/REST endpoints...")
    found = []

    # Try WP REST API with various custom post types
    rest_endpoints = [
        f"{BASE_URL}/wp-json/wp/v2/posts?per_page=100&search=uitspraak",
        f"{BASE_URL}/wp-json/wp/v2/uitspraak?per_page=100",
        f"{BASE_URL}/wp-json/wp/v2/uitspraken?per_page=100",
        f"{BASE_URL}/wp-json/wp/v2/pages?per_page=100&search=uitspraak",
        f"{BASE_URL}/wp-json/kifid/v1/uitspraken",
        f"{BASE_URL}/wp-json/",
    ]

    for endpoint in rest_endpoints:
        resp = fetch(session, endpoint)
        if not resp:
            continue

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            continue

        log.info("  REST endpoint responded: %s", endpoint)

        # If we got the API index, look for useful routes
        if isinstance(data, dict) and "routes" in data:
            log.info("  Found WP REST API index with routes:")
            for route in data.get("routes", {}):
                if "uitspraak" in route.lower() or "kifid" in route.lower():
                    log.info("    → %s", route)
            continue

        # Process list of posts
        if isinstance(data, list):
            for post in data:
                content = post.get("content", {}).get("rendered", "")
                title = post.get("title", {}).get("rendered", "")
                link = post.get("link", "")

                # Extract PDF URLs from post content
                pdf_urls = re.findall(
                    r'https?://[^"\'>\s]+kifid\.nl/media/[a-z0-9]+/uitspraak-\d{4}-\d{3,4}[^"\'>\s]*\.pdf',
                    content,
                )
                for pdf_url in pdf_urls:
                    nr = extract_uitspraak_nr(pdf_url)
                    if nr:
                        found.append({
                            "uitspraaknr": nr, "pdf_url": pdf_url,
                            "title": title, "source": "rest_api",
                        })

                # If no direct PDF but has uitspraak link
                if not pdf_urls and "uitspraak" in (link or ""):
                    nr = extract_uitspraak_nr(link)
                    if nr:
                        found.append({
                            "uitspraaknr": nr, "page_url": link,
                            "title": title, "source": "rest_api",
                        })

            # Paginate REST API
            if len(data) >= 100:
                for page in range(2, max_pages + 1):
                    sep = "&" if "?" in endpoint else "?"
                    paged_url = f"{endpoint}{sep}page={page}"
                    resp = fetch(session, paged_url)
                    if not resp:
                        break
                    try:
                        page_data = resp.json()
                    except (json.JSONDecodeError, ValueError):
                        break
                    if not page_data:
                        break
                    for post in page_data:
                        content = post.get("content", {}).get("rendered", "")
                        pdf_urls = re.findall(
                            r'https?://[^"\'>\s]+kifid\.nl/media/[a-z0-9]+/uitspraak-\d{4}-\d{3,4}[^"\'>\s]*\.pdf',
                            content,
                        )
                        for pdf_url in pdf_urls:
                            nr = extract_uitspraak_nr(pdf_url)
                            if nr:
                                found.append({
                                    "uitspraaknr": nr, "pdf_url": pdf_url,
                                    "source": "rest_api",
                                })

    # Try WordPress admin-ajax.php
    ajax_actions = [
        "kifid_search", "load_uitspraken", "search_uitspraken",
        "get_uitspraken", "kifid_get_results",
    ]
    for action in ajax_actions:
        resp = fetch(
            session,
            f"{BASE_URL}/wp-admin/admin-ajax.php",
            params={"action": action, "page": 1, "per_page": 50},
        )
        if resp and resp.status_code == 200 and len(resp.text) > 50:
            log.info("  AJAX action '%s' responded", action)
            # Try to extract PDF URLs from response
            pdf_urls = re.findall(
                r'https?://[^"\'>\s]+kifid\.nl/media/[a-z0-9]+/uitspraak-\d{4}-\d{3,4}[^"\'>\s]*\.pdf',
                resp.text,
            )
            for pdf_url in pdf_urls:
                nr = extract_uitspraak_nr(pdf_url)
                if nr:
                    found.append({"uitspraaknr": nr, "pdf_url": pdf_url, "source": "ajax"})

    log.info("  AJAX/REST strategy found %d items", len(found))
    return found


def resolve_page_urls(session: requests.Session, items: list[dict]) -> list[dict]:
    """For items that only have a page_url, scrape the page to find the PDF URL."""
    log.info("Resolving %d page URLs to PDF URLs...", sum(1 for i in items if "page_url" in i and "pdf_url" not in i))

    for item in items:
        if "pdf_url" in item or "page_url" not in item:
            continue

        resp = fetch(session, item["page_url"])
        if not resp:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Look for PDF download links on the page
        for a in soup.find_all("a", href=True):
            href = urljoin(resp.url, a["href"])
            if href.endswith(".pdf") and is_kifid_pdf_url(href):
                item["pdf_url"] = href
                nr = extract_uitspraak_nr(href)
                if nr:
                    item["uitspraaknr"] = nr
                if not item.get("title"):
                    title_tag = soup.find("h1")
                    if title_tag:
                        item["title"] = title_tag.get_text(strip=True)
                log.info("  Resolved: %s → %s", item.get("uitspraaknr", "?"), href)
                break

        # Also check for embedded PDF viewers or iframes
        if "pdf_url" not in item:
            for tag in soup.find_all(["iframe", "embed", "object"]):
                src = tag.get("src") or tag.get("data")
                if src and ".pdf" in src:
                    full_url = urljoin(resp.url, src)
                    if is_kifid_pdf_url(full_url):
                        item["pdf_url"] = full_url
                        break

    return items


# ---------------------------------------------------------------------------
# URL collection & deduplication
# ---------------------------------------------------------------------------


def collect_urls(session: requests.Session) -> list[dict]:
    """Run all discovery strategies and merge results."""
    all_items = []

    # Run strategies
    all_items.extend(discover_from_sitemaps(session))
    all_items.extend(discover_from_search_pages(session))
    all_items.extend(discover_from_ajax(session))

    # Resolve page URLs to PDF URLs
    all_items = resolve_page_urls(session, all_items)

    # Deduplicate by uitspraaknr, preferring items with pdf_url
    by_nr: dict[str, dict] = {}
    for item in all_items:
        nr = item.get("uitspraaknr")
        if not nr:
            continue
        existing = by_nr.get(nr)
        if not existing or ("pdf_url" in item and "pdf_url" not in existing):
            by_nr[nr] = item

    # Build final list
    today = date.today().isoformat()
    results = []
    for nr in sorted(by_nr.keys()):
        item = by_nr[nr]
        if "pdf_url" not in item:
            continue  # Skip items where we couldn't find the PDF
        results.append({
            "uitspraaknr": nr,
            "pdf_url": item["pdf_url"],
            "title": item.get("title", f"Uitspraak {nr}"),
            "date_found": today,
            "downloaded": False,
        })

    log.info("Total unique uitspraken with PDF URLs: %d", len(results))
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_urls() -> list[dict]:
    if URLS_FILE.exists():
        return json.loads(URLS_FILE.read_text(encoding="utf-8"))
    return []


def save_urls(urls: list[dict]) -> None:
    URLS_FILE.write_text(json.dumps(urls, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d URLs to %s", len(urls), URLS_FILE)


def merge_urls(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge new URLs into existing, preserving download status."""
    by_nr = {u["uitspraaknr"]: u for u in existing}
    for item in new:
        nr = item["uitspraaknr"]
        if nr in by_nr:
            # Keep existing download status, update PDF URL if we have a new one
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


def download_pdfs(session: requests.Session, urls: list[dict], limit: int | None = None) -> list[dict]:
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
        local_filename = url_filename if url_filename.endswith(".pdf") else f"uitspraak-{nr}.pdf"
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

    new_urls = collect_urls(session)

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
