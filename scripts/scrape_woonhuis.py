#!/usr/bin/env python3
"""
Scraper voor KIFID woonhuisverzekering uitspraken.

KIFID (kifid.nl) is een WordPress-site. Dit script probeert meerdere methodes:
  1. WordPress REST API (/wp-json/wp/v2/judgement)
  2. HTML scraping van archiefpagina's (/category/uitspraak/)
  3. Individuele uitspraakpagina's (/judgement/...)

Gebruik:
    python3 scripts/scrape_woonhuis.py

Output:
    data/uitspraken/woonhuis_raw.json   – ruwe scraped data
    data/uitspraken/woonhuis_dataset.json – gestructureerd in dataset-formaat
"""

import json
import os
import sys
import time
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

try:
    import requests
except ImportError:
    print("requests is niet geinstalleerd. Run: pip3 install requests")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("beautifulsoup4 is niet geinstalleerd. Run: pip3 install beautifulsoup4")
    sys.exit(1)

BASE_URL = "https://www.kifid.nl"
MIN_TARGET = 200
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uitspraken")

# Trefwoorden die aangeven dat een uitspraak over woonhuisverzekering gaat
WOONHUIS_KEYWORDS = [
    "woonhuisverzekering", "opstalverzekering", "inboedelverzekering",
    "woonverzekering", "opstal", "inboedel", "woonhuis",
    "woningverzekering", "huiseigenaar", "glasverzekering",
    "waterschade", "brandschade", "stormschade", "lekkage",
    "woningschade", "riool", "fundering", "dakschade",
    "leidingwater", "overstroming", "inbraakschade",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.8",
})


def fetch_with_retry(url: str, params: dict = None, timeout: int = 30) -> Optional[requests.Response]:
    """Fetch een URL met retry-logica."""
    for attempt in range(4):
        try:
            resp = SESSION.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None
            print(f"  HTTP {resp.status_code} voor {url}")
            if resp.status_code >= 500:
                wait = 2 ** (attempt + 1)
                print(f"  Server fout, wacht {wait}s...")
                time.sleep(wait)
                continue
            return None
        except requests.RequestException as e:
            wait = 2 ** (attempt + 1)
            print(f"  Netwerk fout (poging {attempt+1}/4): {e} – wacht {wait}s")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Methode 1: WordPress REST API
# ---------------------------------------------------------------------------

def try_wp_api() -> Tuple[List[dict], bool]:
    """Probeer uitspraken op te halen via de WordPress REST API."""
    print("\n=== Methode 1: WordPress REST API ===")

    # Probeer verschillende mogelijke endpoints
    endpoints = [
        f"{BASE_URL}/wp-json/wp/v2/judgement",
        f"{BASE_URL}/wp-json/wp/v2/judgements",
        f"{BASE_URL}/wp-json/wp/v2/uitspraak",
        f"{BASE_URL}/wp-json/wp/v2/uitspraken",
        f"{BASE_URL}/wp-json/wp/v2/posts?categories=uitspraak",
    ]

    working_endpoint = None
    for ep in endpoints:
        print(f"  Probeer: {ep}")
        resp = fetch_with_retry(ep, params={"per_page": 1})
        if resp and resp.status_code == 200:
            try:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    working_endpoint = ep
                    total = int(resp.headers.get("X-WP-Total", 0))
                    print(f"  Gevonden! Endpoint: {ep} ({total} totaal)")
                    break
            except (ValueError, KeyError):
                continue
        time.sleep(0.3)

    if not working_endpoint:
        print("  Geen werkend WP REST API endpoint gevonden.")
        return [], False

    # Haal alle pagina's op
    total = int(resp.headers.get("X-WP-Total", 0))
    total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
    all_items = []

    for page in range(1, total_pages + 1):
        print(f"  Pagina {page}/{total_pages}...")
        resp = fetch_with_retry(working_endpoint, params={"per_page": 100, "page": page})
        if not resp:
            break
        items = resp.json()
        all_items.extend(items)
        time.sleep(0.5)

    print(f"  WP API: {len(all_items)} items opgehaald")
    return all_items, True


# ---------------------------------------------------------------------------
# Methode 2: HTML scraping van archiefpagina's
# ---------------------------------------------------------------------------

def scrape_archive_page(url: str) -> Tuple[List[dict], Optional[str]]:
    """Scrape één archiefpagina en retourneer items + URL van volgende pagina."""
    resp = fetch_with_retry(url)
    if not resp:
        return [], None

    soup = BeautifulSoup(resp.text, "html.parser")
    items = []

    # Zoek uitspraak-links in de pagina
    # Mogelijke patronen: /judgement/uitspraak-XXXX-XXX-... of article elements
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/judgement/" in href or "/uitspraak" in href.lower():
            title = link.get_text(strip=True)
            if not title or len(title) < 5:
                continue
            full_url = href if href.startswith("http") else BASE_URL + href
            items.append({
                "title": title,
                "url": full_url,
            })

    # Zoek ook in article/div elementen
    for article in soup.find_all(["article", "div"], class_=re.compile(r"post|uitspraak|judgement|entry", re.I)):
        link_el = article.find("a", href=True)
        if not link_el:
            continue
        href = link_el["href"]
        if "/judgement/" not in href and "/uitspraak" not in href.lower():
            continue
        title = link_el.get_text(strip=True)
        # Probeer ook een datum te vinden
        date_el = article.find(["time", "span", "div"], class_=re.compile(r"date|datum|time", re.I))
        date_str = ""
        if date_el:
            date_str = date_el.get_text(strip=True)
            if date_el.get("datetime"):
                date_str = date_el["datetime"]

        excerpt_el = article.find(["p", "div"], class_=re.compile(r"excerpt|summary|content", re.I))
        excerpt = excerpt_el.get_text(strip=True) if excerpt_el else ""

        full_url = href if href.startswith("http") else BASE_URL + href
        items.append({
            "title": title,
            "url": full_url,
            "date": date_str,
            "excerpt": excerpt,
        })

    # Dedupliceer op URL
    seen = set()
    unique = []
    for item in items:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)

    # Zoek de "volgende pagina" link
    next_url = None
    next_link = soup.find("a", class_=re.compile(r"next|volgende", re.I))
    if next_link and next_link.get("href"):
        next_url = next_link["href"]
        if not next_url.startswith("http"):
            next_url = BASE_URL + next_url

    # Probeer ook /page/N/ patroon
    if not next_url:
        for a in soup.find_all("a", href=True):
            if re.search(r"/page/\d+/?$", a["href"]):
                href = a["href"]
                if not href.startswith("http"):
                    href = BASE_URL + href
                next_url = href

    return unique, next_url


def scrape_archives() -> List[dict]:
    """Scrape alle uitspraken uit de KIFID archiefpagina's."""
    print("\n=== Methode 2: HTML scraping archiefpagina's ===")

    archive_urls = [
        f"{BASE_URL}/category/uitspraak/",
        f"{BASE_URL}/kifid-kennis-en-uitspraken/uitspraken/",
    ]

    all_items = []
    seen_urls = set()

    for start_url in archive_urls:
        print(f"\nScraping archief: {start_url}")
        url = start_url
        page_num = 0

        while url and page_num < 300:  # max 300 pagina's per archief
            page_num += 1
            print(f"  Pagina {page_num}: {url}")
            items, next_url = scrape_archive_page(url)

            new_count = 0
            for item in items:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    all_items.append(item)
                    new_count += 1

            print(f"    {new_count} nieuwe items (totaal: {len(all_items)})")

            if new_count == 0 and page_num > 1:
                break

            # Probeer de volgende pagina
            if next_url:
                url = next_url
            else:
                # Probeer handmatig pagina-nummering
                next_page_url = f"{start_url}page/{page_num + 1}/"
                url = next_page_url

            time.sleep(0.5)

    print(f"\nArchief scraping: {len(all_items)} unieke items gevonden")
    return all_items


# ---------------------------------------------------------------------------
# Methode 3: Individuele uitspraakpagina's ophalen
# ---------------------------------------------------------------------------

def scrape_uitspraak_page(url: str) -> Optional[dict]:
    """Scrape de volledige tekst van een individuele uitspraakpagina."""
    resp = fetch_with_retry(url)
    if not resp:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Titel
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    # Zoek de hoofdcontent
    content = ""
    # Probeer verschillende content containers
    for selector in [
        {"class_": re.compile(r"entry-content|post-content|judgement-content|uitspraak", re.I)},
        {"class_": re.compile(r"content|main", re.I)},
    ]:
        content_el = soup.find(["div", "article", "section"], **selector)
        if content_el:
            content = content_el.get_text(separator="\n", strip=True)
            if len(content) > 200:  # moet substantieel zijn
                break

    # Als er geen goede content container is, pak de hele body
    if len(content) < 200:
        body = soup.find("body")
        if body:
            # Verwijder header, footer, nav
            for tag in body.find_all(["header", "footer", "nav", "script", "style"]):
                tag.decompose()
            content = body.get_text(separator="\n", strip=True)

    # Zoek datum
    date_str = ""
    date_el = soup.find("time")
    if date_el:
        date_str = date_el.get("datetime", date_el.get_text(strip=True))
    if not date_str:
        # Zoek in de tekst
        match = re.search(r"(\d{1,2}\s+(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+\d{4})", content.lower())
        if match:
            date_str = match.group(1)

    # Zoek PDF link
    pdf_url = ""
    for a in soup.find_all("a", href=True):
        if a["href"].endswith(".pdf"):
            pdf_url = a["href"]
            if not pdf_url.startswith("http"):
                pdf_url = BASE_URL + pdf_url
            break

    # Tags/categorieën
    tags = []
    for tag_el in soup.find_all(["a", "span"], class_=re.compile(r"tag|categor|label", re.I)):
        tag_text = tag_el.get_text(strip=True)
        if tag_text and len(tag_text) < 50:
            tags.append(tag_text)

    return {
        "title": title,
        "url": url,
        "date": date_str,
        "content": content,
        "pdf_url": pdf_url,
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# Filtering en conversie
# ---------------------------------------------------------------------------

def is_woonhuis_related(item: dict) -> bool:
    """Check of een uitspraak gerelateerd is aan woonhuisverzekering."""
    text = " ".join([
        item.get("content", ""),
        item.get("title", ""),
        item.get("excerpt", ""),
    ]).lower()
    return any(kw in text for kw in WOONHUIS_KEYWORDS)


def extract_uitspraaknr(item: dict) -> str:
    """Extraheer uitspraaknummer."""
    for field in ["title", "url"]:
        text = item.get(field, "")
        match = re.search(r"(\d{4}-\d{3,5})", text)
        if match:
            return match.group(1)
    return item.get("title", "onbekend")


def extract_datum(item: dict) -> str:
    """Extraheer datum in YYYY-MM-DD formaat."""
    date_str = item.get("date", "")
    if not date_str:
        return ""

    match = re.search(r"(\d{4}-\d{2}-\d{2})", date_str)
    if match:
        return match.group(1)

    # Nederlandse maanden
    maanden = {
        "januari": "01", "februari": "02", "maart": "03", "april": "04",
        "mei": "05", "juni": "06", "juli": "07", "augustus": "08",
        "september": "09", "oktober": "10", "november": "11", "december": "12",
    }
    match = re.search(r"(\d{1,2})\s+(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+(\d{4})", date_str.lower())
    if match:
        day = match.group(1).zfill(2)
        month = maanden[match.group(2)]
        year = match.group(3)
        return f"{year}-{month}-{day}"

    return date_str


def determine_uitkomst(item: dict) -> str:
    """Probeer de uitkomst te bepalen uit de tekst."""
    text = item.get("content", "").lower()

    # Zoek in het laatste deel van de tekst (dictum)
    last_part = text[-3000:] if len(text) > 3000 else text

    if "wijst de vordering af" in last_part or "ongegrond" in last_part:
        return "afgewezen"
    if "wijst de vordering toe" in last_part or "gegrond" in last_part:
        return "toegewezen"
    if "gedeeltelijk" in last_part or "deels" in last_part:
        return "deels"

    # Zoek in tags
    tags_text = " ".join(item.get("tags", [])).lower()
    if "ongegrond" in tags_text or "afgewezen" in tags_text:
        return "afgewezen"
    if "gegrond" in tags_text or "toegewezen" in tags_text:
        return "toegewezen"

    return "onbekend"


def determine_bindend(item: dict) -> bool:
    """Bepaal of de uitspraak bindend is."""
    text = (item.get("title", "") + " " + item.get("url", "")).lower()
    return "bindend" in text


def item_to_uitspraak(item: dict) -> dict:
    """Converteer een scraped item naar het dataset-formaat."""
    content = item.get("content", "")
    return {
        "uitspraaknr": extract_uitspraaknr(item),
        "datum": extract_datum(item),
        "type_verzekering": "woonhuisverzekering",
        "kerngeschil": "dekkingsgeschil",
        "uitkomst": determine_uitkomst(item),
        "bindend": determine_bindend(item),
        "commissie": "geschillencommissie",
        "samenvatting": content[:500] + "..." if len(content) > 500 else content,
        "volledige_tekst": content,
        "bron_url": item.get("url", ""),
        "pdf_url": item.get("pdf_url", ""),
        "tags": item.get("tags", []),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_items: List[dict] = []
    seen_urls: set = set()

    # ---- Stap 1: Probeer WordPress REST API ----
    wp_items, wp_success = try_wp_api()
    if wp_success:
        for item in wp_items:
            url = item.get("link", item.get("url", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                # Converteer WP API formaat
                content = ""
                if "content" in item and isinstance(item["content"], dict):
                    content = BeautifulSoup(item["content"].get("rendered", ""), "html.parser").get_text()
                elif isinstance(item.get("content"), str):
                    content = item["content"]

                title = ""
                if "title" in item and isinstance(item["title"], dict):
                    title = item["title"].get("rendered", "")
                elif isinstance(item.get("title"), str):
                    title = item["title"]

                all_items.append({
                    "title": title,
                    "url": url,
                    "date": item.get("date", ""),
                    "content": content,
                    "tags": [],
                    "pdf_url": "",
                })

    # ---- Stap 2: Scrape archiefpagina's ----
    archive_items = scrape_archives()
    archive_urls_to_fetch = []
    for item in archive_items:
        url = item.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            archive_urls_to_fetch.append(url)
            all_items.append(item)

    print(f"\nTotaal items na archief: {len(all_items)}")

    # ---- Stap 3: Haal volledige tekst op voor items zonder content ----
    items_without_content = [i for i, item in enumerate(all_items) if len(item.get("content", "")) < 200]
    print(f"\n=== Stap 3: Volledige tekst ophalen voor {len(items_without_content)} items ===")

    for count, idx in enumerate(items_without_content):
        url = all_items[idx].get("url", "")
        if not url:
            continue

        if count > 0 and count % 50 == 0:
            print(f"  Voortgang: {count}/{len(items_without_content)}")

        full_item = scrape_uitspraak_page(url)
        if full_item and len(full_item.get("content", "")) > 200:
            all_items[idx].update(full_item)

        time.sleep(0.3)

    # ---- Stap 4: Filter op woonhuisverzekering ----
    print(f"\n=== Filtering op woonhuisverzekering ===")
    relevant = [item for item in all_items if is_woonhuis_related(item)]
    print(f"Totaal scraped: {len(all_items)}")
    print(f"Woonhuis-gerelateerd: {len(relevant)}")

    if len(relevant) < MIN_TARGET:
        print(f"\nLet op: slechts {len(relevant)} gevonden (target: {MIN_TARGET})")
        print("Tip: de KIFID-website kan paginering beperken.")
        print("     Probeer ook: python3 scripts/scrape_woonhuis.py --all")

    # ---- Opslaan ----
    raw_path = os.path.join(OUTPUT_DIR, "woonhuis_raw.json")
    # Sla op zonder volledige_tekst in raw (te groot)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(relevant, f, ensure_ascii=False, indent=2)
    print(f"\nRuwe data opgeslagen: {raw_path}")

    uitspraken = [item_to_uitspraak(item) for item in relevant]

    dataset = {
        "meta": {
            "versie": "1.0",
            "laatst_bijgewerkt": datetime.now().strftime("%Y-%m-%d"),
            "aantal": len(uitspraken),
            "bron": "KIFID website - woonhuisverzekering uitspraken",
            "beschrijving": "Woonhuisverzekering uitspraken gescraped van kifid.nl",
        },
        "uitspraken": uitspraken,
    }

    dataset_path = os.path.join(OUTPUT_DIR, "woonhuis_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Dataset opgeslagen: {dataset_path} ({len(uitspraken)} uitspraken)")

    # Samenvatting
    print(f"\n--- Samenvatting ---")
    print(f"Unieke uitspraken: {len(uitspraken)}")
    if len(uitspraken) >= MIN_TARGET:
        print(f"Target van {MIN_TARGET} behaald!")

    uitkomsten: Dict[str, int] = {}
    for u in uitspraken:
        uitkomsten[u["uitkomst"]] = uitkomsten.get(u["uitkomst"], 0) + 1
    for k, v in sorted(uitkomsten.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
