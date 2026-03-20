#!/usr/bin/env python3
"""
KIFID API Dataset Builder
==========================
Builds the dataset directly from the KIFID Search API without needing
PDF downloads or Claude API parsing. The API returns structured metadata
AND full PDF text (pdfContent), so we can extract everything we need.

For fields requiring text analysis (kerngeschil, uitkomst, beslisfactoren,
argumenten), we use keyword-based heuristics on pdfContent and judgementTags.

Usage:
    python3 scripts/kifid_api_scraper.py                          # Verzekeringen only
    python3 scripts/kifid_api_scraper.py --category ""             # All categories
    python3 scripts/kifid_api_scraper.py --limit 50                # First 50 items
    python3 scripts/kifid_api_scraper.py --page-size 100           # Items per page
    python3 scripts/kifid_api_scraper.py --enrich                  # Use Claude API for deep analysis
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import date, datetime
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://www.kifid.nl"
API_URL = f"{BASE_URL}/api/Search/SearchDecision/"

USER_AGENT = "KIFID-Predictor-Research/1.0"
REQUEST_DELAY = 1.5  # seconds between API pages
REQUEST_TIMEOUT = 30
PAGE_SIZE = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_FILE = DATA_DIR / "uitspraken" / "dataset.json"
SCHEMA_FILE = DATA_DIR / "schema.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kifid-api")

# ---------------------------------------------------------------------------
# .NET ticks → Python datetime
# ---------------------------------------------------------------------------

# .NET DateTime ticks: 100-nanosecond intervals since 0001-01-01
TICKS_EPOCH = 621355968000000000  # ticks between 0001-01-01 and 1970-01-01


def ticks_to_date(ticks: int) -> Optional[str]:
    """Convert .NET DateTime ticks to YYYY-MM-DD string."""
    if not ticks or ticks <= 0:
        return None
    try:
        unix_seconds = (ticks - TICKS_EPOCH) / 10_000_000
        dt = datetime.utcfromtimestamp(unix_seconds)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

def strip_html(html_text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not html_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Category / type_verzekering mapping
# ---------------------------------------------------------------------------

# Map KIFID categories to our schema values
CATEGORY_MAP = {
    "Verzekeringen": None,  # too broad, needs further analysis
    "Bank": None,
    "Beleggen": "beleggingsverzekering",
    "BKR": None,
    "Hypotheken": None,
    "Pensioenen": "levensverzekering",
    "Inkomensverzekeringen": "arbeidsongeschiktheidsverzekering",
}

# Keywords in summary/tags → type_verzekering
TYPE_KEYWORDS = {
    "autoverzekering": ["auto", "motorrijtuig", "voertuig", "casco", "wa-verzekering", "wam", "motorvoertuig"],
    "woonhuisverzekering": ["woonhuis", "opstal", "woningverzekering"],
    "inboedelverzekering": ["inboedel", "huisraad"],
    "reisverzekering": ["reis", "annulering", "vakantie", "repatriering"],
    "aansprakelijkheidsverzekering": ["aansprakelijkheid", "avp", "avb", "bedrijfsaansprakelijkheid"],
    "rechtsbijstandverzekering": ["rechtsbijstand", "juridische bijstand"],
    "levensverzekering": ["levensverzekering", "overlijden", "lijfrente"],
    "arbeidsongeschiktheidsverzekering": ["arbeidsongeschikt", "aov", "inkomensverzekering"],
    "zorgverzekering": ["zorgverzekering", "ziektekost", "zorgpolis"],
    "beleggingsverzekering": ["beleggingsverzekering", "woekerpolis", "unit-linked", "beleggingspolis"],
    "overlijdensrisicoverzekering": ["overlijdensrisico", "orv"],
    "opstalverzekering": ["opstal"],
    "bromfietsverzekering": ["bromfiets", "scooter", "brommer"],
    "brandverzekering": ["brand", "brandverzekering"],
    "transportverzekering": ["transport", "goederen", "vracht"],
}

# Keywords → kerngeschil
KERNGESCHIL_KEYWORDS = {
    "dekkingsweigering": ["dekking geweigerd", "geen dekking", "dekking afgewezen", "niet gedekt", "dekkingsweigering", "buiten dekking"],
    "uitleg_voorwaarden": ["uitleg", "polisvoorwaarden", "interpretatie", "onduidelijk", "contra proferentem"],
    "schadevaststelling": ["schadevaststelling", "hoogte van de schade", "schadebedrag", "taxatie", "expertise", "herstelkosten"],
    "premiegeschil": ["premie", "premieverlaging", "premieverhoging", "no-claim", "schadevrije jaren"],
    "mededelingsplicht": ["mededelingsplicht", "verzwijging", "niet gemeld", "7:928", "vragenformulier"],
    "opzegging": ["opzegging", "opgezegd", "beëindiging", "royement"],
    "zorgplicht": ["zorgplicht", "adviesplicht", "advisering"],
    "informatievoorziening": ["informatieplicht", "voorlichting", "niet geïnformeerd"],
    "clausule": ["clausule", "preventieclausule", "beveiligingse"],
    "vertraging": ["vertraging", "te laat", "termijn", "niet tijdig"],
    "fraude": ["fraude", "opzettelijk", "misleiding"],
    "eigen_gebrek": ["eigen gebrek", "slijtage", "onderhoud"],
}

# Keywords → uitkomst
UITKOMST_KEYWORDS = {
    "toegewezen": ["vordering toegewezen", "klacht gegrond", "consument in het gelijk", "toewijzing", "geheel toegewezen"],
    "afgewezen": ["vordering afgewezen", "klacht ongegrond", "niet gegrond", "afwijzing", "geheel afgewezen"],
    "deels": ["deels toegewezen", "gedeeltelijk", "ten dele", "deels gegrond"],
}


def detect_type_verzekering(text: str, tags: str, category: str) -> str:
    """Detect insurance type from text content, tags, and category."""
    # First check category mapping
    mapped = CATEGORY_MAP.get(category)
    if mapped:
        return mapped

    # Skip non-insurance categories
    if category and category not in ("Verzekeringen", ""):
        return "overig"

    combined = (text + " " + tags).lower()
    for vtype, keywords in TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return vtype
    return "overig"


def detect_kerngeschil(text: str, tags: str) -> str:
    """Detect core dispute type from text and tags."""
    combined = (text + " " + tags).lower()
    for geschil, keywords in KERNGESCHIL_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return geschil
    return "overig"


def detect_uitkomst(text: str, tags: str) -> str:
    """Detect outcome from text and tags."""
    combined = (text + " " + tags).lower()

    # Check judgementTags first (most reliable)
    tags_lower = tags.lower()
    if "afgewezen" in tags_lower or "ongegrond" in tags_lower:
        return "afgewezen"
    if "toegewezen" in tags_lower or "gegrond" in tags_lower:
        if "deels" in tags_lower or "gedeeltelijk" in tags_lower:
            return "deels"
        return "toegewezen"

    # Fall back to text analysis
    for uitkomst, keywords in UITKOMST_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return uitkomst
    return "afgewezen"  # default


def extract_bedragen(text: str) -> tuple:
    """Try to extract gevorderd and toegewezen amounts from text."""
    gevorderd = 0
    toegewezen = 0

    # Look for euro amounts
    amounts = re.findall(r'€\s*([\d.,]+)', text)
    if not amounts:
        amounts = re.findall(r'EUR\s*([\d.,]+)', text, re.IGNORECASE)

    # Parse amounts
    parsed = []
    for a in amounts:
        try:
            # Handle Dutch number format: 1.234,56
            clean = a.replace(".", "").replace(",", ".")
            parsed.append(float(clean))
        except ValueError:
            continue

    if len(parsed) >= 2:
        gevorderd = max(parsed)
        toegewezen = min(parsed) if min(parsed) != max(parsed) else 0
    elif len(parsed) == 1:
        gevorderd = parsed[0]

    return gevorderd, toegewezen


def detect_beslisfactoren(text: str, tags: str) -> dict:
    """Detect decision factors from text using keyword heuristics."""
    combined = (text + " " + tags).lower()

    # Evidence strength
    bewijs = "gemiddeld"
    if any(w in combined for w in ["overtuigend bewijs", "aangetoond", "bewezen"]):
        bewijs = "sterk"
    elif any(w in combined for w in ["onvoldoende bewijs", "niet aangetoond", "niet bewezen"]):
        bewijs = "zwak"

    # Expert report
    deskundigen = "geen"
    if "deskundige" in combined or "expertiserapport" in combined or "taxateur" in combined:
        if "beide" in combined:
            deskundigen = "beide"
        elif "onafhankelijk" in combined:
            deskundigen = "onafhankelijk"
        else:
            deskundigen = "verzekeraar"

    return {
        "bewijs_consument": bewijs,
        "deskundigenrapport": deskundigen,
        "coulance_aangeboden": "coulance" in combined,
        "polisvoorwaarden_duidelijk": "onduidelijk" not in combined and "niet duidelijk" not in combined,
        "consument_nalatig": any(w in combined for w in ["nalatig", "eigen schuld", "niet voldaan aan"]),
        "verzekeraar_informatieplicht_geschonden": any(
            w in combined for w in ["informatieplicht geschonden", "niet geïnformeerd", "onvoldoende geïnformeerd"]
        ),
    }


def extract_tags_from_judgement(judgement_tags: str) -> List[str]:
    """Convert KIFID judgementTags string to a list of tags."""
    if not judgement_tags:
        return []
    return [t.strip() for t in judgement_tags.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# API item → dataset uitspraak
# ---------------------------------------------------------------------------


def api_item_to_uitspraak(item: dict) -> Optional[dict]:
    """Convert a single KIFID API item to our dataset schema."""
    name = item.get("name", "") or item.get("title", "")
    nr_match = re.search(r"(\d{4}-\d{3,4})", name)
    if not nr_match:
        return None

    nr = nr_match.group(1)
    datum = ticks_to_date(item.get("date", 0))
    if not datum:
        datum = "2025-01-01"  # fallback

    summary_html = item.get("summary", "")
    summary = strip_html(summary_html)

    pdf_text = item.get("pdfContent", "") or ""
    tags_str = item.get("judgementTags", "") or ""
    category = item.get("category", "") or ""
    authority = item.get("authority", "") or ""
    defendant = item.get("defendant", "") or ""
    page_url = item.get("url", "") or ""

    # Use the first ~3000 chars of pdfContent for analysis (enough for heuristics)
    analysis_text = summary + " " + pdf_text[:3000]

    type_verz = detect_type_verzekering(analysis_text, tags_str, category)
    kerngeschil = detect_kerngeschil(analysis_text, tags_str)
    uitkomst = detect_uitkomst(analysis_text, tags_str)
    gevorderd, toegewezen = extract_bedragen(analysis_text)

    # Commissie mapping
    commissie = "geschillencommissie"
    if "beroep" in authority.lower():
        commissie = "commissie_van_beroep"

    beslisfactoren = detect_beslisfactoren(analysis_text, tags_str)
    tags = extract_tags_from_judgement(tags_str)

    # Verzekeraar
    verzekeraar = defendant if defendant else None

    uitspraak = {
        "uitspraaknr": nr,
        "datum": datum,
        "type_verzekering": type_verz,
        "kerngeschil": kerngeschil,
        "uitkomst": uitkomst,
        "bedrag_gevorderd": gevorderd,
        "bedrag_toegewezen": toegewezen,
        "bindend": True,  # Most KIFID decisions are bindend advies
        "commissie": commissie,
        "samenvatting": summary[:500] if summary else f"KIFID uitspraak {nr}",
        "argumenten_consument": [],
        "argumenten_verzekeraar": [],
        "juridische_grondslag": [],
        "beslisfactoren": beslisfactoren,
        "tags": tags,
        "bron_url": page_url if page_url else None,
    }

    if verzekeraar:
        uitspraak["verzekeraar"] = verzekeraar

    return uitspraak


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    return session


# ---------------------------------------------------------------------------
# API pagination
# ---------------------------------------------------------------------------


def fetch_all_items(
    session: requests.Session,
    category: str = "Verzekeringen",
    max_items: int = 0,
    page_size: int = PAGE_SIZE,
) -> List[dict]:
    """Paginate through the KIFID API and collect all items."""
    all_items = []
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
            "pageSize": page_size,
        }

        log.info("  Fetching page %d ...", page)
        for attempt in range(4):
            try:
                resp = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt < 3:
                    wait = 2 ** (attempt + 1)
                    log.warning("  Request failed (attempt %d), retrying in %ds: %s", attempt + 1, wait, e)
                    time.sleep(wait)
                else:
                    log.error("  API request failed after 4 attempts on page %d: %s", page, e)
                    return all_items

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            log.error("  Failed to parse JSON on page %d: %s", page, e)
            break

        decision = data.get("decision", {})
        items = decision.get("decisionItemsList", [])

        if total_items is None:
            total_items = data.get("totalItems", 0)
            log.info("  Total items available: %s", total_items)

        if not items:
            break

        all_items.extend(items)
        log.info("  Page %d: %d items (total: %d)", page, len(items), len(all_items))

        if max_items and len(all_items) >= max_items:
            all_items = all_items[:max_items]
            break
        if total_items and len(all_items) >= total_items:
            break
        if len(items) < page_size:
            break

        page += 1
        time.sleep(REQUEST_DELAY)

    log.info("Fetched %d items from API.", len(all_items))
    return all_items


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------


def load_dataset() -> dict:
    if DATASET_FILE.exists():
        return json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    return {
        "meta": {
            "versie": "2.0",
            "laatst_bijgewerkt": date.today().isoformat(),
            "aantal": 0,
            "bron": "KIFID API (kifid.nl/api/Search/SearchDecision)",
            "beschrijving": "Trainingsdata voor de KIFID Insurance Claim Predictor",
        },
        "uitspraken": [],
    }


def save_dataset(dataset: dict) -> None:
    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset["meta"]["laatst_bijgewerkt"] = date.today().isoformat()
    dataset["meta"]["aantal"] = len(dataset["uitspraken"])
    DATASET_FILE.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Saved dataset: %d uitspraken → %s", len(dataset["uitspraken"]), DATASET_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="KIFID API Dataset Builder")
    parser.add_argument("--category", type=str, default="Verzekeringen",
                        help="KIFID category filter (default: Verzekeringen, empty=all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max items to fetch (0=all)")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE,
                        help="Items per API page")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing dataset (default: replace)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("=" * 60)
    log.info("KIFID API Dataset Builder")
    log.info("Category: %s", args.category or "(all)")
    log.info("=" * 60)

    session = create_session()

    # Fetch from API
    items = fetch_all_items(session, category=args.category, max_items=args.limit, page_size=args.page_size)

    if not items:
        log.error("No items fetched from API.")
        sys.exit(1)

    # Convert to dataset format
    log.info("Converting %d items to dataset format...", len(items))

    if args.merge:
        dataset = load_dataset()
        existing_nrs = {u["uitspraaknr"] for u in dataset["uitspraken"]}
    else:
        dataset = {
            "meta": {
                "versie": "2.0",
                "laatst_bijgewerkt": date.today().isoformat(),
                "aantal": 0,
                "bron": "KIFID API (kifid.nl/api/Search/SearchDecision)",
                "beschrijving": "Trainingsdata voor de KIFID Insurance Claim Predictor",
            },
            "uitspraken": [],
        }
        existing_nrs = set()

    converted = 0
    skipped = 0

    for item in items:
        uitspraak = api_item_to_uitspraak(item)
        if not uitspraak:
            skipped += 1
            continue

        nr = uitspraak["uitspraaknr"]
        if nr in existing_nrs:
            # Update existing
            for i, u in enumerate(dataset["uitspraken"]):
                if u["uitspraaknr"] == nr:
                    dataset["uitspraken"][i] = uitspraak
                    break
        else:
            dataset["uitspraken"].append(uitspraak)
            existing_nrs.add(nr)

        converted += 1

    # Sort by uitspraaknr
    dataset["uitspraken"].sort(key=lambda u: u.get("uitspraaknr", ""))

    save_dataset(dataset)

    log.info("=" * 60)
    log.info("Done!")
    log.info("  API items:    %d", len(items))
    log.info("  Converted:    %d", converted)
    log.info("  Skipped:      %d", skipped)
    log.info("  Dataset size: %d uitspraken", len(dataset["uitspraken"]))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
