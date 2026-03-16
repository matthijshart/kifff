#!/usr/bin/env python3
"""
Scraper voor KIFID woonhuisverzekering uitspraken.

Gebruikt de KIFID API om alle woonhuisverzekering-gerelateerde uitspraken
op te halen. De API retourneert direct de volledige tekst (pdfContent),
dus er hoeven geen PDFs gedownload te worden.

Gebruik:
    python3 scripts/scrape_woonhuis.py

Output:
    data/uitspraken/woonhuis_raw.json   – ruwe API-responses
    data/uitspraken/woonhuis_dataset.json – gestructureerd in dataset-formaat
"""

import json
import os
import sys
import time
import re
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    import requests
except ImportError:
    print("requests is niet geinstalleerd. Run: pip install requests")
    sys.exit(1)

API_BASE = "https://www.kifid.nl/api/Search/SearchDecision/"

# Zoektermen die woonhuisverzekeringen opleveren
SEARCH_TERMS = [
    "woonhuisverzekering",
    "opstalverzekering",
    "inboedelverzekering",
    "woonverzekering",
]

PAGE_SIZE = 100
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uitspraken")


def fetch_page(search_term: str, page: int, category: str = "Verzekeringen") -> Optional[dict]:
    """Haal een pagina op van de KIFID API."""
    params = {
        "searchTerm": search_term,
        "category": category,
        "authority": "",
        "targetGroup": "",
        "startDate": 0,
        "endDate": 0,
        "page": page,
        "pageSize": PAGE_SIZE,
    }

    for attempt in range(4):
        try:
            resp = requests.get(API_BASE, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            print(f"  HTTP {resp.status_code} voor {search_term} pagina {page}")
            return None
        except requests.RequestException as e:
            wait = 2 ** (attempt + 1)
            print(f"  Netwerk fout (poging {attempt+1}/4): {e} – wacht {wait}s")
            time.sleep(wait)

    print(f"  Opgegeven na 4 pogingen voor {search_term} pagina {page}")
    return None


def fetch_all(search_term: str) -> List[dict]:
    """Haal alle resultaten op voor een zoekterm."""
    all_items = []
    page = 1

    print(f"\nZoeken: '{search_term}'")
    first = fetch_page(search_term, 1)
    if not first:
        return []

    total = first.get("totalResults", 0)
    items = first.get("items", [])
    all_items.extend(items)
    print(f"  Totaal: {total} resultaten, pagina 1 opgehaald ({len(items)} items)")

    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

    for page in range(2, total_pages + 1):
        time.sleep(0.5)  # netjes zijn voor de server
        data = fetch_page(search_term, page)
        if not data:
            break
        items = data.get("items", [])
        all_items.extend(items)
        print(f"  Pagina {page}/{total_pages} opgehaald ({len(items)} items)")

    return all_items


def extract_uitspraaknr(item: dict) -> str:
    """Extraheer uitspraaknummer uit API-item."""
    # Probeer het uit de title of statementLink te halen
    title = item.get("title", "")
    match = re.search(r"(\d{4}-\d{3,5})", title)
    if match:
        return match.group(1)

    link = item.get("statementLink", "")
    match = re.search(r"(\d{4}-\d{3,5})", link)
    if match:
        return match.group(1)

    return title


def extract_datum(item: dict) -> str:
    """Extraheer datum in YYYY-MM-DD formaat."""
    date_str = item.get("date", "")
    if not date_str:
        return ""

    # API geeft vaak "2024-01-15T00:00:00" of vergelijkbaar
    match = re.search(r"(\d{4}-\d{2}-\d{2})", date_str)
    if match:
        return match.group(1)
    return date_str


def determine_type_verzekering(item: dict) -> str:
    """Bepaal het type verzekering op basis van tekst."""
    text = (
        item.get("pdfContent", "")
        + " "
        + item.get("summary", "")
        + " "
        + item.get("title", "")
    ).lower()

    if "opstal" in text:
        return "woonhuisverzekering"
    if "inboedel" in text:
        return "woonhuisverzekering"
    if "woonhuis" in text:
        return "woonhuisverzekering"
    if "brand" in text and ("woning" in text or "huis" in text):
        return "woonhuisverzekering"
    return "woonhuisverzekering"


def determine_uitkomst(item: dict) -> str:
    """Probeer de uitkomst te bepalen uit tags of tekst."""
    tags = [t.lower() for t in item.get("judgementTags", [])]

    for tag in tags:
        if "toegewezen" in tag or "gegrond" in tag:
            return "toegewezen"
        if "afgewezen" in tag or "ongegrond" in tag:
            return "afgewezen"
        if "deels" in tag or "gedeeltelijk" in tag:
            return "deels"

    text = item.get("pdfContent", "").lower()[-2000:]  # laatste deel bevat vaak uitspraak
    if "wijst de vordering af" in text or "ongegrond" in text:
        return "afgewezen"
    if "wijst de vordering toe" in text or "gegrond" in text:
        return "toegewezen"
    if "gedeeltelijk" in text or "deels" in text:
        return "deels"

    return "afgewezen"  # default


def item_to_uitspraak(item: dict) -> dict:
    """Converteer een API-item naar het dataset-formaat."""
    uitspraak = {
        "uitspraaknr": extract_uitspraaknr(item),
        "datum": extract_datum(item),
        "type_verzekering": determine_type_verzekering(item),
        "kerngeschil": "dekkingsgeschil",  # meest voorkomend, kan later verfijnd
        "uitkomst": determine_uitkomst(item),
        "bindend": True,
        "commissie": "geschillencommissie",
        "samenvatting": item.get("summary", ""),
        "bron_url": item.get("statementLink", ""),
        "tags": item.get("judgementTags", []),
    }
    return uitspraak


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verzamel alle unieke items over alle zoektermen
    seen_ids = set()
    all_raw_items = []

    for term in SEARCH_TERMS:
        items = fetch_all(term)
        for item in items:
            item_id = item.get("statementLink", "") or item.get("title", "")
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                all_raw_items.append(item)
            else:
                print(f"  Duplicaat overgeslagen: {item_id[:60]}")

    print(f"\nTotaal unieke resultaten: {len(all_raw_items)}")

    # Sla ruwe data op
    raw_path = os.path.join(OUTPUT_DIR, "woonhuis_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_items, f, ensure_ascii=False, indent=2)
    print(f"Ruwe data opgeslagen: {raw_path}")

    # Converteer naar dataset-formaat
    uitspraken = [item_to_uitspraak(item) for item in all_raw_items]

    dataset = {
        "meta": {
            "versie": "1.0",
            "laatst_bijgewerkt": datetime.now().strftime("%Y-%m-%d"),
            "aantal": len(uitspraken),
            "bron": "KIFID API - woonhuisverzekering uitspraken",
            "beschrijving": "Woonhuisverzekering uitspraken gescraped via KIFID API",
        },
        "uitspraken": uitspraken,
    }

    dataset_path = os.path.join(OUTPUT_DIR, "woonhuis_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Dataset opgeslagen: {dataset_path} ({len(uitspraken)} uitspraken)")

    # Toon samenvatting
    print(f"\n--- Samenvatting ---")
    print(f"Zoektermen: {', '.join(SEARCH_TERMS)}")
    print(f"Unieke uitspraken: {len(uitspraken)}")

    uitkomsten = {}
    for u in uitspraken:
        uitkomsten[u["uitkomst"]] = uitkomsten.get(u["uitkomst"], 0) + 1
    for k, v in sorted(uitkomsten.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
