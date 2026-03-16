#!/usr/bin/env python3
"""
KIFID PDF Parser — Claude API
==============================
Parses downloaded KIFID uitspraak PDFs into structured JSON using Claude's
vision capabilities (PDF → structured data).

Reads PDFs from data/pdfs/, sends each to Claude API, and outputs structured
data matching data/schema.json. Results are merged into
data/uitspraken/dataset.json.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

    python3 scripts/parse_pdfs.py                  # Parse all unprocessed PDFs
    python3 scripts/parse_pdfs.py --limit 5        # Parse max 5 PDFs
    python3 scripts/parse_pdfs.py --file data/pdfs/uitspraak-2025-0448.pdf  # Parse one
    python3 scripts/parse_pdfs.py --dry-run        # Show what would be parsed
    python3 scripts/parse_pdfs.py --reparse        # Re-parse already processed PDFs
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
REQUEST_DELAY = 1.0  # seconds between API calls (rate limiting)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
DATASET_FILE = DATA_DIR / "uitspraken" / "dataset.json"
PARSE_LOG_FILE = DATA_DIR / "parse_log.json"
SCHEMA_FILE = DATA_DIR / "schema.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kifid-parser")

# ---------------------------------------------------------------------------
# Schema & Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Je bent een juridisch AI-assistent gespecialiseerd in Nederlandse verzekeringsgeschillen \
bij KIFID (Klachteninstituut Financiële Dienstverlening). Je taak is om KIFID-uitspraken \
(PDF-documenten) te analyseren en om te zetten in gestructureerde JSON-data.

Je bent nauwkeurig, objectief en volledig. Je extraheert alleen informatie die \
daadwerkelijk in het document staat. Als iets niet te bepalen is uit het document, \
gebruik dan null."""

def build_extraction_prompt(schema: dict) -> str:
    """Build the extraction prompt with enum values from schema."""
    props = schema["properties"]

    type_verzekering_vals = props["type_verzekering"]["enum"]
    kerngeschil_vals = props["kerngeschil"]["enum"]
    uitkomst_vals = props["uitkomst"]["enum"]
    commissie_vals = props["commissie"]["enum"]
    bewijs_vals = props["beslisfactoren"]["properties"]["bewijs_consument"]["enum"]
    deskundigen_vals = props["beslisfactoren"]["properties"]["deskundigenrapport"]["enum"]

    return f"""\
Analyseer deze KIFID-uitspraak en extraheer de volgende gegevens als JSON.

## Velden

1. **uitspraaknr** (string, verplicht): Het uitspraaknummer, bijv. "2025-0448"
2. **datum** (string, verplicht): Datum van de uitspraak in YYYY-MM-DD formaat
3. **type_verzekering** (string, verplicht): Eén van: {json.dumps(type_verzekering_vals)}
4. **kerngeschil** (string, verplicht): Eén van: {json.dumps(kerngeschil_vals)}
5. **uitkomst** (string, verplicht): Eén van: {json.dumps(uitkomst_vals)}
   - "toegewezen" = consument krijgt (grotendeels) gelijk
   - "afgewezen" = verzekeraar krijgt gelijk
   - "deels" = gedeeltelijk toegewezen
6. **bedrag_gevorderd** (number): Door consument gevorderd bedrag in euro's. 0 als niet van toepassing.
7. **bedrag_toegewezen** (number): Door commissie toegewezen bedrag in euro's. 0 als niets toegewezen.
8. **bindend** (boolean): true = bindend advies, false = niet-bindend advies
9. **commissie** (string): Eén van: {json.dumps(commissie_vals)}
10. **verzekeraar** (string): Naam van de verzekeraar indien genoemd, anders null
11. **samenvatting** (string): Korte samenvatting (1-3 zinnen) van de casus en het oordeel
12. **argumenten_consument** (array of strings): De kernargumenten van de consument (2-5 items)
13. **argumenten_verzekeraar** (array of strings): De kernargumenten/verweer van de verzekeraar (2-5 items)
14. **juridische_grondslag** (array of strings): Relevante wetsartikelen, polisvoorwaarden, jurisprudentie
15. **beslisfactoren** (object):
    - bewijs_consument: Eén van {json.dumps(bewijs_vals)}
    - deskundigenrapport: Eén van {json.dumps(deskundigen_vals)}
    - coulance_aangeboden (boolean): Heeft verzekeraar coulance aangeboden?
    - polisvoorwaarden_duidelijk (boolean): Waren de polisvoorwaarden duidelijk?
    - consument_nalatig (boolean): Was de consument nalatig?
    - verzekeraar_informatieplicht_geschonden (boolean): Heeft verzekeraar informatieplicht geschonden?
16. **tags** (array of strings): Relevante tags voor categorisatie (bijv. "stormschade", "woekerpolis", "whiplash")
17. **bron_url** (string): Laat dit veld weg, wordt later ingevuld.

## Regels
- Antwoord ALLEEN met valide JSON, geen markdown codeblocks, geen uitleg.
- Gebruik null voor velden die niet uit het document te bepalen zijn.
- Bedragen altijd als getal (geen euroteken, geen duizendtallen-scheidingsteken).
- Gebruik uitsluitend de opgegeven enum-waarden.
- Samenvatting in het Nederlands."""


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------


def read_pdf_base64(pdf_path: Path) -> str:
    """Read a PDF file and return base64-encoded content."""
    return base64.standard_b64encode(pdf_path.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------


def parse_pdf_with_claude(
    client: anthropic.Anthropic,
    pdf_path: Path,
    schema: dict,
    model: str = MODEL,
) -> dict | None:
    """Send a PDF to Claude and get structured JSON back."""
    pdf_b64 = read_pdf_base64(pdf_path)
    extraction_prompt = build_extraction_prompt(schema)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": extraction_prompt,
                        },
                    ],
                }
            ],
        )
    except anthropic.APIError as e:
        log.error("API error for %s: %s", pdf_path.name, e)
        return None

    # Extract text from response
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    # Parse JSON from response (strip markdown fences if present)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove first line
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        log.error("Failed to parse JSON from Claude response for %s: %s", pdf_path.name, e)
        log.debug("Response text: %s", text[:500])
        return None

    # Basic validation
    required = ["uitspraaknr", "datum", "type_verzekering", "kerngeschil", "uitkomst"]
    missing = [f for f in required if not result.get(f)]
    if missing:
        log.warning("Missing required fields for %s: %s", pdf_path.name, missing)

    return result


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------


def load_dataset() -> dict:
    """Load existing dataset or create empty one."""
    if DATASET_FILE.exists():
        return json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    return {
        "meta": {
            "versie": "1.0",
            "laatst_bijgewerkt": date.today().isoformat(),
            "aantal": 0,
            "bron": "KIFID openbaar uitsprakenregister + Claude API extractie",
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
    log.info("Saved dataset with %d uitspraken to %s", len(dataset["uitspraken"]), DATASET_FILE)


def merge_into_dataset(dataset: dict, new_item: dict) -> dict:
    """Add or update an uitspraak in the dataset."""
    existing = {u["uitspraaknr"]: i for i, u in enumerate(dataset["uitspraken"])}
    nr = new_item.get("uitspraaknr")

    if nr in existing:
        dataset["uitspraken"][existing[nr]] = new_item
        log.info("  Updated existing: %s", nr)
    else:
        dataset["uitspraken"].append(new_item)
        log.info("  Added new: %s", nr)

    # Keep sorted by uitspraaknr
    dataset["uitspraken"].sort(key=lambda u: u.get("uitspraaknr", ""))
    return dataset


def load_parse_log() -> dict:
    if PARSE_LOG_FILE.exists():
        return json.loads(PARSE_LOG_FILE.read_text(encoding="utf-8"))
    return {"parsed": [], "failed": [], "last_run": None}


def save_parse_log(log_data: dict) -> None:
    PARSE_LOG_FILE.write_text(
        json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def get_pdfs_to_parse(parse_log: dict, reparse: bool = False) -> list[Path]:
    """Get list of PDFs that need parsing."""
    if not PDF_DIR.exists():
        return []

    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if reparse:
        return all_pdfs

    already_parsed = set(parse_log.get("parsed", []))
    return [p for p in all_pdfs if p.name not in already_parsed]


def main():
    parser = argparse.ArgumentParser(description="KIFID PDF Parser — Claude API")
    parser.add_argument("--limit", type=int, default=None, help="Max number of PDFs to parse")
    parser.add_argument("--file", type=str, default=None, help="Parse a single PDF file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be parsed")
    parser.add_argument("--reparse", action="store_true", help="Re-parse already processed PDFs")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Claude model to use (default: {MODEL})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        log.error("ANTHROPIC_API_KEY environment variable not set.")
        log.error("Export your key: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Load schema
    if not SCHEMA_FILE.exists():
        log.error("Schema file not found: %s", SCHEMA_FILE)
        sys.exit(1)
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))

    # Determine PDFs to parse
    parse_log = load_parse_log()

    if args.file:
        pdf_path = Path(args.file).resolve()
        if not pdf_path.exists():
            log.error("File not found: %s", pdf_path)
            sys.exit(1)
        pdfs = [pdf_path]
    else:
        pdfs = get_pdfs_to_parse(parse_log, reparse=args.reparse)

    if args.limit:
        pdfs = pdfs[: args.limit]

    if not pdfs:
        log.info("No PDFs to parse. Download PDFs first with kifid_scraper.py.")
        return

    log.info("=" * 60)
    log.info("KIFID PDF Parser — %d PDFs to process", len(pdfs))
    log.info("Model: %s", args.model)
    log.info("=" * 60)

    if args.dry_run:
        for p in pdfs:
            print(f"  Would parse: {p.name} ({p.stat().st_size // 1024} KB)")
        return

    # Init API client
    client = anthropic.Anthropic(api_key=api_key)
    model = args.model

    dataset = load_dataset()
    parsed_count = 0
    failed_count = 0

    for i, pdf_path in enumerate(pdfs, 1):
        log.info("[%d/%d] Parsing: %s", i, len(pdfs), pdf_path.name)

        result = parse_pdf_with_claude(client, pdf_path, schema, model=model)

        if result:
            # Add source URL from urls.json if available
            urls_file = DATA_DIR / "urls.json"
            if urls_file.exists():
                urls = json.loads(urls_file.read_text(encoding="utf-8"))
                nr = result.get("uitspraaknr")
                for u in urls:
                    if u.get("uitspraaknr") == nr and u.get("pdf_url"):
                        result["bron_url"] = u["pdf_url"]
                        break

            dataset = merge_into_dataset(dataset, result)
            parse_log["parsed"].append(pdf_path.name)
            parsed_count += 1

            # Save after each successful parse (resume support)
            save_dataset(dataset)
        else:
            parse_log["failed"].append({"file": pdf_path.name, "date": date.today().isoformat()})
            failed_count += 1

        # Rate limiting
        if i < len(pdfs):
            time.sleep(REQUEST_DELAY)

    # Save log
    parse_log["last_run"] = date.today().isoformat()
    parse_log["stats"] = {
        "total_parsed": parsed_count,
        "total_failed": failed_count,
        "dataset_size": len(dataset["uitspraken"]),
    }
    save_parse_log(parse_log)

    # Summary
    log.info("=" * 60)
    log.info("Summary:")
    log.info("  Parsed:         %d", parsed_count)
    log.info("  Failed:         %d", failed_count)
    log.info("  Dataset total:  %d uitspraken", len(dataset["uitspraken"]))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
