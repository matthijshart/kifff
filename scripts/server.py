#!/usr/bin/env python3
"""
KIFID Predictor — Backend Server
=================================
Serves the web app and provides API endpoints for polisvoorwaarden analysis.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 scripts/server.py
    # Open http://localhost:8000
"""

import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import cross_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCHEMA_FILE = DATA_DIR / "schema.json"

app = Flask(__name__, static_folder=str(PROJECT_ROOT), static_url_path="")

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Claude API client (lazy init)
# ---------------------------------------------------------------------------

_client = None


def get_client():
    global _client
    if _client is None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory(str(PROJECT_ROOT), "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(str(PROJECT_ROOT), path)


# ---------------------------------------------------------------------------
# Polisvoorwaarden Analysis API
# ---------------------------------------------------------------------------

POLICY_ANALYSIS_PROMPT = """\
Je bent een juridisch AI-assistent gespecialiseerd in Nederlandse verzekeringspolissen en KIFID-geschillen.

Analyseer deze polisvoorwaarden en geef een gestructureerde analyse in JSON-formaat.

## Wat je moet doen:

1. **Identificeer het type verzekering** (autoverzekering, woonhuisverzekering, etc.)
2. **Lijst de belangrijkste dekkingen** op (wat is wel gedekt)
3. **Lijst de belangrijkste uitsluitingen** op (wat is NIET gedekt)
4. **Identificeer risicovolle clausules** — clausules die vaak tot KIFID-geschillen leiden:
   - Vage of onduidelijke formuleringen
   - Brede uitsluitingen
   - Strikte meldplichten
   - Eenzijdige wijzigingsbevoegdheden
   - Vervaltermijnen
5. **Geef een risicoscore** van 1-10 (1=weinig geschilrisico, 10=veel geschilrisico)
6. **Geef concrete aanbevelingen** voor de verzekeraar om geschillen te voorkomen

## Output formaat (JSON):

{
  "type_verzekering": "string",
  "verzekeraar": "string of null",
  "product_naam": "string of null",
  "dekkingen": ["string", ...],
  "uitsluitingen": ["string", ...],
  "risicovolle_clausules": [
    {
      "clausule": "korte beschrijving",
      "artikel": "artikelnummer indien beschikbaar",
      "risico": "waarom dit tot geschillen kan leiden",
      "ernst": "hoog/middel/laag"
    }
  ],
  "risicoscore": 7,
  "risico_toelichting": "korte uitleg van de score",
  "aanbevelingen": ["string", ...],
  "samenvatting": "korte samenvatting van de polisvoorwaarden (2-3 zinnen)"
}

## Regels:
- Antwoord ALLEEN met valide JSON, geen markdown codeblocks, geen uitleg.
- Wees specifiek en noem artikelnummers waar mogelijk.
- Focus op clausules die in de praktijk tot KIFID-klachten leiden.
- Samenvatting in het Nederlands."""


@app.route("/api/analyze-policy", methods=["POST"])
def analyze_policy():
    """Analyze uploaded polisvoorwaarden PDF with Claude."""
    if "file" not in request.files:
        return jsonify({"error": "Geen bestand geüpload"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Geen bestand geselecteerd"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Alleen PDF-bestanden worden ondersteund"}), 400

    try:
        # Read PDF and encode as base64
        pdf_bytes = file.read()
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        # Send to Claude
        client = get_client()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system="Je bent een juridisch AI-assistent gespecialiseerd in Nederlandse verzekeringspolissen en KIFID-geschillen. Je bent nauwkeurig, objectief en volledig.",
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
                            "text": POLICY_ANALYSIS_PROMPT,
                        },
                    ],
                }
            ],
        )

        # Extract text from response
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Parse JSON
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        result = json.loads(text)
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({"error": "Claude gaf geen valide JSON terug", "details": str(e)}), 500
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Analyse mislukt", "details": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("WAARSCHUWING: ANTHROPIC_API_KEY niet ingesteld.")
        print("Polisvoorwaarden-analyse werkt niet zonder API key.")
        print("Stel in met: export ANTHROPIC_API_KEY=sk-ant-...")
        print()

    print("=" * 50)
    print("KIFID Predictor — http://localhost:8000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8000, debug=False)
