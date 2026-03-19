#!/usr/bin/env python3
"""
KIFID Predictor — Backend Server
=================================
Serves the web app and provides:
- Static file serving
- Polisvoorwaarden analysis (Claude AI)
- REST API for system integration (verzekeraar analytics, predictions)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 scripts/server.py
    # Open http://localhost:8000
    # API docs: http://localhost:8000/api/docs
"""

import base64
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import cross_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCHEMA_FILE = DATA_DIR / "schema.json"
DATASET_PATH = DATA_DIR / "uitspraken" / "dataset.json"

app = Flask(__name__, static_folder=str(PROJECT_ROOT), static_url_path="")

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Dataset cache (lazy loaded)
# ---------------------------------------------------------------------------

_dataset_cache: Optional[List[dict]] = None


def get_dataset() -> List[dict]:
    global _dataset_cache
    if _dataset_cache is None:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _dataset_cache = data.get("uitspraken", [])
    return _dataset_cache


# ---------------------------------------------------------------------------
# Verzekeraar name normalization
# ---------------------------------------------------------------------------

VERZEKERAAR_GROUPS = {
    "asr": "ASR",
    "achmea": "Achmea",
    "aegon": "Aegon",
    "nationale-nederlanden": "Nationale-Nederlanden",
    "nn ": "Nationale-Nederlanden",
    "das": "DAS",
    "abn amro": "ABN AMRO",
    "interpolis": "Interpolis",
    "centraal beheer": "Centraal Beheer",
    "univé": "Univé",
    "unive": "Univé",
    "klaverblad": "Klaverblad",
    "arag": "ARAG",
    "allianz": "Allianz",
    "ing": "ING",
    "delta lloyd": "Delta Lloyd",
    "reaal": "Reaal",
    "ohra": "OHRA",
    "sns": "SNS",
    "vivat": "Vivat",
    "unigarant": "Unigarant",
    "bovemij": "Bovemij",
    "zilveren kruis": "Zilveren Kruis",
    "cz": "CZ",
    "menzis": "Menzis",
    "vgz": "VGZ",
}


def normalize_verzekeraar(name: str) -> str:
    """Normalize insurer name to group level."""
    if not name:
        return ""
    # Remove ", gevestigd te..." and "h.o.d.n." suffixes
    cleaned = re.sub(r",?\s*(gevestigd\s+te|h\.o\.d\.n\.).*$", "", name, flags=re.IGNORECASE).strip()
    lower = cleaned.lower()
    for pattern, group in VERZEKERAAR_GROUPS.items():
        if pattern in lower:
            return group
    return cleaned


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
    # Don't serve API paths as static
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(PROJECT_ROOT), path)


# ---------------------------------------------------------------------------
# API: Documentation
# ---------------------------------------------------------------------------


@app.route("/api/docs")
def api_docs():
    """API documentation endpoint."""
    docs = {
        "name": "ClaimWise KIFID Predictor API",
        "version": "1.0",
        "description": "REST API voor KIFID uitspraak-analyse en verzekeraar-benchmarking",
        "endpoints": {
            "GET /api/docs": "Deze documentatie",
            "POST /api/analyze-policy": "Analyseer polisvoorwaarden PDF (vereist ANTHROPIC_API_KEY)",
            "POST /api/predict": "Voorspel uitkomst van een KIFID-zaak",
            "GET /api/verzekeraars": "Lijst van alle verzekeraars met statistieken",
            "GET /api/verzekeraar/<naam>": "Gedetailleerde analytics voor een specifieke verzekeraar",
            "GET /api/benchmark": "Benchmark een verzekeraar tegen de markt",
            "GET /api/stats": "Algemene dataset statistieken",
            "GET /api/trends": "Trend-analyse per jaar",
        },
        "authentication": "Geen (lokale deployment). Voeg authenticatie toe voor productie.",
    }
    return jsonify(docs)


# ---------------------------------------------------------------------------
# API: Predict outcome
# ---------------------------------------------------------------------------


@app.route("/api/predict", methods=["POST"])
@cross_origin()
def api_predict():
    """Predict KIFID ruling outcome based on case parameters.

    Request body (JSON):
    {
        "type_verzekering": "autoverzekering",
        "kerngeschil": "dekkingsweigering",
        "bedrag_gevorderd": 5000,
        "bindend": true,
        "bewijs_consument": "sterk",
        "deskundigenrapport": "geen",
        "coulance_aangeboden": false,
        "context": "optional free text"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    uitspraken = get_dataset()

    type_verz = data.get("type_verzekering", "overig")
    kerngeschil = data.get("kerngeschil", "overig")
    bedrag = data.get("bedrag_gevorderd", 0) or 0
    bindend = data.get("bindend", False)
    bewijs = data.get("bewijs_consument", "gemiddeld")
    expert = data.get("deskundigenrapport", "geen")
    coulance = data.get("coulance_aangeboden", False)

    # Filter relevant cases
    type_cases = [u for u in uitspraken if u.get("type_verzekering") == type_verz]
    combi_cases = [u for u in type_cases if u.get("kerngeschil") == kerngeschil]

    def afw_rate(cases):
        if not cases:
            return 50.0
        afw = sum(1 for u in cases if u.get("uitkomst") == "afgewezen")
        return round(afw / len(cases) * 100, 1)

    def toeg_rate(cases):
        if not cases:
            return 0.0
        toeg = sum(1 for u in cases if u.get("uitkomst") == "toegewezen")
        return round(toeg / len(cases) * 100, 1)

    def deels_rate(cases):
        if not cases:
            return 0.0
        deels = sum(1 for u in cases if u.get("uitkomst") == "deels")
        return round(deels / len(cases) * 100, 1)

    # Base prediction
    overall_afw = afw_rate(uitspraken)
    type_afw = afw_rate(type_cases) if type_cases else overall_afw

    score = type_afw
    factors = []

    factors.append({
        "factor": f"Base rate {type_verz}",
        "value": f"{type_afw}%",
        "n": len(type_cases),
        "impact": 0,
    })

    if combi_cases and len(combi_cases) >= 3:
        combi_afw = afw_rate(combi_cases)
        adj = round((combi_afw - score) * 0.4)
        score += adj
        factors.append({
            "factor": f"{type_verz} + {kerngeschil}",
            "value": f"{combi_afw}%",
            "n": len(combi_cases),
            "impact": adj,
        })

    # Evidence adjustment
    bewijs_map = {"sterk": -8, "gemiddeld": 0, "zwak": 5, "geen": 8}
    ev_adj = bewijs_map.get(bewijs, 0)
    score += ev_adj
    factors.append({"factor": f"Bewijs: {bewijs}", "impact": ev_adj})

    # Coulance
    if coulance:
        score += 3
        factors.append({"factor": "Coulance aangeboden", "impact": 3})

    score = max(5, min(95, round(score)))

    return jsonify({
        "prediction": {
            "afwijzingskans": score,
            "toewijzingskans": 100 - score,
            "aanbeveling": "Uitkeren" if score < 40 else "Afwachten" if score < 65 else "Afwijzen",
            "confidence": "hoog" if (len(combi_cases) >= 10) else "gemiddeld" if (len(type_cases) >= 20) else "laag",
        },
        "factors": factors,
        "comparable_cases": {
            "total": len(combi_cases) if combi_cases else len(type_cases),
            "afgewezen_pct": afw_rate(combi_cases or type_cases),
            "toegewezen_pct": toeg_rate(combi_cases or type_cases),
            "deels_pct": deels_rate(combi_cases or type_cases),
        },
    })


# ---------------------------------------------------------------------------
# API: Verzekeraar analytics
# ---------------------------------------------------------------------------


@app.route("/api/verzekeraars")
@cross_origin()
def api_verzekeraars():
    """List all insurers with summary statistics."""
    uitspraken = get_dataset()
    overall_afw = sum(1 for u in uitspraken if u.get("uitkomst") == "afgewezen") / max(1, len(uitspraken)) * 100

    # Group by normalized name
    groups: Dict[str, List[dict]] = defaultdict(list)
    for u in uitspraken:
        name = normalize_verzekeraar(u.get("verzekeraar", ""))
        if name:
            groups[name].append(u)

    result = []
    for name, cases in sorted(groups.items(), key=lambda x: -len(x[1])):
        n = len(cases)
        if n < 3:
            continue
        afw = sum(1 for u in cases if u.get("uitkomst") == "afgewezen")
        toeg = sum(1 for u in cases if u.get("uitkomst") == "toegewezen")
        deels = sum(1 for u in cases if u.get("uitkomst") == "deels")
        result.append({
            "naam": name,
            "totaal_zaken": n,
            "afgewezen_pct": round(afw / n * 100, 1),
            "toegewezen_pct": round(toeg / n * 100, 1),
            "deels_pct": round(deels / n * 100, 1),
            "vs_markt": round(afw / n * 100 - overall_afw, 1),
        })

    return jsonify({
        "verzekeraars": result,
        "markt_gemiddelde": {
            "afgewezen_pct": round(overall_afw, 1),
            "totaal_zaken": len(uitspraken),
        },
    })


@app.route("/api/verzekeraar/<naam>")
@cross_origin()
def api_verzekeraar_detail(naam: str):
    """Detailed analytics for a specific insurer."""
    uitspraken = get_dataset()
    norm_naam = normalize_verzekeraar(naam)

    # Find all cases for this insurer
    cases = [u for u in uitspraken if normalize_verzekeraar(u.get("verzekeraar", "")) == norm_naam]
    if not cases:
        return jsonify({"error": f"Verzekeraar '{naam}' niet gevonden"}), 404

    n = len(cases)
    overall_n = len(uitspraken)
    overall_afw = sum(1 for u in uitspraken if u.get("uitkomst") == "afgewezen") / max(1, overall_n) * 100

    # Outcome distribution
    afw = sum(1 for u in cases if u.get("uitkomst") == "afgewezen")
    toeg = sum(1 for u in cases if u.get("uitkomst") == "toegewezen")
    deels = sum(1 for u in cases if u.get("uitkomst") == "deels")

    # Per insurance type
    type_breakdown = defaultdict(lambda: {"n": 0, "afgewezen": 0, "toegewezen": 0, "deels": 0})
    for u in cases:
        t = u.get("type_verzekering", "overig")
        type_breakdown[t]["n"] += 1
        type_breakdown[t][u.get("uitkomst", "afgewezen")] += 1

    type_stats = []
    for t, stats in sorted(type_breakdown.items(), key=lambda x: -x[1]["n"]):
        tn = stats["n"]
        # Market average for this type
        market_type = [u for u in uitspraken if u.get("type_verzekering") == t]
        market_afw = sum(1 for u in market_type if u.get("uitkomst") == "afgewezen") / max(1, len(market_type)) * 100
        insurer_afw = stats["afgewezen"] / max(1, tn) * 100
        type_stats.append({
            "type": t,
            "n": tn,
            "afgewezen_pct": round(insurer_afw, 1),
            "toegewezen_pct": round(stats["toegewezen"] / max(1, tn) * 100, 1),
            "deels_pct": round(stats["deels"] / max(1, tn) * 100, 1),
            "markt_afgewezen_pct": round(market_afw, 1),
            "vs_markt": round(insurer_afw - market_afw, 1),
        })

    # Per dispute type
    dispute_breakdown = defaultdict(lambda: {"n": 0, "afgewezen": 0, "toegewezen": 0, "deels": 0})
    for u in cases:
        d = u.get("kerngeschil", "overig")
        dispute_breakdown[d]["n"] += 1
        dispute_breakdown[d][u.get("uitkomst", "afgewezen")] += 1

    dispute_stats = []
    for d, stats in sorted(dispute_breakdown.items(), key=lambda x: -x[1]["n"]):
        dn = stats["n"]
        dispute_stats.append({
            "type": d,
            "n": dn,
            "afgewezen_pct": round(stats["afgewezen"] / max(1, dn) * 100, 1),
            "toegewezen_pct": round(stats["toegewezen"] / max(1, dn) * 100, 1),
            "deels_pct": round(stats["deels"] / max(1, dn) * 100, 1),
        })

    # Year trend
    year_data = defaultdict(lambda: {"n": 0, "afgewezen": 0, "toegewezen": 0, "deels": 0})
    for u in cases:
        datum = u.get("datum", "")
        if datum and len(datum) >= 4:
            year = datum[:4]
            year_data[year]["n"] += 1
            year_data[year][u.get("uitkomst", "afgewezen")] += 1

    trend = []
    for year in sorted(year_data.keys()):
        stats = year_data[year]
        yn = stats["n"]
        trend.append({
            "jaar": year,
            "n": yn,
            "afgewezen_pct": round(stats["afgewezen"] / max(1, yn) * 100, 1),
            "toegewezen_pct": round(stats["toegewezen"] / max(1, yn) * 100, 1),
            "deels_pct": round(stats["deels"] / max(1, yn) * 100, 1),
        })

    # Risk areas: types where this insurer is worse than market
    risk_areas = [t for t in type_stats if t["vs_markt"] < -5 and t["n"] >= 3]

    return jsonify({
        "verzekeraar": norm_naam,
        "totaal_zaken": n,
        "uitkomsten": {
            "afgewezen": afw,
            "afgewezen_pct": round(afw / n * 100, 1),
            "toegewezen": toeg,
            "toegewezen_pct": round(toeg / n * 100, 1),
            "deels": deels,
            "deels_pct": round(deels / n * 100, 1),
        },
        "benchmark": {
            "markt_afgewezen_pct": round(overall_afw, 1),
            "verzekeraar_afgewezen_pct": round(afw / n * 100, 1),
            "verschil": round(afw / n * 100 - overall_afw, 1),
            "beoordeling": "beter" if afw / n * 100 > overall_afw else "slechter",
        },
        "per_type": type_stats,
        "per_geschil": dispute_stats,
        "trend": trend,
        "risicogebieden": risk_areas,
    })


@app.route("/api/benchmark")
@cross_origin()
def api_benchmark():
    """Benchmark an insurer against the market.

    Query params:
    - naam: Verzekeraar name (required)
    - type: Insurance type filter (optional)
    """
    naam = request.args.get("naam", "")
    type_filter = request.args.get("type", "")

    if not naam:
        return jsonify({"error": "Parameter 'naam' is verplicht"}), 400

    uitspraken = get_dataset()
    norm_naam = normalize_verzekeraar(naam)

    insurer_cases = [u for u in uitspraken if normalize_verzekeraar(u.get("verzekeraar", "")) == norm_naam]
    if not insurer_cases:
        return jsonify({"error": f"Verzekeraar '{naam}' niet gevonden"}), 404

    if type_filter:
        insurer_cases = [u for u in insurer_cases if u.get("type_verzekering") == type_filter]
        market_cases = [u for u in uitspraken if u.get("type_verzekering") == type_filter]
    else:
        market_cases = uitspraken

    def calc_stats(cases):
        n = len(cases)
        if n == 0:
            return {"n": 0}
        afw = sum(1 for u in cases if u.get("uitkomst") == "afgewezen")
        toeg = sum(1 for u in cases if u.get("uitkomst") == "toegewezen")
        deels = sum(1 for u in cases if u.get("uitkomst") == "deels")
        return {
            "n": n,
            "afgewezen_pct": round(afw / n * 100, 1),
            "toegewezen_pct": round(toeg / n * 100, 1),
            "deels_pct": round(deels / n * 100, 1),
        }

    insurer_stats = calc_stats(insurer_cases)
    market_stats = calc_stats(market_cases)

    return jsonify({
        "verzekeraar": norm_naam,
        "filter": type_filter or "alle",
        "verzekeraar_stats": insurer_stats,
        "markt_stats": market_stats,
        "verschil_afgewezen": round(
            insurer_stats.get("afgewezen_pct", 0) - market_stats.get("afgewezen_pct", 0), 1
        ) if insurer_stats["n"] > 0 and market_stats["n"] > 0 else None,
    })


# ---------------------------------------------------------------------------
# API: General statistics
# ---------------------------------------------------------------------------


@app.route("/api/stats")
@cross_origin()
def api_stats():
    """General dataset statistics."""
    uitspraken = get_dataset()
    n = len(uitspraken)

    uitkomsten = Counter(u.get("uitkomst", "onbekend") for u in uitspraken)
    types = Counter(u.get("type_verzekering", "overig") for u in uitspraken)
    geschillen = Counter(u.get("kerngeschil", "overig") for u in uitspraken)

    bedragen = [u["bedrag_gevorderd"] for u in uitspraken if u.get("bedrag_gevorderd")]
    gem_bedrag = round(sum(bedragen) / max(1, len(bedragen)), 2) if bedragen else 0

    return jsonify({
        "totaal_uitspraken": n,
        "uitkomsten": dict(uitkomsten.most_common()),
        "type_verzekering": dict(types.most_common(20)),
        "kerngeschil": dict(geschillen.most_common(15)),
        "bedragen": {
            "bekend": len(bedragen),
            "onbekend": n - len(bedragen),
            "gemiddeld": gem_bedrag,
            "mediaan": round(sorted(bedragen)[len(bedragen) // 2], 2) if bedragen else 0,
        },
    })


# ---------------------------------------------------------------------------
# API: Trends
# ---------------------------------------------------------------------------


@app.route("/api/trends")
@cross_origin()
def api_trends():
    """Year-over-year trend analysis."""
    uitspraken = get_dataset()
    type_filter = request.args.get("type", "")

    if type_filter:
        uitspraken = [u for u in uitspraken if u.get("type_verzekering") == type_filter]

    year_data = defaultdict(lambda: {"n": 0, "afgewezen": 0, "toegewezen": 0, "deels": 0})
    for u in uitspraken:
        datum = u.get("datum", "")
        if datum and len(datum) >= 4:
            year = datum[:4]
            year_data[year]["n"] += 1
            year_data[year][u.get("uitkomst", "afgewezen")] += 1

    trends = []
    for year in sorted(year_data.keys()):
        stats = year_data[year]
        yn = stats["n"]
        trends.append({
            "jaar": int(year),
            "n": yn,
            "afgewezen_pct": round(stats["afgewezen"] / max(1, yn) * 100, 1),
            "toegewezen_pct": round(stats["toegewezen"] / max(1, yn) * 100, 1),
            "deels_pct": round(stats["deels"] / max(1, yn) * 100, 1),
        })

    return jsonify({
        "filter": type_filter or "alle",
        "trends": trends,
    })


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
    print("API docs:          http://localhost:8000/api/docs")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8000, debug=False)
