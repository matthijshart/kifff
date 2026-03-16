#!/usr/bin/env python3
"""
Train Model — Analyseer KIFID-uitspraken en genereer een voorspellingsmodel.

Leest dataset.json, berekent statistische patronen per verzekeringstype,
kerngeschil, beslisfactoren-combinaties en juridische grondslagen.
Slaat het resultaat op als data/model.json voor gebruik door de frontend.

Gebruik:
    python scripts/train_model.py
    python scripts/train_model.py --focus woonhuisverzekering
    python scripts/train_model.py --focus woonhuisverzekering,autoverzekering
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASET_PATH = DATA_DIR / "uitspraken" / "dataset.json"
MODEL_PATH = DATA_DIR / "model.json"


def load_dataset(path: Path = DATASET_PATH) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("uitspraken", [])


def afw_rate(items: List[dict]) -> float:
    """Afwijzingspercentage (0-100)."""
    if not items:
        return 50.0
    afw = sum(1 for u in items if u.get("uitkomst") == "afgewezen")
    return round(afw / len(items) * 100, 1)


def outcome_dist(items: List[dict]) -> Dict[str, int]:
    """Verdeling van uitkomsten."""
    dist = Counter(u.get("uitkomst", "onbekend") for u in items)
    return dict(dist)


def avg_bedrag(items: List[dict], field: str = "bedrag_gevorderd") -> float:
    """Gemiddeld bedrag."""
    vals = [u.get(field, 0) for u in items if u.get(field, 0) > 0]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def median_bedrag(items: List[dict], field: str = "bedrag_gevorderd") -> float:
    """Mediaan bedrag."""
    vals = sorted(u.get(field, 0) for u in items if u.get(field, 0) > 0)
    if not vals:
        return 0.0
    n = len(vals)
    if n % 2 == 1:
        return float(vals[n // 2])
    return (vals[n // 2 - 1] + vals[n // 2]) / 2


def toewijzings_ratio(items: List[dict]) -> float:
    """Gemiddelde ratio bedrag_toegewezen / bedrag_gevorderd (als beide > 0)."""
    ratios = []
    for u in items:
        gev = u.get("bedrag_gevorderd", 0)
        toe = u.get("bedrag_toegewezen", 0)
        if gev > 0:
            ratios.append(min(toe / gev, 1.0))
    return round(sum(ratios) / len(ratios) * 100, 1) if ratios else 0.0


# ── Beslisfactoren analyse ──

def analyze_beslisfactor(
    items: List[dict], factor: str, values: Optional[List] = None
) -> Dict[str, Any]:
    """Analyseer impact van een beslisfactor op uitkomst."""
    with_bf = [u for u in items if u.get("beslisfactoren")]
    if not with_bf:
        return {}

    result = {}
    if values is None:
        # Boolean factor
        true_set = [u for u in with_bf if u["beslisfactoren"].get(factor) is True]
        false_set = [u for u in with_bf if u["beslisfactoren"].get(factor) is False]
        if true_set:
            result["true"] = {
                "n": len(true_set),
                "afw_pct": afw_rate(true_set),
                "uitkomst": outcome_dist(true_set),
            }
        if false_set:
            result["false"] = {
                "n": len(false_set),
                "afw_pct": afw_rate(false_set),
                "uitkomst": outcome_dist(false_set),
            }
        if true_set and false_set:
            result["impact"] = round(afw_rate(true_set) - afw_rate(false_set), 1)
    else:
        # Enum factor
        for val in values:
            subset = [u for u in with_bf if u["beslisfactoren"].get(factor) == val]
            if subset:
                result[val] = {
                    "n": len(subset),
                    "afw_pct": afw_rate(subset),
                    "uitkomst": outcome_dist(subset),
                }
    return result


def analyze_factor_combinations(items: List[dict]) -> List[Dict[str, Any]]:
    """Analyseer 2-factor combinaties voor sterkste voorspellers."""
    with_bf = [u for u in items if u.get("beslisfactoren")]
    if len(with_bf) < 10:
        return []

    bool_factors = [
        "polisvoorwaarden_duidelijk",
        "consument_nalatig",
        "verzekeraar_informatieplicht_geschonden",
        "coulance_aangeboden",
    ]
    enum_factors = {
        "bewijs_consument": ["sterk", "gemiddeld", "zwak"],
        "deskundigenrapport": ["geen", "consument", "verzekeraar", "beide", "onafhankelijk"],
    }

    overall_rate = afw_rate(with_bf)
    combos = []

    # Bool × Bool
    for i, f1 in enumerate(bool_factors):
        for f2 in bool_factors[i + 1:]:
            for v1 in [True, False]:
                for v2 in [True, False]:
                    subset = [
                        u for u in with_bf
                        if u["beslisfactoren"].get(f1) is v1
                        and u["beslisfactoren"].get(f2) is v2
                    ]
                    if len(subset) >= 5:
                        rate = afw_rate(subset)
                        impact = round(rate - overall_rate, 1)
                        if abs(impact) >= 8:
                            label = f"{f1}={'ja' if v1 else 'nee'} + {f2}={'ja' if v2 else 'nee'}"
                            combos.append({
                                "factoren": label,
                                "n": len(subset),
                                "afw_pct": rate,
                                "impact": impact,
                                "uitkomst": outcome_dist(subset),
                            })

    # Enum × Bool
    for ef, vals in enum_factors.items():
        for val in vals:
            for bf in bool_factors:
                for bv in [True, False]:
                    subset = [
                        u for u in with_bf
                        if u["beslisfactoren"].get(ef) == val
                        and u["beslisfactoren"].get(bf) is bv
                    ]
                    if len(subset) >= 5:
                        rate = afw_rate(subset)
                        impact = round(rate - overall_rate, 1)
                        if abs(impact) >= 8:
                            label = f"{ef}={val} + {bf}={'ja' if bv else 'nee'}"
                            combos.append({
                                "factoren": label,
                                "n": len(subset),
                                "afw_pct": rate,
                                "impact": impact,
                                "uitkomst": outcome_dist(subset),
                            })

    # Enum × Enum
    ef_list = list(enum_factors.items())
    for i, (ef1, vals1) in enumerate(ef_list):
        for ef2, vals2 in ef_list[i + 1:]:
            for v1 in vals1:
                for v2 in vals2:
                    subset = [
                        u for u in with_bf
                        if u["beslisfactoren"].get(ef1) == v1
                        and u["beslisfactoren"].get(ef2) == v2
                    ]
                    if len(subset) >= 5:
                        rate = afw_rate(subset)
                        impact = round(rate - overall_rate, 1)
                        if abs(impact) >= 8:
                            label = f"{ef1}={v1} + {ef2}={v2}"
                            combos.append({
                                "factoren": label,
                                "n": len(subset),
                                "afw_pct": rate,
                                "impact": impact,
                                "uitkomst": outcome_dist(subset),
                            })

    combos.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return combos[:30]


# ── Juridische grondslag analyse ──

def analyze_grondslagen(items: List[dict]) -> List[Dict[str, Any]]:
    """Analyseer welke juridische grondslagen correleren met welke uitkomsten."""
    grondslag_cases = defaultdict(list)
    for u in items:
        for g in u.get("juridische_grondslag", []):
            g_clean = g.strip().lower()
            if g_clean:
                grondslag_cases[g_clean].append(u)

    results = []
    for g, cases in grondslag_cases.items():
        if len(cases) >= 3:
            results.append({
                "grondslag": g,
                "n": len(cases),
                "afw_pct": afw_rate(cases),
                "uitkomst": outcome_dist(cases),
            })

    results.sort(key=lambda x: x["n"], reverse=True)
    return results[:40]


# ── Tags analyse ──

def analyze_tags(items: List[dict]) -> List[Dict[str, Any]]:
    """Analyseer welke tags correleren met uitkomsten."""
    tag_cases = defaultdict(list)
    for u in items:
        for t in u.get("tags", []):
            t_clean = t.strip().lower()
            if t_clean:
                tag_cases[t_clean].append(u)

    results = []
    for tag, cases in tag_cases.items():
        if len(cases) >= 3:
            results.append({
                "tag": tag,
                "n": len(cases),
                "afw_pct": afw_rate(cases),
                "uitkomst": outcome_dist(cases),
            })

    results.sort(key=lambda x: x["n"], reverse=True)
    return results[:30]


# ── Argument-patronen ──

def analyze_arguments(items: List[dict]) -> Dict[str, Any]:
    """Extraheer veelvoorkomende argumenten en hun correlatie met uitkomst."""
    cons_args = defaultdict(list)
    verz_args = defaultdict(list)

    for u in items:
        uitkomst = u.get("uitkomst", "")
        for arg in u.get("argumenten_consument", []):
            if arg and len(arg) > 10:
                cons_args[arg.strip().lower()].append(uitkomst)
        for arg in u.get("argumenten_verzekeraar", []):
            if arg and len(arg) > 10:
                verz_args[arg.strip().lower()].append(uitkomst)

    def summarize_args(arg_dict: dict, min_count: int = 3) -> List[dict]:
        result = []
        for arg, outcomes in arg_dict.items():
            if len(outcomes) >= min_count:
                afw = sum(1 for o in outcomes if o == "afgewezen")
                result.append({
                    "argument": arg,
                    "n": len(outcomes),
                    "afw_pct": round(afw / len(outcomes) * 100, 1),
                })
        result.sort(key=lambda x: x["n"], reverse=True)
        return result[:20]

    return {
        "consument": summarize_args(cons_args),
        "verzekeraar": summarize_args(verz_args),
    }


# ── Bedrag-ranges ──

def analyze_bedrag_ranges(items: List[dict]) -> List[Dict[str, Any]]:
    """Analyseer uitkomsten per bedragrange."""
    ranges = [
        (0, 1000, "€0-1k"),
        (1000, 5000, "€1k-5k"),
        (5000, 10000, "€5k-10k"),
        (10000, 25000, "€10k-25k"),
        (25000, 50000, "€25k-50k"),
        (50000, 100000, "€50k-100k"),
        (100000, float("inf"), "€100k+"),
    ]
    results = []
    for lo, hi, label in ranges:
        subset = [
            u for u in items
            if lo <= (u.get("bedrag_gevorderd") or 0) < hi
            and (u.get("bedrag_gevorderd") or 0) > 0
        ]
        if subset:
            results.append({
                "range": label,
                "n": len(subset),
                "afw_pct": afw_rate(subset),
                "uitkomst": outcome_dist(subset),
                "gem_toegewezen_ratio": toewijzings_ratio(subset),
            })
    return results


# ── Per kerngeschil deep-dive ──

def analyze_per_kerngeschil(items: List[dict]) -> Dict[str, Any]:
    """Diepteanalyse per kerngeschil."""
    geschillen = defaultdict(list)
    for u in items:
        kg = u.get("kerngeschil", "overig")
        geschillen[kg].append(u)

    result = {}
    for kg, cases in geschillen.items():
        if len(cases) < 2:
            continue
        entry = {
            "n": len(cases),
            "afw_pct": afw_rate(cases),
            "uitkomst": outcome_dist(cases),
            "gem_bedrag_gevorderd": avg_bedrag(cases, "bedrag_gevorderd"),
            "gem_bedrag_toegewezen": avg_bedrag(cases, "bedrag_toegewezen"),
            "toewijzings_ratio": toewijzings_ratio(cases),
        }

        # Beslisfactoren per kerngeschil
        bf_cases = [u for u in cases if u.get("beslisfactoren")]
        if bf_cases:
            entry["beslisfactoren"] = {
                "bewijs_consument": analyze_beslisfactor(
                    cases, "bewijs_consument",
                    ["sterk", "gemiddeld", "zwak", "geen"]
                ),
                "polisvoorwaarden_duidelijk": analyze_beslisfactor(
                    cases, "polisvoorwaarden_duidelijk"
                ),
                "coulance_aangeboden": analyze_beslisfactor(
                    cases, "coulance_aangeboden"
                ),
            }

        # Top grondslagen per kerngeschil
        kg_grondslagen = analyze_grondslagen(cases)
        if kg_grondslagen:
            entry["top_grondslagen"] = kg_grondslagen[:10]

        result[kg] = entry

    return result


# ── Tijdstrend ──

def analyze_trends(items: List[dict]) -> Dict[str, Any]:
    """Analyseer trends over tijd (per jaar)."""
    per_year = defaultdict(list)
    for u in items:
        datum = u.get("datum", "")
        if datum and len(datum) >= 4:
            year = datum[:4]
            per_year[year].append(u)

    result = {}
    for year in sorted(per_year.keys()):
        cases = per_year[year]
        if cases:
            result[year] = {
                "n": len(cases),
                "afw_pct": afw_rate(cases),
                "uitkomst": outcome_dist(cases),
            }
    return result


# ── Hoofdfunctie ──

def train_model(
    items: List[dict],
    focus_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Bouw het complete voorspellingsmodel."""
    model = {
        "meta": {
            "versie": "1.0",
            "getraind_op": date.today().isoformat(),
            "totaal_uitspraken": len(items),
            "focus": focus_types or "alle",
        },
        "globaal": {
            "n": len(items),
            "afw_pct": afw_rate(items),
            "uitkomst": outcome_dist(items),
            "gem_bedrag_gevorderd": avg_bedrag(items, "bedrag_gevorderd"),
            "mediaan_bedrag_gevorderd": median_bedrag(items, "bedrag_gevorderd"),
            "toewijzings_ratio": toewijzings_ratio(items),
        },
        "beslisfactoren": {},
        "factor_combinaties": [],
        "per_kerngeschil": {},
        "bedrag_ranges": [],
        "juridische_grondslagen": [],
        "tags": [],
        "argumenten": {},
        "trends": {},
        "per_type": {},
    }

    # Globale beslisfactoren analyse
    print("  Analyseer beslisfactoren...")
    model["beslisfactoren"] = {
        "bewijs_consument": analyze_beslisfactor(
            items, "bewijs_consument",
            ["sterk", "gemiddeld", "zwak", "geen"]
        ),
        "deskundigenrapport": analyze_beslisfactor(
            items, "deskundigenrapport",
            ["geen", "consument", "verzekeraar", "beide", "onafhankelijk"]
        ),
        "polisvoorwaarden_duidelijk": analyze_beslisfactor(items, "polisvoorwaarden_duidelijk"),
        "consument_nalatig": analyze_beslisfactor(items, "consument_nalatig"),
        "verzekeraar_informatieplicht_geschonden": analyze_beslisfactor(items, "verzekeraar_informatieplicht_geschonden"),
        "coulance_aangeboden": analyze_beslisfactor(items, "coulance_aangeboden"),
    }

    # Factor-combinaties
    print("  Analyseer factor-combinaties...")
    model["factor_combinaties"] = analyze_factor_combinations(items)

    # Per kerngeschil
    print("  Analyseer per kerngeschil...")
    model["per_kerngeschil"] = analyze_per_kerngeschil(items)

    # Bedragranges
    print("  Analyseer bedragranges...")
    model["bedrag_ranges"] = analyze_bedrag_ranges(items)

    # Juridische grondslagen
    print("  Analyseer juridische grondslagen...")
    model["juridische_grondslagen"] = analyze_grondslagen(items)

    # Tags
    print("  Analyseer tags...")
    model["tags"] = analyze_tags(items)

    # Argumenten
    print("  Analyseer argument-patronen...")
    model["argumenten"] = analyze_arguments(items)

    # Trends
    print("  Analyseer trends...")
    model["trends"] = analyze_trends(items)

    # Per verzekeringstype (diepteanalyse)
    print("  Analyseer per verzekeringstype...")
    types = defaultdict(list)
    for u in items:
        t = u.get("type_verzekering", "overig")
        types[t].append(u)

    for vtype, cases in types.items():
        if len(cases) < 3:
            continue
        type_entry = {
            "n": len(cases),
            "afw_pct": afw_rate(cases),
            "uitkomst": outcome_dist(cases),
            "gem_bedrag_gevorderd": avg_bedrag(cases, "bedrag_gevorderd"),
            "mediaan_bedrag_gevorderd": median_bedrag(cases, "bedrag_gevorderd"),
            "toewijzings_ratio": toewijzings_ratio(cases),
            "per_kerngeschil": analyze_per_kerngeschil(cases),
            "beslisfactoren": {
                "bewijs_consument": analyze_beslisfactor(
                    cases, "bewijs_consument",
                    ["sterk", "gemiddeld", "zwak", "geen"]
                ),
                "deskundigenrapport": analyze_beslisfactor(
                    cases, "deskundigenrapport",
                    ["geen", "consument", "verzekeraar", "beide", "onafhankelijk"]
                ),
                "polisvoorwaarden_duidelijk": analyze_beslisfactor(cases, "polisvoorwaarden_duidelijk"),
                "consument_nalatig": analyze_beslisfactor(cases, "consument_nalatig"),
                "verzekeraar_informatieplicht_geschonden": analyze_beslisfactor(cases, "verzekeraar_informatieplicht_geschonden"),
                "coulance_aangeboden": analyze_beslisfactor(cases, "coulance_aangeboden"),
            },
            "factor_combinaties": analyze_factor_combinations(cases),
            "juridische_grondslagen": analyze_grondslagen(cases),
            "tags": analyze_tags(cases),
            "argumenten": analyze_arguments(cases),
            "bedrag_ranges": analyze_bedrag_ranges(cases),
            "trends": analyze_trends(cases),
        }
        model["per_type"][vtype] = type_entry

    return model


def print_summary(model: Dict[str, Any]) -> None:
    """Print een samenvatting van het getrainde model."""
    print("\n" + "=" * 60)
    print("MODEL SAMENVATTING")
    print("=" * 60)

    g = model["globaal"]
    print(f"\nTotaal: {g['n']} uitspraken")
    print(f"Afwijzingspercentage: {g['afw_pct']}%")
    print(f"Uitkomstverdeling: {g['uitkomst']}")
    print(f"Gem. bedrag gevorderd: €{g['gem_bedrag_gevorderd']:,.0f}")
    print(f"Toewijzingsratio: {g['toewijzings_ratio']}%")

    print(f"\nVerzekeringstypen: {len(model['per_type'])}")
    for t, data in sorted(model["per_type"].items(), key=lambda x: x[1]["n"], reverse=True):
        print(f"  {t}: {data['n']}x ({data['afw_pct']}% afw.)")

    print(f"\nKerngeschillen: {len(model['per_kerngeschil'])}")
    for kg, data in sorted(model["per_kerngeschil"].items(), key=lambda x: x[1]["n"], reverse=True):
        print(f"  {kg}: {data['n']}x ({data['afw_pct']}% afw.)")

    combos = model.get("factor_combinaties", [])
    if combos:
        print(f"\nSterkste factor-combinaties ({len(combos)}):")
        for c in combos[:10]:
            direction = "↑" if c["impact"] > 0 else "↓"
            print(f"  {direction} {c['factoren']}: {c['afw_pct']}% afw. (impact {c['impact']:+.1f}, n={c['n']})")

    gs = model.get("juridische_grondslagen", [])
    if gs:
        print(f"\nTop juridische grondslagen ({len(gs)}):")
        for g in gs[:10]:
            print(f"  {g['grondslag']}: {g['n']}x ({g['afw_pct']}% afw.)")


def main():
    parser = argparse.ArgumentParser(description="Train KIFID voorspellingsmodel")
    parser.add_argument(
        "--focus", type=str, default=None,
        help="Kommagescheiden lijst van verzekeringstypen om te focussen, bijv. woonhuisverzekering"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DATASET_PATH),
        help="Pad naar dataset.json"
    )
    parser.add_argument(
        "--output", type=str, default=str(MODEL_PATH),
        help="Pad voor output model.json"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    print(f"Laden dataset: {dataset_path}")
    items = load_dataset(dataset_path)
    print(f"  {len(items)} uitspraken geladen")

    focus_types = None
    if args.focus:
        focus_types = [t.strip() for t in args.focus.split(",")]
        before = len(items)
        items = [u for u in items if u.get("type_verzekering") in focus_types]
        print(f"  Focus op {focus_types}: {len(items)} van {before} uitspraken")

    if not items:
        print("Geen uitspraken gevonden!")
        sys.exit(1)

    print(f"\nTrainen model op {len(items)} uitspraken...")
    model = train_model(items, focus_types)

    print(f"\nOpslaan model: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"  Model opgeslagen ({size_kb:.0f} KB)")

    print_summary(model)
    print(f"\nKlaar! Model beschikbaar op: {output_path}")


if __name__ == "__main__":
    main()
