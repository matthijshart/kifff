#!/usr/bin/env python3
"""
Train Ensemble Model — Geavanceerd KIFID voorspellingsmodel.

Combineert drie modellen:
1. TF-IDF similarity met eerdere zaken
2. Logistic Regression op gestructureerde features
3. Naive Bayes tekst-classifier op samenvattingen

Met feature engineering:
- Juridische termen frequentie
- Tekststatistieken (lengte, complexiteit)
- Metadata encoding (categorie, commissie, bindend, jaar)
- Interactie-features

Cross-validatie voor betrouwbare prestatieschatting.

Gebruik:
    python scripts/train_ensemble.py
    python scripts/train_ensemble.py --dataset data/uitspraken/dataset.json
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASET_PATH = DATA_DIR / "uitspraken" / "dataset.json"
MODEL_PATH = DATA_DIR / "model.json"

# ── Juridische termen voor feature engineering ──
JURIDISCHE_TERMEN = [
    "bewijslast",
    "redelijkheid en billijkheid",
    "zorgplicht",
    "contra proferentem",
    "art. 6:248 bw",
    "art. 7:940 bw",
    "avg",
    "onaanvaardbaar",
    "haviltex",
    "mededelingsplicht",
    "informatieplicht",
    "klachtplicht",
    "schadebeperkingsplicht",
    "dekkingsomvang",
    "eigen schuld",
    "eigen gebrek",
    "merkelijke schuld",
    "grove nalatigheid",
    "opzet",
    "molest",
    "polisvoorwaarden",
    "uitsluitingsclausule",
    "kernbeding",
    "algemene voorwaarden",
    "verjaringstermijn",
    "schending",
    "onredelijk bezwarend",
    "dwingend recht",
    "art. 7:941 bw",
    "art. 7:943 bw",
    "art. 7:952 bw",
    "art. 7:953 bw",
    "art. 6:233 bw",
    "art. 6:236 bw",
    "art. 6:237 bw",
    "coulance",
    "bindend advies",
    "geschillencommissie",
    "commissie van beroep",
    "wft",
    "bgfo",
]

# ── Stopwoorden (Nederlands) ──
STOPWOORDEN = set(
    "de het een van in is dat op te zijn voor met als door "
    "aan er maar om ook dan tot uit bij niet of over nog "
    "dit wel geen werd die kan naar meer heeft hem had "
    "haar ze we wat men heb mijn ons hun u zo al zo nu "
    "was worden was ge werd mij werd werden heeft"
    .split()
)


def load_dataset(path: Path = DATASET_PATH) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("uitspraken", [])


# ── Feature Engineering ──

def extract_text(u: dict) -> str:
    """Combineer alle beschikbare tekst van een uitspraak."""
    parts = [
        u.get("samenvatting", ""),
        " ".join(u.get("tags", [])),
        " ".join(u.get("juridische_grondslag", [])),
        " ".join(u.get("argumenten_consument", [])),
        " ".join(u.get("argumenten_verzekeraar", [])),
    ]
    return " ".join(p for p in parts if p).lower()


def count_juridische_termen(text: str) -> Dict[str, int]:
    """Tel juridische termen in tekst."""
    text_lower = text.lower()
    counts = {}
    for term in JURIDISCHE_TERMEN:
        c = text_lower.count(term)
        if c > 0:
            counts[term] = c
    return counts


def text_complexity(text: str) -> Dict[str, float]:
    """Bereken tekststatistieken."""
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    n_sentences = max(1, len(re.split(r'[.!?]+', text)))

    # Gemiddelde woordlengte
    avg_word_len = sum(len(w) for w in words) / max(1, n_words)

    # Lexicale diversiteit (type-token ratio)
    unique_words = len(set(w.lower() for w in words if len(w) > 2))
    ttr = unique_words / max(1, n_words)

    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "n_sentences": n_sentences,
        "avg_word_len": round(avg_word_len, 2),
        "words_per_sentence": round(n_words / n_sentences, 2),
        "lexical_diversity": round(ttr, 3),
    }


def encode_categorical(value: str, categories: List[str]) -> List[int]:
    """One-hot encode."""
    return [1 if value == cat else 0 for cat in categories]


# ── Feature matrix bouwen ──

INSURANCE_TYPES = [
    "autoverzekering", "woonhuisverzekering", "inboedelverzekering",
    "reisverzekering", "aansprakelijkheidsverzekering", "rechtsbijstandverzekering",
    "levensverzekering", "arbeidsongeschiktheidsverzekering", "zorgverzekering",
    "beleggingsverzekering", "overlijdensrisicoverzekering", "opstalverzekering",
    "bromfietsverzekering", "brandverzekering", "transportverzekering", "overig",
]

DISPUTE_TYPES = [
    "dekkingsweigering", "uitleg_voorwaarden", "schadevaststelling",
    "premiegeschil", "mededelingsplicht", "opzegging", "zorgplicht",
    "informatievoorziening", "clausule", "vertraging", "fraude",
    "eigen_gebrek", "overig",
]

EVIDENCE_LEVELS = ["sterk", "gemiddeld", "zwak", "geen"]
EXPERT_TYPES = ["geen", "consument", "verzekeraar", "beide", "onafhankelijk"]


def build_feature_vector(u: dict, text: str) -> List[float]:
    """Bouw een gestructureerde feature vector voor een uitspraak."""
    features = []

    # 1. Type verzekering (one-hot)
    features.extend(encode_categorical(u.get("type_verzekering", "overig"), INSURANCE_TYPES))

    # 2. Kerngeschil (one-hot)
    features.extend(encode_categorical(u.get("kerngeschil", "overig"), DISPUTE_TYPES))

    # 3. Bedrag features
    bedrag = u.get("bedrag_gevorderd", 0) or 0
    features.append(math.log1p(bedrag))  # log-bedrag
    features.append(1 if bedrag > 50000 else 0)  # hoog bedrag flag
    features.append(1 if bedrag > 100000 else 0)  # zeer hoog bedrag flag
    features.append(1 if 0 < bedrag <= 5000 else 0)  # laag bedrag flag

    # 4. Bindend advies
    features.append(1 if u.get("bindend") else 0)

    # 5. Commissie type
    features.append(1 if u.get("commissie") == "commissie_van_beroep" else 0)

    # 6. Jaar (genormaliseerd)
    datum = u.get("datum", "")
    jaar = int(datum[:4]) if datum and len(datum) >= 4 else 2020
    features.append((jaar - 2000) / 26.0)  # genormaliseerd 0-1

    # 7. Beslisfactoren
    bf = u.get("beslisfactoren", {})

    # Bewijs consument (ordinal encoding)
    bewijs_map = {"sterk": 3, "gemiddeld": 2, "zwak": 1, "geen": 0}
    features.append(bewijs_map.get(bf.get("bewijs_consument", "gemiddeld"), 2) / 3.0)

    # Deskundigenrapport (one-hot)
    features.extend(encode_categorical(bf.get("deskundigenrapport", "geen"), EXPERT_TYPES))

    # Boolean beslisfactoren
    features.append(1 if bf.get("polisvoorwaarden_duidelijk") else 0)
    features.append(1 if bf.get("consument_nalatig") else 0)
    features.append(1 if bf.get("verzekeraar_informatieplicht_geschonden") else 0)
    features.append(1 if bf.get("coulance_aangeboden") else 0)

    # 8. Juridische termen (counts)
    term_counts = count_juridische_termen(text)
    for term in JURIDISCHE_TERMEN:
        features.append(term_counts.get(term, 0))

    # 9. Tekststatistieken
    stats = text_complexity(text)
    features.append(stats["n_words"] / 100.0)  # genormaliseerd
    features.append(stats["avg_word_len"] / 10.0)
    features.append(stats["words_per_sentence"] / 30.0)
    features.append(stats["lexical_diversity"])

    # 10. Interactie-features
    bewijs_score = bewijs_map.get(bf.get("bewijs_consument", "gemiddeld"), 2)

    # Interactie: sterk bewijs + hoog bedrag
    features.append(bewijs_score / 3.0 * math.log1p(bedrag) / 15.0)

    # Interactie: type × geschil hash (proxy voor combinatie-effect)
    type_idx = INSURANCE_TYPES.index(u.get("type_verzekering", "overig")) if u.get("type_verzekering", "overig") in INSURANCE_TYPES else 0
    disp_idx = DISPUTE_TYPES.index(u.get("kerngeschil", "overig")) if u.get("kerngeschil", "overig") in DISPUTE_TYPES else 0
    features.append(type_idx * len(DISPUTE_TYPES) + disp_idx)  # combinatie-index

    # Interactie: nalatigheid × bewijs
    features.append((1 if bf.get("consument_nalatig") else 0) * bewijs_score / 3.0)

    # Interactie: coulance × bedrag
    features.append((1 if bf.get("coulance_aangeboden") else 0) * math.log1p(bedrag) / 15.0)

    # Interactie: informatieplicht × polisvoorwaarden
    features.append(
        (1 if bf.get("verzekeraar_informatieplicht_geschonden") else 0) *
        (0 if bf.get("polisvoorwaarden_duidelijk") else 1)
    )

    # 11. Heeft tags
    features.append(1 if u.get("tags") else 0)
    features.append(len(u.get("tags", [])) / 5.0)

    # 12. Totaal juridische termen
    total_jur = sum(term_counts.values())
    features.append(total_jur / 5.0)

    return features


def get_feature_names() -> List[str]:
    """Retourneer feature namen voor interpretatie."""
    names = []
    names.extend([f"type_{t}" for t in INSURANCE_TYPES])
    names.extend([f"geschil_{d}" for d in DISPUTE_TYPES])
    names.extend(["log_bedrag", "hoog_bedrag", "zeer_hoog_bedrag", "laag_bedrag"])
    names.append("bindend")
    names.append("commissie_van_beroep")
    names.append("jaar_norm")
    names.append("bewijs_ordinal")
    names.extend([f"expert_{e}" for e in EXPERT_TYPES])
    names.extend(["pv_duidelijk", "consument_nalatig", "info_geschonden", "coulance"])
    names.extend([f"jur_{t.replace(' ', '_')}" for t in JURIDISCHE_TERMEN])
    names.extend(["text_words", "text_avg_wordlen", "text_words_per_sent", "text_diversity"])
    names.extend(["interact_bewijs_bedrag", "interact_type_geschil",
                   "interact_nalatig_bewijs", "interact_coulance_bedrag",
                   "interact_info_pv"])
    names.extend(["heeft_tags", "n_tags"])
    names.append("totaal_jur_termen")
    return names


# ── Outcome encoding ──

OUTCOME_MAP = {"afgewezen": 0, "toegewezen": 1, "deels": 2}
OUTCOME_NAMES = ["afgewezen", "toegewezen", "deels"]


def encode_outcome(outcome: str) -> int:
    return OUTCOME_MAP.get(outcome, 0)


# ── Model 1: TF-IDF Similarity ──

def train_tfidf_model(
    texts: List[str], outcomes: np.ndarray, max_features: int = 800
) -> Dict[str, Any]:
    """Train TF-IDF model en bereken centroids per uitkomstklasse."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=list(STOPWOORDEN),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out().tolist()
    idf = vectorizer.idf_.tolist()

    # Bereken centroids per klasse
    centroids = {}
    for cls_idx, cls_name in enumerate(OUTCOME_NAMES):
        mask = outcomes == cls_idx
        if mask.sum() > 0:
            cls_matrix = tfidf_matrix[mask]
            centroid = np.asarray(cls_matrix.mean(axis=0)).flatten()
            centroids[cls_name] = centroid.tolist()

    return {
        "vectorizer": vectorizer,
        "matrix": tfidf_matrix,
        "vocab": vocab,
        "idf": idf,
        "centroids": centroids,
    }


# ── Model 2: Logistic Regression ──

def train_logistic_regression(
    X: np.ndarray, y: np.ndarray
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train Logistic Regression op gestructureerde features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    )
    model.fit(X_scaled, y)
    return model, scaler


# ── Model 3: Naive Bayes ──

def train_naive_bayes(
    tfidf_matrix, y: np.ndarray
) -> MultinomialNB:
    """Train Multinomial Naive Bayes op TF-IDF features."""
    model = MultinomialNB(alpha=0.5)
    model.fit(tfidf_matrix, y)
    return model


# ── Cross-validatie ──

def cross_validate_ensemble(
    X_struct: np.ndarray,
    texts: List[str],
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Stratified K-Fold cross-validatie voor het ensemble model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_preds = np.zeros(len(y), dtype=int)
    all_probs = np.zeros((len(y), 3))
    fold_scores = []

    # Scores per model voor gewichtsoptimalisatie
    model_fold_scores = {"logreg": [], "nb": [], "tfidf": []}

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_struct, y)):
        print(f"    Fold {fold_i + 1}/{n_splits}...")

        # Split
        X_train, X_test = X_struct[train_idx], X_struct[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]

        # Model 1: TF-IDF
        tfidf_data = train_tfidf_model(texts_train, y_train, max_features=800)
        tfidf_test = tfidf_data["vectorizer"].transform(texts_test)

        # TF-IDF predictions via cosine similarity met centroids
        tfidf_probs = np.zeros((len(test_idx), 3))
        for cls_idx, cls_name in enumerate(OUTCOME_NAMES):
            if cls_name in tfidf_data["centroids"]:
                centroid = np.array(tfidf_data["centroids"][cls_name])
                # Cosine similarity
                test_dense = np.asarray(tfidf_test.todense())
                norms_test = np.linalg.norm(test_dense, axis=1, keepdims=True)
                norms_test[norms_test == 0] = 1
                norm_centroid = np.linalg.norm(centroid)
                if norm_centroid > 0:
                    sim = (test_dense @ centroid) / (norms_test.flatten() * norm_centroid)
                    tfidf_probs[:, cls_idx] = np.maximum(0, sim)

        # Normaliseer tfidf_probs
        row_sums = tfidf_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        tfidf_probs /= row_sums

        # Model 2: Logistic Regression
        lr_model, lr_scaler = train_logistic_regression(X_train, y_train)
        X_test_scaled = lr_scaler.transform(X_test)
        lr_probs = lr_model.predict_proba(X_test_scaled)
        # Zorg dat alle klassen aanwezig zijn
        lr_probs_full = np.zeros((len(test_idx), 3))
        for j, cls in enumerate(lr_model.classes_):
            lr_probs_full[:, cls] = lr_probs[:, j]

        # Model 3: Naive Bayes
        nb_model = train_naive_bayes(tfidf_data["matrix"], y_train)
        nb_probs = nb_model.predict_proba(tfidf_test)
        nb_probs_full = np.zeros((len(test_idx), 3))
        for j, cls in enumerate(nb_model.classes_):
            nb_probs_full[:, cls] = nb_probs[:, j]

        # Per-model accuracy
        model_fold_scores["tfidf"].append(accuracy_score(y_test, tfidf_probs.argmax(axis=1)))
        model_fold_scores["logreg"].append(accuracy_score(y_test, lr_probs_full.argmax(axis=1)))
        model_fold_scores["nb"].append(accuracy_score(y_test, nb_probs_full.argmax(axis=1)))

        # Ensemble: gewogen gemiddelde
        ensemble_probs = 0.35 * lr_probs_full + 0.35 * nb_probs_full + 0.30 * tfidf_probs
        ensemble_preds = ensemble_probs.argmax(axis=1)

        all_preds[test_idx] = ensemble_preds
        all_probs[test_idx] = ensemble_probs

        fold_acc = accuracy_score(y_test, ensemble_preds)
        fold_scores.append(fold_acc)

    # Bereken optimale gewichten op basis van per-model prestaties
    avg_scores = {k: np.mean(v) for k, v in model_fold_scores.items()}
    total_score = sum(avg_scores.values())
    optimal_weights = {k: round(float(v / total_score), 3) for k, v in avg_scores.items()}

    # Finale metrics
    accuracy = accuracy_score(y, all_preds)
    f1_macro = f1_score(y, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y, all_preds, average="weighted", zero_division=0)
    precision = precision_score(y, all_preds, average="weighted", zero_division=0)
    recall = recall_score(y, all_preds, average="weighted", zero_division=0)

    # Per-klasse metrics
    per_class = {}
    for cls_idx, cls_name in enumerate(OUTCOME_NAMES):
        mask = y == cls_idx
        if mask.sum() > 0:
            cls_preds = all_preds[mask]
            cls_acc = (cls_preds == cls_idx).mean()
            per_class[cls_name] = {
                "n": int(mask.sum()),
                "accuracy": round(float(cls_acc), 3),
                "predicted": int((all_preds == cls_idx).sum()),
            }

    # Confusion matrix
    cm = confusion_matrix(y, all_preds, labels=[0, 1, 2])

    return {
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "fold_scores": [round(float(s), 4) for s in fold_scores],
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "model_scores": {k: round(float(v), 4) for k, v in avg_scores.items()},
        "optimal_weights": optimal_weights,
    }


# ── Export voor browser ──

def export_logistic_regression(
    model: LogisticRegression, scaler: StandardScaler, feature_names: List[str]
) -> Dict[str, Any]:
    """Exporteer logistic regression als JSON-serializable dict."""
    return {
        "weights": model.coef_.tolist(),
        "bias": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
    }


def export_naive_bayes(model: MultinomialNB) -> Dict[str, Any]:
    """Exporteer Naive Bayes als JSON-serializable dict."""
    return {
        "log_priors": model.class_log_prior_.tolist(),
        "feature_log_probs": model.feature_log_prob_.tolist(),
        "classes": model.classes_.tolist(),
    }


def export_tfidf(tfidf_data: Dict[str, Any]) -> Dict[str, Any]:
    """Exporteer TF-IDF model voor browser gebruik."""
    # Beperk vocabulary tot top terms per centroid
    vocab = tfidf_data["vocab"]
    idf = tfidf_data["idf"]

    # Exporteer alleen vocab/idf en centroids
    # Round centroids for size
    centroids_rounded = {}
    for cls_name, centroid in tfidf_data["centroids"].items():
        # Alleen niet-nul waarden opslaan (sparse)
        sparse = {}
        for i, val in enumerate(centroid):
            if abs(val) > 0.001:
                sparse[str(i)] = round(val, 4)
        centroids_rounded[cls_name] = sparse

    return {
        "vocabulary": {word: idx for idx, word in enumerate(vocab)},
        "idf": [round(v, 4) for v in idf],
        "centroids": centroids_rounded,
    }


# ── Feature importance ──

def compute_feature_importance(
    lr_model: LogisticRegression, feature_names: List[str]
) -> List[Dict[str, Any]]:
    """Bereken feature importance vanuit logistic regression gewichten."""
    # Gebruik absolute gewichten gemiddeld over klassen
    abs_weights = np.abs(lr_model.coef_).mean(axis=0)

    # Sorteer op belang
    indices = np.argsort(abs_weights)[::-1]
    importance = []
    for idx in indices[:30]:  # Top 30
        # Bepaal richting: positief gewicht voor 'toegewezen' klasse
        toegewezen_idx = list(lr_model.classes_).index(1) if 1 in lr_model.classes_ else 0
        direction = "pro_consument" if lr_model.coef_[toegewezen_idx, idx] > 0 else "pro_verzekeraar"

        importance.append({
            "feature": feature_names[idx],
            "importance": round(float(abs_weights[idx]), 4),
            "direction": direction,
        })

    return importance


# ── Hoofdfunctie ──

def train_ensemble(items: List[dict]) -> Dict[str, Any]:
    """Train het volledige ensemble model."""
    print(f"\n{'='*60}")
    print(f"ENSEMBLE MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Dataset: {len(items)} uitspraken\n")

    # Prep
    print("1. Feature engineering...")
    texts = [extract_text(u) for u in items]
    y = np.array([encode_outcome(u.get("uitkomst", "afgewezen")) for u in items])

    # Gestructureerde features
    feature_names = get_feature_names()
    X_struct = np.array([build_feature_vector(u, texts[i]) for i, u in enumerate(items)])
    print(f"   Gestructureerde features: {X_struct.shape[1]} dimensies")
    print(f"   Juridische termen: {len(JURIDISCHE_TERMEN)}")
    print(f"   Interactie features: 5")

    # Outcome verdeling
    for cls_idx, cls_name in enumerate(OUTCOME_NAMES):
        count = (y == cls_idx).sum()
        print(f"   {cls_name}: {count} ({count/len(y)*100:.1f}%)")

    # Cross-validatie
    print("\n2. Cross-validatie (5-fold)...")
    cv_results = cross_validate_ensemble(X_struct, texts, y, n_splits=5)
    print(f"\n   Ensemble resultaten:")
    print(f"   Accuracy:           {cv_results['accuracy']:.4f}")
    print(f"   F1 (macro):         {cv_results['f1_macro']:.4f}")
    print(f"   F1 (weighted):      {cv_results['f1_weighted']:.4f}")
    print(f"   Precision (w):      {cv_results['precision_weighted']:.4f}")
    print(f"   Recall (w):         {cv_results['recall_weighted']:.4f}")
    print(f"   Per-model scores:   {cv_results['model_scores']}")
    print(f"   Optimale gewichten: {cv_results['optimal_weights']}")
    print(f"\n   Per klasse:")
    for cls_name, cls_data in cv_results["per_class"].items():
        print(f"   {cls_name:15s}: accuracy={cls_data['accuracy']:.3f}, n={cls_data['n']}, predicted={cls_data['predicted']}")

    # Train finale modellen op volledige dataset
    print("\n3. Training finale modellen op volledige dataset...")

    # TF-IDF
    print("   TF-IDF model...")
    tfidf_data = train_tfidf_model(texts, y, max_features=800)
    print(f"   Vocabulary: {len(tfidf_data['vocab'])} termen")

    # Logistic Regression
    print("   Logistic Regression...")
    lr_model, lr_scaler = train_logistic_regression(X_struct, y)

    # Naive Bayes
    print("   Naive Bayes...")
    nb_model = train_naive_bayes(tfidf_data["matrix"], y)

    # Feature importance
    print("   Feature importance...")
    importance = compute_feature_importance(lr_model, feature_names)
    print(f"\n   Top 10 features:")
    for fi in importance[:10]:
        print(f"     {fi['feature']:35s} {fi['importance']:.4f} ({fi['direction']})")

    # Ensemble gewichten
    weights = cv_results["optimal_weights"]

    # Exporteer
    print("\n4. Exporteren...")
    exported = {
        "logreg": export_logistic_regression(lr_model, lr_scaler, feature_names),
        "naive_bayes": export_naive_bayes(nb_model),
        "tfidf": export_tfidf(tfidf_data),
        "ensemble_weights": weights,
        "feature_importance": importance,
        "cross_validation": cv_results,
        "juridische_termen": JURIDISCHE_TERMEN,
        "feature_config": {
            "insurance_types": INSURANCE_TYPES,
            "dispute_types": DISPUTE_TYPES,
            "evidence_levels": EVIDENCE_LEVELS,
            "expert_types": EXPERT_TYPES,
            "outcome_names": OUTCOME_NAMES,
        },
    }

    return exported


def main():
    parser = argparse.ArgumentParser(description="Train KIFID ensemble model")
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

    # Laad dataset
    print(f"Laden dataset: {dataset_path}")
    items = load_dataset(dataset_path)
    print(f"  {len(items)} uitspraken geladen")

    if not items:
        print("Geen uitspraken!")
        sys.exit(1)

    # Train ensemble
    ensemble_export = train_ensemble(items)

    # Laad bestaand statistisch model als basis
    existing_model = {}
    if output_path.exists():
        print(f"\nBestaand model laden: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            existing_model = json.load(f)

    # Merge: behoud statistisch model, voeg ensemble toe
    existing_model["meta"] = {
        "versie": "2.0",
        "getraind_op": date.today().isoformat(),
        "totaal_uitspraken": len(items),
        "focus": "alle",
        "model_type": "ensemble",
        "modellen": ["logistic_regression", "naive_bayes", "tfidf_similarity"],
    }
    existing_model["ensemble"] = ensemble_export

    # Opslaan
    print(f"\nOpslaan model: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing_model, f, ensure_ascii=False, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"  Model opgeslagen ({size_kb:.0f} KB)")

    # Samenvatting
    cv = ensemble_export["cross_validation"]
    print(f"\n{'='*60}")
    print(f"SAMENVATTING")
    print(f"{'='*60}")
    print(f"Dataset:              {len(items)} uitspraken")
    print(f"Features:             {len(ensemble_export['logreg']['feature_names'])} gestructureerd + {len(ensemble_export['tfidf']['vocabulary'])} tekst")
    print(f"Cross-val accuracy:   {cv['accuracy']:.1%}")
    print(f"Cross-val F1 (macro): {cv['f1_macro']:.1%}")
    print(f"Ensemble gewichten:   LR={ensemble_export['ensemble_weights']['logreg']}, NB={ensemble_export['ensemble_weights']['nb']}, TF-IDF={ensemble_export['ensemble_weights']['tfidf']}")
    print(f"Model grootte:        {size_kb:.0f} KB")
    print(f"\nKlaar!")


if __name__ == "__main__":
    main()
