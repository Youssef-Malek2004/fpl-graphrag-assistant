# src/intent_classification.py

import re
from typing import Dict, Any
from nltk.stem import PorterStemmer

# ==============================
# TEXT CLEANING
# ==============================

stemmer = PorterStemmer()

FPL_SYNONYMS = {
    r"\br.?com+.*\b": "recommend",
    r"\brecom+\w*\b": "recommend",
    r"\brecmd\b": "recommend",
    r"\brecomend\b": "recommend",
    r"\brecomd\b": "recommend",
    r"\bwho to pick\b": "recommend",
    r"\bpick\b": "recommend",
    r"\bor\b": "vs",
    r"\bversus\b": "vs",
    r"\bcompare\b": "vs",
}

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s:/\-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    for pattern, replacement in FPL_SYNONYMS.items():
        text = re.sub(pattern, replacement, text)

    return " ".join(stemmer.stem(w) for w in text.split())

# ==============================
# INTENT CONSTANTS
# ==============================

INTENT_PLAYER_RECOMMENDATION = "RECOMMENDATION"
INTENT_PLAYER_PERFORMANCE = "PLAYER_PERFORMANCE_ANALYSIS"
INTENT_PLAYER_COMPARISON = "PLAYER_COMPARISON"
INTENT_TEAM_ANALYSIS = "TEAM_ANALYSIS"
INTENT_AGGREGATION = "AGGREGATION"

COMPARISON_KEYWORDS = [" vs ", "versus", "compare", "better than", "outperform"]

RECOMMENDATION_KEYWORDS = [
    "recommend", "suggest", "who should i", "who to pick",
    "best", "cheap", "budget", "pick for", "buy", "captain",
    "optimal" , "build"
]

# ==============================
# CLASSIFIER
# ==============================

def classify_intent(cleaned_text: str, entities: Dict[str, Any]) -> str:

    t = cleaned_text
    players = entities.get("players", [])
    teams = entities.get("teams", [])
    statistics = entities.get("statistics", [])
    aggregate = entities.get("aggregate")

    # ---------------------------------------------------------
    # 1️⃣ PLAYER COMPARISON
    # ---------------------------------------------------------
    if any(kw in t for kw in COMPARISON_KEYWORDS) or len(players) >= 2:
        return INTENT_PLAYER_COMPARISON

    # ---------------------------------------------------------
    # 2️⃣ PLAYER RECOMMENDATION
    # ---------------------------------------------------------
    if any(kw in t for kw in RECOMMENDATION_KEYWORDS):
        return INTENT_PLAYER_RECOMMENDATION

    if aggregate is not None:
        # If a team is mentioned, you probably want TEAM_ANALYSIS style aggregates later
        if not teams:
            return INTENT_AGGREGATION

    # ---------------------------------------------------------
    # 3️⃣ TEAM ANALYSIS
    # ---------------------------------------------------------
    if (teams and not players) or (
        not players and aggregate is not None and
        any(stat in statistics for stat in [
            "goals_scored",
            "goals_conceded",
            "clean_sheets",
            "yellow_cards",
            "red_cards",
            "assists",
            "saves",
            "total_points"
        ])
    ):
        return INTENT_TEAM_ANALYSIS

    # ---------------------------------------------------------
    # 4️⃣ PLAYER PERFORMANCE ANALYSIS (DEFAULT)
    # ---------------------------------------------------------
    return INTENT_PLAYER_PERFORMANCE



def should_generate_embedding(intent: str) -> bool:
    return intent in {
        INTENT_PLAYER_RECOMMENDATION,
        INTENT_PLAYER_COMPARISON
        # ,PLAYER_PERFORMANCE_ANALYSIS, TEAM_ANALYSIS    (uncomment if u want to generate imbeddings for all types of queries)
    }


