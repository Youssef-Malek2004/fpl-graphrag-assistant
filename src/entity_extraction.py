# src/entity_extraction.py


import re
import spacy
import pandas as pd
from spacy.pipeline import EntityRuler
from typing import List, Set, Dict, Optional, Any
import os
csv_path = None
possible_paths = [
    "../../neo4j/fpl_two_seasons.csv",
    "../neo4j/fpl_two_seasons.csv",
    "neo4j/fpl_two_seasons.csv",
    os.path.join(os.path.dirname(__file__), "..", "..", "neo4j", "fpl_two_seasons.csv"),
]
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    raise FileNotFoundError("Cannot find fpl_two_seasons.csv. Please check the file location.")

# ============================================================
# Load spaCy model (same as notebook)
# ============================================================

nlp = spacy.load("en_core_web_sm")


# ============================================================
# Utility: Reset EntityRuler
# ============================================================

def _reset_entity_ruler() -> EntityRuler:
    """Remove existing EntityRuler (if any) and add a fresh one."""
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    return nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})


# ============================================================
# AGGREGATE & LIMIT
# ============================================================

AGGREGATE_FUNCTION_KEYWORDS: Dict[str, List[str]] = {
    "MAX": [ "most", "highest", "best", "maximum", "max", "leading", "top scorer", "top points", "top ranked" ],
    # "MAX": [ "most", "highest", "best", "top", "maximum", "max", "leading", "top scorer", "top points", "top ranked" ],
    "MIN": [ "least", "lowest", "worst", "minimum", "min", "cheapest"],
    "SUM": [ "total", "overall", "sum", "combined" ],
    "AVG": [ "average", "avg", "mean", "typical", "consistent"],
    "COUNT": [ "how many", "number of", "count", "quantity"]
}

NUMBER_WORDS: Dict[str, int] = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}



def extract_aggregate_function(text: str) -> Optional[str]:
    text_l = text.lower()

    for agg_func, keywords in AGGREGATE_FUNCTION_KEYWORDS.items():
        for kw in keywords:

            # Exact whole-word match using regex:
            # \b ensures "min" does NOT match "minutes"
            if re.search(rf"\b{re.escape(kw)}\b", text_l):
                return agg_func

    return None





# ------------------ Extract Limit -------------------------------

def extract_limit(text: str) -> Optional[int]:
    text_l = text.lower()

    # 1) Direct numeric: "top 5", "best10", "top-3" ...
    match = re.search(r"(top|best|highest|lowest|worst|bottom)[\s\-]?(\d+)", text_l)
    if match:
        return int(match.group(2))

    # 2) Word numbers: "top five", "best three", etc.
    for word, num in NUMBER_WORDS.items():
        if re.search(rf"(top|best|highest|lowest|worst|bottom)\s+{word}", text_l):
            return num

    # 3) NEW — direct limit: "5 players", "10 midfielders", etc.
    match = re.search(r"\b(\d+)\s+(players?|defenders?|midfielders?|forwards?|gks?)\b", text_l)
    if match:
        return int(match.group(1))

    # 4) NEW — word-number version: "five players", "three forwards"
    for word, num in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+(players?|defenders?|midfielders?|forwards?|gks?)\b", text_l):
            return num

    return None



# ============================================================
# PLAYERS
# ============================================================



def extract_players(text: str, vocab_players: Set[str]) -> List[str]:
    """
    Extract player names using strict matching rules.

    Rules:
      - Single word: Return only if unique match (1 player), else []
      - Two words: First must match first name, second a remaining part
      - Full name: Direct match
      - Complex text: scan for two-word combos, then singles, then full-name NER
    """
    # Build lookup maps from provided vocab_players
    # (Expected to be a set/list of full names)
    vocab_players = set(vocab_players)
    full_names: Dict[str, str] = {name.lower(): name for name in vocab_players}
    first_map: Dict[str, List[str]] = {}
    last_map: Dict[str, List[str]] = {}

    for name in vocab_players:
        parts = name.lower().split()
        if len(parts) >= 2:
            first_map.setdefault(parts[0], []).append(name)
            last_map.setdefault(parts[-1], []).append(name)

    # EntityRuler: full names + all first/last tokens
    ruler = _reset_entity_ruler()
    patterns = [{"label": "PLAYER_FULL", "pattern": n.lower()} for n in vocab_players]
    part_tokens = set(first_map.keys()) | set(last_map.keys())
    patterns += [{"label": "PLAYER_PART", "pattern": tok} for tok in part_tokens]
    ruler.add_patterns(patterns)

    def match_single(word: str) -> List[str]:
        """Return players if there's a unique match (first or last name)."""
        w = word.lower()
        matches = set(first_map.get(w, [])) | set(last_map.get(w, []))
        return list(matches) if len(matches) == 1 else []

    def match_two(first: str, second: str) -> List[str]:
        """
        Two-token match: first must be a first name; second must match
        one of the remaining parts of the full name.
        """
        f, s = first.lower(), second.lower()
        if f not in first_map:
            return []
        return sorted([
            n for n in first_map[f]
            if s in n.lower().split()[1:]
        ])

    # ----------------------------
    # Start parsing the input text
    # ----------------------------
    raw_text = text.strip()
    text_l = raw_text.lower()
    tokens = re.findall(r'\b\w+\b', text_l)

    # Direct full-name match
    if text_l in full_names:
        return [full_names[text_l]]

    # 1-2 tokens special handling
    if len(tokens) == 1:
        return match_single(tokens[0])
    if len(tokens) == 2:
        return match_two(tokens[0], tokens[1]) or match_single(tokens[0])

    # Longer text: use spaCy tokenization & passes
    doc = nlp(text_l)
    words = [t.text for t in doc if t.is_alpha]

    detected: Set[str] = set()
    used_indices: Set[int] = set()

    # Pass 1: two-word combinations
    for i in range(len(words) - 1):
        if i in used_indices:
            continue
        matches = match_two(words[i], words[i + 1])
        if matches:
            detected.update(matches)
            used_indices.update({i, i + 1})

    # Pass 2: unique singles
    for i, word in enumerate(words):
        if i in used_indices:
            continue
        detected.update(match_single(word))

    # Pass 3: full names from NER
    for ent in doc.ents:
        if ent.label_ == "PLAYER_FULL":
            full = ent.text.lower()
            if full in full_names:
                detected.add(full_names[full])

    return sorted(detected)


# ============================================================
# TEAMS
# ============================================================



TEAM_ALIASES_BASE: Dict[str, List[str]] = {
    "Arsenal": ["arsenal", "ars", "afc", "gunners"],
    "Aston Villa": ["aston villa", "villa", "avfc"],
    "Bournemouth": ["bournemouth", "afcb", "cherries"],
    "Brentford": ["brentford", "bfc"],
    "Brighton": ["brighton", "brighton & hove", "bhafc", "seagulls"],
    "Burnley": ["burnley", "burnley fc", "clarets"],
    "Chelsea": ["chelsea", "cfc", "the blues"],
    "Crystal Palace": ["crystal palace", "palace", "cpfc", "eagles"],
    "Everton": ["everton", "efc", "toffees"],
    "Fulham": ["fulham", "ffc", "the whites"],
    "Leeds": ["leeds", "leeds united", "lufc"],
    "Leicester": ["leicester", "leicester city", "lcfc", "foxes"],
    "Liverpool": ["liverpool", "lfc", "reds"],
    "Man City": ["man city", "manchester city", "city", "mcfc"],
    "Man Utd": ["man utd", "manchester united", "man united", "united", "mufc"],
    "Newcastle": ["newcastle", "newcastle united", "nufc", "magpies"],
    "Norwich": ["norwich", "norwich city", "ncfc"],
    "Nott'm Forest": ["nottm forest", "nottingham forest", "forest", "nffc"],
    "Southampton": ["southampton", "saints", "sfc"],
    "Spurs": ["spurs", "tottenham", "tottenham hotspur", "thfc"],
    "Watford": ["watford", "watford fc", "hornets", "wfc"],
    "West Ham": ["west ham", "west ham united", "whu", "whufc", "hammers"],
    "Wolves": ["wolves", "wolverhampton", "wolverhampton wanderers", "wwfc"],
}


def extract_teams(text: str, vocab_teams) -> List[str]:

    # Accept either a DataFrame or a set/list
    if hasattr(vocab_teams, "columns"):
        teams_set = (
            set(vocab_teams["home_team"].dropna().unique()) |
            set(vocab_teams["away_team"].dropna().unique())
        )
    else:
        teams_set = set(vocab_teams)

    # Keep only aliases for teams actually present in dataset
    team_aliases = {
        team: aliases
        for team, aliases in TEAM_ALIASES_BASE.items()
        if team in teams_set
    }

    alias_map: Dict[str, str] = {}
    for team, aliases in team_aliases.items():
        for alias in aliases:
            alias_map[alias] = team

    ruler = _reset_entity_ruler()
    ruler.add_patterns([
        {"label": "TEAM", "pattern": alias}
        for alias in alias_map.keys()
    ])

    doc = nlp(text.lower())
    detected = set()

    for ent in doc.ents:
        if ent.label_ == "TEAM":
            alias = ent.text.lower()
            if alias in alias_map:
                detected.add(alias_map[alias])

    return sorted(detected)



# ============================================================
# POSITIONS
# ============================================================



POSITION_KEYWORDS_BASE: Dict[str, List[str]] = {
    "GK":  ["goalkeeper", "keeper", "gk", "goalie", "goalkeepers"],
    "DEF": ["defender", "defence", "defense", "def", "centre back", "center back", "fullback", "defenders", "defensive"],
    "MID": ["midfielder", "midfield", "mid", "winger", "attacking mid", "defensive mid", "midfielders"],
    "FWD": ["forward", "forwards", "fwd", "striker", "attacker", "attacking"],
}


def extract_positions(text: str, df) -> List[str]:

    dataset_positions = set(df['position'].dropna().unique())

    # Keep only positions that actually exist in dataset
    position_keywords = {
        pos: keywords
        for pos, keywords in POSITION_KEYWORDS_BASE.items()
        if pos in dataset_positions
    }

    ruler = _reset_entity_ruler()
    patterns = []
    for pos_code, keywords in position_keywords.items():
        for keyword in keywords:
            patterns.append({"label": "POSITION", "pattern": keyword.lower()})
    ruler.add_patterns(patterns)

    doc = nlp(text.lower())
    detected_positions = set()

    for ent in doc.ents:
        if ent.label_ == "POSITION":
            for pos_code, keywords in position_keywords.items():
                if ent.text in keywords:
                    detected_positions.add(pos_code)
                    break

    return sorted(detected_positions)




# ============================================================
# GAMEWEEK
# ============================================================



# Prebuild GAMEWEEK patterns once (GW1–GW38)
GAMEWEEK_PATTERNS: List[Dict] = []
for gw_num in range(1, 39):
    GAMEWEEK_PATTERNS.extend([
        {"label": "GAMEWEEK", "pattern": [{"LOWER": "gw"}, {"TEXT": {"REGEX": f"[- ]?{gw_num}"}}]},
        {"label": "GAMEWEEK", "pattern": [{"LOWER": f"gw{gw_num}"}]},
        {"label": "GAMEWEEK", "pattern": [{"LOWER": "gameweek"}, {"LOWER": str(gw_num)}]},
        {"label": "GAMEWEEK", "pattern": [{"LOWER": "game"}, {"LOWER": "week"}, {"LOWER": str(gw_num)}]},
    ])


def extract_gameweek(text: str) -> Optional[int]:
    
    # Extract a gameweek number (1–38) from text. Uses NER with EntityRuler + regex fallback.

    ruler = _reset_entity_ruler()
    ruler.add_patterns(GAMEWEEK_PATTERNS)

    doc = nlp(text.lower())

    # NER detection
    for ent in doc.ents:
        if ent.label_ == "GAMEWEEK":
            nums = re.findall(r"\d+", ent.text)
            if nums:
                gw = int(nums[0])
                if 1 <= gw <= 38:
                    return gw

    # Regex fallback
    fallback = re.search(r"(gw|game[\s-]?week)\s*[- ]?([0-9]{1,2})", text, re.IGNORECASE)
    if fallback:
        gw = int(fallback.group(2))
        if 1 <= gw <= 38:
            return gw

    return None


# ============================================================
# SEASON
# ============================================================

def extract_season(text: str, dataset_seasons) -> Optional[str]:
    seasons_set = set(dataset_seasons)

    # Build normalized mapping
    normalized_seasons: Dict[str, str] = {}
    for season in seasons_set:
        s = season.lower()
        normalized_seasons[s] = season

        if "-" in season or "/" in season:
            parts = re.split(r"[-/]", season)
            if len(parts) == 2:
                start, end = parts
                normalized_seasons[f"{start}-{end}"] = season
                normalized_seasons[f"{start}/{end}"] = season
                # Short years
                if len(start) == 4:
                    ss = start[-2:]
                    normalized_seasons[f"{ss}-{end}"] = season
                    normalized_seasons[f"{ss}/{end}"] = season

    # Setup spaCy EntityRuler
    ruler = _reset_entity_ruler()
    patterns = [{"label": "SEASON", "pattern": key} for key in normalized_seasons.keys()]
    ruler.add_patterns(patterns)

    doc = nlp(text.lower())

    # NER detection
    for ent in doc.ents:
        if ent.label_ == "SEASON":
            key = ent.text.lower().strip()
            if key in normalized_seasons:
                return normalized_seasons[key]

    # Fallback regex 1: Full form (2021-22)
    match = re.search(r"(20[0-9]{2})\s*[-/]\s*([0-9]{2})", text)
    if match:
        start, end = match.groups()
        for season in seasons_set:
            if start in season and end in season:
                return season

    # Fallback regex 2: Short form (22/23)
    match = re.search(r"\b([0-9]{2})\s*[-/]\s*([0-9]{2})\b", text)
    if match:
        start2, end2 = match.groups()
        for season in seasons_set:
            if start2 in season and end2 in season:
                return season

    return None


# ============================================================
# STATISTICS
# ============================================================


STATISTIC_KEYWORDS: Dict[str, List[str]] = {
    "goals_scored": ["goal", "goals", "scored", "netted", "goalscorers"],
    "goals_conceded": ["conceded", "goals conceded"],
    "assists": ["assist", "assists"],
    "clean_sheets": ["clean sheet", "clean sheets", "cs"],
    "bonus": ["bonus", "bonus points"],
    "bps": ["bps"],
    "yellow_cards": ["yellow card", "yellow cards", "yellows", "yellow"],
    "red_cards": ["red card", "red cards", "reds", "red"],
    "penalties_saved": ["penalty save", "penalties saved"],
    "penalties_missed": ["penalty missed", "penalties missed"],
    "saves": ["save", "saves"],
    "own_goals": ["own goal", "own goals"],
    "minutes": ["minutes", "mins"],
    "influence": ["influence"],
    "creativity": ["creativity"],
    "threat": ["threat"],
    "ict_index": ["ict", "ict index"],
    "total_points": ["points", "pts", "total points"],
    "value": ["price", "value", "cost"],
    "selected": ["selected by", "ownership"],
    "transfers_in": ["transfers in", "brought in"],
    "transfers_out": ["transfers out", "sold"],
    "form": ["form"],
}


def extract_statistics(text: str) -> Optional[str]:
    ruler = _reset_entity_ruler()

    patterns = []
    for stat_key, keywords in STATISTIC_KEYWORDS.items():
        for kw in keywords:
            patterns.append({"label": "STATISTIC", "pattern": kw.lower()})
    ruler.add_patterns(patterns)

    doc = nlp(text.lower())
    detected: Set[str] = set()

    for ent in doc.ents:
        if ent.label_ == "STATISTIC":
            matched_kw = ent.text.lower()
            for stat_key, keywords in STATISTIC_KEYWORDS.items():
                if matched_kw in keywords:
                    detected.add(stat_key)

    return sorted(detected)[0] if detected else sorted(detected)


# ============================================================
# FINAL WRAPPER — EXTRACT ALL 8 ENTITIES
# ============================================================

def extract_entities(query: str, vocab: Dict[str, Set[str]]) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    return {
        "players": extract_players(query, vocab["players"]),
        "teams": extract_teams(query, vocab["teams"]),
        "positions": extract_positions(query, df),
        "gameweek": extract_gameweek(query),
        "season": extract_season(query, vocab["seasons"]),
        "metric": extract_statistics(query),
        "aggregate": extract_aggregate_function(query),
        "limit": extract_limit(query),
    }
