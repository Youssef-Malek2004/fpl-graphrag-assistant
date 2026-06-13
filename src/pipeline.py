# src/pipeline.py

from typing import Dict, Set, Any
import pandas as pd

# ---- imports from your other modules ----
from src.entity_extraction import extract_entities
from src.intent_classification import clean_text, classify_intent, should_generate_embedding
from src.input_embedding import generate_embeddings


# ============================================================
# 1) VOCAB BUILDING FROM CSV
# ============================================================

def load_fpl_vocab(csv_path: str) -> Dict[str, Set[str]]:
    df = pd.read_csv(csv_path)

    # Basic cleaning / cast (EXACT same logic)
    df["season"] = df["season"].astype(str)
    df["GW"] = df["GW"].astype(int)
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)
    df["name"] = df["name"].astype(str)
    df["position"] = df["position"].astype(str)

    players = set(df["name"].unique().tolist())
    teams = set(df["home_team"].unique().tolist()) | set(df["away_team"].unique().tolist())
    positions = set(df["position"].unique().tolist())
    seasons = set(df["season"].unique().tolist())
    gameweeks = set(df["GW"].unique().tolist())

    vocab = {
        "players": players,
        "teams": teams,
        "positions": positions,
        "seasons": seasons,
        "gameweeks": gameweeks,
    }
    return vocab


# ============================================================
# 6) MAIN WRAPPER: PROCESS ONE USER QUERY
# ============================================================

def process_input(query: str, vocab: Dict[str, Set[str]]) -> Dict[str, Any]:

    # 1) Extract entities
    entities = extract_entities(query, vocab)

    # 2) Classify intent
    intent = classify_intent(clean_text(query), entities)

    # 3) Generate embeddings ONLY if needed
    embedding = None
    if should_generate_embedding(intent):
        embedding = generate_embeddings(clean_text(query))

    # 4) Final structured output (EXACT same structure)
    result = {
        "raw_query": query,
        "cleaned_text": clean_text(query),
        "intent": intent,
        "entities": entities,
        "embedding": embedding,
    }

    return result


from pprint import pprint

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load vocabulary 
    # ------------------------------------------------------------
    CSV_PATH = "../neo4j/fpl_two_seasons.csv"
    vocab = load_fpl_vocab(CSV_PATH)

    # ------------------------------------------------------------
    # TEST QUERIES
    # ------------------------------------------------------------
    TEST_QUERIES = [

        # ============================================================
        # 1) PLAYER_PERFORMANCE_ANALYSIS
        # ============================================================
        ("PLAYER_PERFORMANCE_ANALYSIS", "How many assists did Bukayo Saka make in gameweek 12?"),
        ("PLAYER_PERFORMANCE_ANALYSIS", "Show me the form and threat of Mohamed Salah this season."),
        ("PLAYER_PERFORMANCE_ANALYSIS", "What were Erling Haaland's minutes and BPS in the 2023/24 season?"),
        ("PLAYER_PERFORMANCE_ANALYSIS", "Display the creativity and influence stats for Mohamed Salah."),
        ("PLAYER_PERFORMANCE_ANALYSIS", "How many saves and clean sheets did Alisson record in the 2022/23 season?"),

        # ============================================================
        # 2) TEAM_ANALYSIS
        # ============================================================
        ("TEAM_ANALYSIS", "How did Manchester United perform in gameweek 5?"),
        ("TEAM_ANALYSIS", "Show the total goals scored by Arsenal across the 2022/23 season."),
        ("TEAM_ANALYSIS", "Which team conceded the most goals last season?"),
        ("TEAM_ANALYSIS", "Analyze Liverpool's defensive performance in the 2021/22 season."),
        ("TEAM_ANALYSIS", "What was Chelsea’s overall form during this period?"),

        # ============================================================
        # 3) PLAYER_RECOMMENDATION
        # ============================================================
        ("PLAYER_RECOMMENDATION", "Recommend three midfielders with strong form this week."),
        ("PLAYER_RECOMMENDATION", "Which defenders should I pick for double gameweek 20?"),
        ("PLAYER_RECOMMENDATION", "Suggest a cheap forward under value 6.0."),
        ("PLAYER_RECOMMENDATION", "Who are the best penalty takers to buy right now?"),
        ("PLAYER_RECOMMENDATION", "Recommend two players from Manchester City for the next fixture."),

        # ============================================================
        # 4) PLAYER_COMPARISON
        # ============================================================
        ("PLAYER_COMPARISON", "Is Martin Ødegaard performing better than James Maddison this season?"),
        ("PLAYER_COMPARISON", "Compare Luis Díaz and Darwin Núñez in terms of goal scoring."),
        ("PLAYER_COMPARISON", "Who is better in midfield, Bruno Fernandes or Bernardo Silva?"),
        ("PLAYER_COMPARISON", "Compare the creativity of Bukayo Saka vs Dejan Kulusevski."),
        ("PLAYER_COMPARISON", "Is Phil Foden outperforming Jack Grealish in the 2023/24 season?"),
        ("PLAYER_COMPARISON", "Haaland or Kane?"),
        ("PLAYER_COMPARISON", "Salah or Son for this season?"),
        ("PLAYER_COMPARISON", "Trent Alexander-Arnold or Reece James?"),
        ("PLAYER_COMPARISON", "Rashford or Saka this gameweek?"),
    ]

    print("\n\n==================== INPUT PREPROCESSING — TEST RESULTS ====================\n")

    # ------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------
    for expected_category, query in TEST_QUERIES:
        out = process_input(query, vocab)

        print("=" * 80)
        print(f" CATEGORY (Expected): {expected_category}")
        print(f" Query: {query}")
        print(f" Intent → {out['intent']}\n")

        print(" Extracted Entities:")
        print(f"    Players     → {out['entities']['players']}")
        print(f"    Teams       → {out['entities']['teams']}")
        print(f"    Positions   → {out['entities']['positions']}")
        print(f"    Gameweek    → {out['entities']['gameweek']}")
        print(f"    Season      → {out['entities']['season']}")
        print(f"    Statistics  → {out['entities']['statistics']}")
        print(f"    Aggregate   → {out['entities']['aggregate']}")
        print(f"    Limit       → {out['entities']['limit']}\n")

        if out["embedding"] is not None:
            print(" Embeddings generated ✔")
            for model_name, vector in out["embedding"].items():
                print(f"    {model_name}: vector size = {len(vector)}")
        else:
            print(" Embeddings skipped (not needed for this intent)")

        print("=" * 80 + "\n")
