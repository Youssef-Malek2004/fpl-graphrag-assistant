import argparse
from typing import Dict, Tuple, List, Any, Optional
import re

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

MODEL_CONFIG: Dict[str, Tuple[str, str]] = {
    "miniLM": ("sentence-transformers/all-MiniLM-L6-v2", "playerGameweek_miniLM_index"),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "playerGameweek_mpnet_index"),
}

# How many vector candidates to pull before filtering down to top_k
CANDIDATE_MULTIPLIER = 50


def load_config(path: str = "../neo4j/config.txt") -> Dict[str, str]:
    config: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            config[key] = value
    return config


def canonicalize_query(q: str) -> str:
    """
    Normalize natural language into the same key=value dialect used in embed_text_*.
    """
    s = q.strip()
    s = re.sub(r"\s+", " ", s)

    # gameweek
    s = re.sub(r"\bgameweek\s*(\d{1,2})\b", r"gw=\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\bgw\s*(\d{1,2})\b", r"gw=\1", s, flags=re.IGNORECASE)

    # season: "2021-22" or "2021/22"
    s = re.sub(r"\b(20\d{2}[-/]\d{2})\b", r"season=\1", s)

    # positions
    s = re.sub(r"\b(goalkeeper|gk)\b", "pos=GK", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(defender|def)\b", "pos=DEF", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(midfielder|mid)\b", "pos=MID", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(forward|fwd|striker)\b", "pos=FWD", s, flags=re.IGNORECASE)

    # numeric mentions
    s = re.sub(r"\b(\d{1,2})\s*(?:pts?|points?)\b", r"points=\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(\d{1,3})\s*(?:mins?|minutes?)\b", r"minutes=\1", s, flags=re.IGNORECASE)

    # allow "goals 3" "assists 2" as shorthand
    s = re.sub(r"\bgoals?\s*(\d+)\b", r"goals=\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\bassists?\s*(\d+)\b", r"assists=\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\bbonus\s*(\d+)\b", r"bonus=\1", s, flags=re.IGNORECASE)

    # also handle reversed order: "3 goals", "3 goal", "scored 3 goals"
    s = re.sub(r"\bscored\s*(\d+)\s*goals?\b", r"goals=\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(\d+)\s*goals?\b", r"goals=\1", s, flags=re.IGNORECASE)

    # same idea for assists/bonus/points/minutes (optional but recommended)
    s = re.sub(r"\b(\d+)\s*assists?\b", r"assists=\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(\d+)\s*bonus\b", r"bonus=\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(\d{1,2})\s*points?\b", r"points=\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(\d{1,3})\s*minutes?\b", r"minutes=\1", s, flags=re.IGNORECASE)

    return s


_OP_RE = re.compile(r"\b([a-z_]+)\s*(=|==|>=|<=|>|<)\s*(-?\d+)\b", re.IGNORECASE)


def extract_numeric_constraints(q: str) -> Dict[str, Tuple[str, int]]:
    """
    Parse constraints like:
      goals=3, assists>=2, points>10, minutes<=60, bonus=3
    Returns: { "goals": ("=", 3), ... }
    """
    out: Dict[str, Tuple[str, int]] = {}
    for m in _OP_RE.finditer(q):
        key = m.group(1).lower()
        op = m.group(2)
        val = int(m.group(3))
        # normalize == to =
        if op == "==":
            op = "="

        # map synonyms -> your relationship property names
        key_map = {
            "goals": "goals_scored",
            "goal": "goals_scored",
            "assists": "assists",
            "assist": "assists",
            "bonus": "bonus",
            "points": "total_points",
            "pts": "total_points",
            "minutes": "minutes",
            "mins": "minutes",
        }
        if key in key_map:
            out[key_map[key]] = (op, val)
    return out


def verify_index_exists(session, index_name: str) -> bool:
    rows = session.run(
        """
        SHOW INDEXES YIELD name, type, state
        WHERE type = 'VECTOR'
        RETURN name, state
        """
    ).data()

    for r in rows:
        if r["name"] == index_name:
            if r["state"] != "ONLINE":
                print(f"Warning: Index '{index_name}' exists but state is: {r['state']}")
                return False
            return True

    print(f"Error: Index '{index_name}' does not exist.")
    print("Run create_embeddings.py first.")
    return False


def _cypher_predicate(field: str, op: str) -> str:
    # safe, whitelisted numeric fields only
    if field not in {"goals_scored", "assists", "bonus", "total_points", "minutes"}:
        raise ValueError(f"Unsupported filter field: {field}")
    if op not in {"=", ">", "<", ">=", "<="}:
        raise ValueError(f"Unsupported operator: {op}")
    # handle nulls by requiring field IS NOT NULL
    return f"pi.{field} IS NOT NULL AND pi.{field} {op} ${field}"

def filtered_similarity_search(
    session,
    model_name: str,
    query_embedding: List[float],
    top_k: int,
    numeric_filters: Dict[str, Tuple[str, int]],
) -> List[Dict[str, Any]]:
    """
    When hard filters exist, query by filters FIRST, then rank by cosine similarity
    against the stored relationship embedding property.

    This avoids the 'vector retrieval misses filtered rows' failure mode.
    """
    embedding_property = f"embedding_{model_name}"

    where_clauses = ["pi.minutes IS NOT NULL AND pi.minutes > 0", f"pi.{embedding_property} IS NOT NULL"]
    params: Dict[str, Any] = {"top_k": top_k, "query_embedding": query_embedding}

    for field, (op, val) in numeric_filters.items():
        where_clauses.append(_cypher_predicate(field, op))
        params[field] = val

    where_sql = " AND ".join(where_clauses)

    query = f"""
    MATCH (p:Player)-[pi:PLAYED_IN]->(f:Fixture)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
    MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    MATCH (p)-[:PLAYS_AS]->(pos:Position)
    WHERE {where_sql}
    WITH p, pi, f, gw, s, home, away, pos,
         vector.similarity.cosine(pi.{embedding_property}, $query_embedding) AS score
    RETURN
        p.player_name as player_name,
        pos.name as position,
        pi.minutes as minutes,
        pi.total_points as total_points,
        pi.goals_scored as goals_scored,
        pi.assists as assists,
        pi.bonus as bonus,
        home.name as home_team,
        away.name as away_team,
        s.season_name as season,
        gw.GW_number as gw_number,
        score
    ORDER BY score DESC
    LIMIT $top_k
    """
    return [dict(r) for r in session.run(query, **params)]



def vector_search_relationships(
    session,
    index_name: str,
    query_embedding: List[float],
    top_k: int,
    numeric_filters: Dict[str, Tuple[str, int]],
    candidate_k: int,
) -> List[Dict[str, Any]]:
    where_clauses = ["pi.minutes IS NOT NULL AND pi.minutes > 0"]
    params: Dict[str, Any] = {
        "index_name": index_name,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "query_embedding": query_embedding,
    }

    for field, (op, val) in numeric_filters.items():
        where_clauses.append(_cypher_predicate(field, op))
        params[field] = val

    where_sql = " AND ".join(where_clauses)

    query = f"""
    CALL db.index.vector.queryRelationships($index_name, $candidate_k, $query_embedding)
    YIELD relationship, score
    WITH relationship AS pi, score
    MATCH (p:Player)-[pi:PLAYED_IN]->(f:Fixture)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
    MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    MATCH (p)-[:PLAYS_AS]->(pos:Position)
    WHERE {where_sql}
    RETURN
        p.player_name as player_name,
        pos.name as position,
        pi.minutes as minutes,
        pi.total_points as total_points,
        pi.goals_scored as goals_scored,
        pi.assists as assists,
        pi.bonus as bonus,
        home.name as home_team,
        away.name as away_team,
        s.season_name as season,
        gw.GW_number as gw_number,
        score
    ORDER BY score DESC
    LIMIT $top_k
    """
    return [dict(r) for r in session.run(query, **params)]


def display_results(results: List[Dict[str, Any]], header: str) -> None:
    print("\n" + "=" * 90)
    print(header)
    print("=" * 90)

    if not results:
        print("No results.")
        return

    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['player_name']} ({r['position']})  score={r['score']:.4f}")
        print(f"    {r['season']} GW{r['gw_number']} — {r['home_team']} vs {r['away_team']}")
        print(
            f"    mins={r.get('minutes')} pts={r.get('total_points')} "
            f"goals={r.get('goals_scored', 0)} assists={r.get('assists', 0)} bonus={r.get('bonus', 0)}"
        )


def run_query(driver, model_name: str, user_query: str, top_k: int) -> None:
    model_path, index_name = MODEL_CONFIG[model_name]

    with driver.session() as session:
        if not verify_index_exists(session, index_name):
            return

    canon = canonicalize_query(user_query)
    numeric_filters = extract_numeric_constraints(canon)

    print(f"\nModel: {model_name} | Index: {index_name}")
    print(f"Raw query: {user_query}")
    print(f"Canonical query: {canon}")
    if numeric_filters:
        print(f"Hard filters: {numeric_filters}")

    model = SentenceTransformer(model_path)

    # IMPORTANT: embed the canonical query, not the raw one
    q_emb = model.encode(canon, normalize_embeddings=True).tolist()

    candidate_k = max(top_k * CANDIDATE_MULTIPLIER, top_k)

    with driver.session() as session:
        if numeric_filters:
            results = filtered_similarity_search(
                session=session,
                model_name=model_name,
                query_embedding=q_emb,
                top_k=top_k,
                numeric_filters=numeric_filters,
            )
        else:
            results = vector_search_relationships(
                session=session,
                index_name=index_name,
                query_embedding=q_emb,
                top_k=top_k,
                numeric_filters=numeric_filters,
                candidate_k=candidate_k,
            )

    display_results(results, header=f"Top {len(results)} results (VECTOR + FILTERS) — model={model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid semantic search over PLAYED_IN via vector index + hard numeric filters.")
    parser.add_argument("query", type=str, nargs="+", help="Any raw query string, e.g. 'goals=3' or 'season 2021-22 gw 15 points>=10'")
    parser.add_argument("--model", choices=["miniLM", "mpnet", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    user_query = " ".join(args.query)

    cfg = load_config()
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))

    try:
        if args.model == "both":
            for m in MODEL_CONFIG.keys():
                run_query(driver, m, user_query, args.top_k)
        else:
            run_query(driver, args.model, user_query, args.top_k)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
