import time
from typing import Dict, Tuple, List, Any

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_CONFIG: Dict[str, Tuple[str, str]] = {
    "miniLM": ("sentence-transformers/all-MiniLM-L6-v2", "playerGameweek_miniLM_index"),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "playerGameweek_mpnet_index"),
}

BATCH_SIZE = 200
ENCODE_BATCH_SIZE = 64


def load_config(path: str = "../neo4j/config.txt") -> Dict[str, str]:
    config: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            config[key] = value
    return config


def bucket_minutes(minutes: Any) -> str:
    if minutes is None:
        return "unknown"
    try:
        m = int(minutes)
    except Exception:
        return "unknown"

    if m == 0:
        return "0"
    if m < 30:
        return "1-29"
    if m < 60:
        return "30-59"
    if m < 80:
        return "60-79"
    return "80-90"


def bucket_points(points: Any) -> str:
    if points is None:
        return "unknown"
    try:
        p = int(points)
    except Exception:
        return "unknown"

    if p <= 0:
        return "<=0"
    if p <= 2:
        return "1-2"
    if p <= 5:
        return "3-5"
    if p <= 8:
        return "6-8"
    if p <= 12:
        return "9-12"
    return "13+"


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def create_sparse_text(r: Dict[str, Any]) -> str:
    """
    Query-friendly text: compact, tokenized, low-boilerplate.
    This is what we embed and index, so raw user queries can match better.
    """
    minutes = r.get("minutes")
    total_points = r.get("total_points")

    goals = safe_int(r.get("goals_scored"), 0)
    assists = safe_int(r.get("assists"), 0)
    bonus = safe_int(r.get("bonus"), 0)

    # Important: keep schema stable (key=value tokens)
    return (
        f"player={r.get('player_name','unknown')} "
        f"pos={r.get('position','unknown')} "
        f"season={r.get('season','unknown')} "
        f"gw={r.get('gw_number','unknown')} "
        f"home={r.get('home_team','unknown')} "
        f"away={r.get('away_team','unknown')} "
        f"minutes={minutes if minutes is not None else 'unknown'} "
        f"minutes_bucket={bucket_minutes(minutes)} "
        f"points={total_points if total_points is not None else 'unknown'} "
        f"points_bucket={bucket_points(total_points)} "
        f"goals={goals} assists={assists} bonus={bonus}"
    )


def fetch_player_gameweek_data(session) -> List[Dict[str, Any]]:
    query = """
    MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    MATCH (p:Player)-[pi:PLAYED_IN]->(f)
    MATCH (p)-[:PLAYS_AS]->(pos:Position)
    RETURN
        s.season_name as season,
        gw.GW_number as gw_number,
        f.fixture_number as fixture_number,
        p.player_name as player_name,
        p.player_element as player_element,
        pos.name as position,
        home.name as home_team,
        away.name as away_team,
        pi.minutes as minutes,
        pi.total_points as total_points,
        pi.goals_scored as goals_scored,
        pi.assists as assists,
        pi.bonus as bonus,
        id(p) as player_id,
        id(f) as fixture_id
    ORDER BY s.season_name, gw.GW_number, f.fixture_number
    """
    return [dict(r) for r in session.run(query)]


def create_vector_index(session, index_name: str, embedding_property: str, dimension: int) -> None:
    session.run(f"DROP INDEX {index_name} IF EXISTS")

    create_query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR ()-[r:PLAYED_IN]-()
    ON r.{embedding_property}
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimension},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    session.run(create_query)


def wait_for_index_online(session, index_name: str, timeout_sec: int = 120, poll_every_sec: float = 2.0) -> bool:
    start = time.time()
    while True:
        rows = session.run(
            """
            SHOW INDEXES YIELD name, type, state
            WHERE type = 'VECTOR'
            RETURN name, state
            """
        ).data()

        state = None
        for r in rows:
            if r["name"] == index_name:
                state = r["state"]
                break

        if state == "ONLINE":
            print(f"  ✓ Index '{index_name}' is ONLINE")
            return True

        if time.time() - start > timeout_sec:
            print(f"  ✗ Timeout waiting for index '{index_name}' to be ONLINE (last state: {state})")
            return False

        print(f"  … waiting for index '{index_name}' (state: {state})")
        time.sleep(poll_every_sec)


def store_embeddings_batch(tx, batch: List[Dict[str, Any]], embedding_property: str, text_property: str) -> None:
    query = f"""
    UNWIND $batch as row
    MATCH (p:Player)-[pi:PLAYED_IN]->(f:Fixture)
    WHERE id(p) = row.player_id AND id(f) = row.fixture_id
    SET pi.{embedding_property} = row.embedding,
        pi.{text_property} = row.text
    """
    tx.run(query, batch=batch)


def generate_and_store_embeddings(driver, model_name: str, model_path: str, index_name: str, records: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 60}")
    print(f"Processing model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Index name: {index_name}")
    print(f"{'=' * 60}\n")

    embedding_property = f"embedding_{model_name}"
    text_property = f"embed_text_{model_name}"

    print(f"Loading {model_name} model...")
    model = SentenceTransformer(model_path)
    dim = model.get_sentence_embedding_dimension()
    print(f"✓ Model loaded. Embedding dimension: {dim}\n")

    print("Building query-friendly texts...")
    texts: List[str] = []
    for r in tqdm(records, desc="Building texts"):
        texts.append(create_sparse_text(r))

    print(f"\nEncoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    payload: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        payload.append(
            {
                "player_id": r["player_id"],
                "fixture_id": r["fixture_id"],
                "embedding": embeddings[i].tolist(),
                "text": texts[i],
            }
        )

    print(f"\nStoring embeddings + text properties in batches (BATCH_SIZE={BATCH_SIZE})...")
    with driver.session() as session:
        for i in tqdm(range(0, len(payload), BATCH_SIZE), desc="Writing"):
            batch = payload[i : i + BATCH_SIZE]
            session.execute_write(store_embeddings_batch, batch, embedding_property, text_property)

    print("Creating vector index...")
    with driver.session() as session:
        create_vector_index(session, index_name, embedding_property, dim)
        if not wait_for_index_online(session, index_name):
            raise RuntimeError(f"Vector index '{index_name}' failed to become ONLINE")

    print(f"✓ Completed {model_name}\n")


def main() -> None:
    cfg = load_config()
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))

    try:
        print("Fetching player-gameweek data from Neo4j...")
        with driver.session() as session:
            records = fetch_player_gameweek_data(session)

        if not records:
            print("No records found. Please ensure data is loaded into Neo4j.")
            return

        for model_name, (model_path, index_name) in MODEL_CONFIG.items():
            generate_and_store_embeddings(driver, model_name, model_path, index_name, records)

        print("\nVector indices created:")
        for model_name, (_, index_name) in MODEL_CONFIG.items():
            print(f"  - {index_name} (model: {model_name})")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
