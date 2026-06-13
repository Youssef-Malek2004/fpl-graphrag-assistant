import argparse
import json
from typing import Dict, List, Any

from neo4j import GraphDatabase, Driver
from sentence_transformers import SentenceTransformer

from src.cypher_manual_queries_builder import build_cypher
from src.manual_parsers import parse_results_for_llm
from src.semantic_query import (
    MODEL_CONFIG,
    load_config,
    canonicalize_query,
    extract_numeric_constraints,
    verify_index_exists,
    filtered_similarity_search,
    vector_search_relationships,
    CANDIDATE_MULTIPLIER,
)

# from src.pipeline import process_input, load_fpl_vocab
from src.pipeline import process_input, load_fpl_vocab



# ---------------------------
# Entity normalization (baseline only)
# ---------------------------
def _normalize_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize baseline entities (used only for CYPHER baseline)."""
    e = dict(entities or {})

    if "gameweek" in e and e["gameweek"] is not None:
        try:
            e["gameweek"] = int(e["gameweek"])
        except Exception:
            pass

    if "positions" in e and "position" not in e:
        if isinstance(e["positions"], list) and e["positions"]:
            e["position"] = e["positions"][0]

    if "season" in e and isinstance(e["season"], str):
        s = e["season"]
        if "-" in s:
            e["season_alt"] = s.replace("-", "/")
        elif "/" in s:
            e["season_alt"] = s.replace("/", "-")

    return e



# ---------------------------
# Semantic retrieval wrapper
# ---------------------------
def semantic_items(
    session,
    *,
    index_name: str,
    model_name: str,
    query_embedding: List[float],
    top_k: int,
    intent: str,
    canon_query: str,
) -> List[Dict[str, Any]]:
    """
    Semantic retrieval producing pipeline `item` dicts.
    Uses semantic_query.py logic for retrieval.
    """
    numeric_filters = extract_numeric_constraints(canon_query)
    candidate_k = max(top_k * CANDIDATE_MULTIPLIER, top_k)

    # Use semantic_query functions
    if numeric_filters:
        rows = filtered_similarity_search(
            session=session,
            model_name=model_name,
            query_embedding=query_embedding,
            top_k=top_k,
            numeric_filters=numeric_filters,
        )
    else:
        rows = vector_search_relationships(
            session=session,
            index_name=index_name,
            query_embedding=query_embedding,
            top_k=top_k,
            numeric_filters=numeric_filters,
            candidate_k=candidate_k,
        )

    # Convert to pipeline item format
    items: List[Dict[str, Any]] = []
    for r in rows:
        it = {
            "kind": "player_gameweek",
            "intent": intent,
            "player": {
                "player_name": r.get("player_name"),
                "player_element": r.get("player_element"),
            },
            "position": r.get("position"),
            "season": r.get("season"),
            "gameweek": r.get("gw_number"),
            "fixture": {
                "fixture_number": r.get("fixture_number"),
                "kickoff_time": r.get("kickoff_time"),
                "team_h_score": r.get("team_h_score"),
                "team_a_score": r.get("team_a_score"),
            },
            "home_team": {"name": r.get("home_team")},
            "away_team": {"name": r.get("away_team")},
            "stats": {
                "minutes": r.get("minutes"),
                "total_points": r.get("total_points"),
                "goals_scored": r.get("goals_scored"),
                "assists": r.get("assists"),
                "bonus": r.get("bonus"),
            },
            "semantic_score": r.get("score"),
        }
        items.append(it)

    return items


# ---------------------------
# Dedupe key (kind-aware)
# ---------------------------
def _safe_json_hash(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(obj)


def _evidence_key(item: Dict[str, Any]) -> str:
    kind = (item.get("kind") or "unknown").strip().lower()

    if kind == "player_gameweek":
        p = item.get("player") or {}
        fixture = item.get("fixture") or {}
        pid = p.get("player_element") or (p.get("player_name") or "").strip().lower()
        season = str(item.get("season") or "")
        gw = item.get("gameweek")
        fx = fixture.get("fixture_number")
        return f"player_gameweek|{pid}|{season}|{gw}|{fx}"

    if kind == "player_season":
        p = item.get("player") or {}
        pid = p.get("player_element") or (p.get("player_name") or "").strip().lower()
        season = str(item.get("season") or "")
        return f"player_season|{pid}|{season}"

    if kind.startswith("team"):
        team = (item.get("team") or {}).get("name") or item.get("team_name") or ""
        season = str(item.get("season") or "")
        gw = item.get("gameweek") or item.get("gw_number")
        # Include player info if present (for team_top_players_* kinds)
        p = item.get("player") or {}
        pid = p.get("player_element") or (p.get("player_name") or "").strip().lower()
        if pid:
            return f"{kind}|{team.strip().lower()}|{season}|{gw}|{pid}"
        return f"{kind}|{team.strip().lower()}|{season}|{gw}"

    return f"fallback|{kind}|{_safe_json_hash(item)}"


# ---------------------------
# Merge baseline + semantic
# ---------------------------
def build_unified_llm_context(
    *,
    raw_query: str,
    canonical_query: str,
    intent: str,
    entities: Dict[str, Any],
    baseline_items: List[Dict[str, Any]],
    semantic_items_list: List[Dict[str, Any]],
    model_name: str,
) -> Dict[str, Any]:
    evidence_by_key: Dict[str, Dict[str, Any]] = {}

    # Add baseline items
    for item in baseline_items:
        if not isinstance(item, dict):
            continue
        k = _evidence_key(item)
        evidence_by_key[k] = {
            "source": "cypher",
            "confidence": 1.0,
            "item": item,
            "semantic_match": None,
        }

    # Add/merge semantic items
    for item in semantic_items_list:
        if not isinstance(item, dict):
            continue
        k = _evidence_key(item)
        sem_score = item.get("semantic_score", 0.0)
        try:
            sem_score_f = float(sem_score)
        except Exception:
            sem_score_f = 0.0

        if k in evidence_by_key:
            evidence_by_key[k]["semantic_match"] = {"model": model_name, "score": sem_score_f}
        else:
            evidence_by_key[k] = {
                "source": "semantic",
                "confidence": sem_score_f,
                "item": item,
                "semantic_match": {"model": model_name, "score": sem_score_f},
            }

    evidence = list(evidence_by_key.values())

    # Sort: cypher first, then by score
    def _rank_key(e: Dict[str, Any]):
        src_rank = 0 if e.get("source") == "cypher" else 1
        score = float(e.get("confidence") or 0.0)
        return (src_rank, -score)

    evidence.sort(key=_rank_key)

    return {
        "query": raw_query,
        "canonical_query": canonical_query,
        "intent": intent,
        "entities": entities,
        "evidence": evidence,
    }

def _normalize_dynamic_metric_aggregate(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    If a Cypher item returns aggregates: {metric_name, metric_total},
    convert it to aggregates: { total_<metric_name>: metric_total }.
    Safe no-op for other kinds.
    """
    if not isinstance(item, dict):
        return item

    aggs = item.get("aggregates")
    if not isinstance(aggs, dict):
        return item

    metric_name = aggs.get("metric_name") or item.get("metric")
    metric_total = aggs.get("metric_total")

    # only run when it matches your Option-A shape
    if metric_name is None or "metric_total" not in aggs:
        return item

    key = f"total_{str(metric_name).strip().lower()}"
    item["aggregates"] = {key: metric_total}
    return item


# ---------------------------
# Main pipeline
# ---------------------------
def run_connected_pipeline(
    driver: Driver,
    query: str,
    intent: str,
    entities: Dict[str, Any],
    model_name: str = "miniLM",
    semantic_top_k: int = 10,
    debug: bool = False,
) -> Dict[str, Any]:
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_CONFIG.keys())}")

    entities_norm = _normalize_entities(entities)
    model_path, index_name = MODEL_CONFIG[model_name]
    canon_query = canonicalize_query(query)

    # 1) Semantic retrieval (independent from entities)
    semantic_list: List[Dict[str, Any]] = []
    model = SentenceTransformer(model_path)
    q_emb = model.encode(canon_query, normalize_embeddings=True).tolist()

    with driver.session() as session:
        if verify_index_exists(session, index_name):
            semantic_list = semantic_items(
                session,
                index_name=index_name,
                model_name=model_name,
                query_embedding=q_emb,
                top_k=semantic_top_k,
                intent=intent,
                canon_query=canon_query,
            )
        elif debug:
            print(f"[WARN] Vector index '{index_name}' not ONLINE / missing. Skipping semantic retrieval.")

    # 2) Baseline Cypher from (intent, entities)
    cypher, params = build_cypher(intent, entities_norm)

    # 3) Execute baseline Cypher
    with driver.session() as session:
        result = session.run(cypher, params)
        records = list(result)

    # 4) Parse baseline
    parsed = parse_results_for_llm(intent, entities_norm, records)
    baseline_items = parsed.get("results") or []
    if not isinstance(baseline_items, list):
        baseline_items = []

    baseline_items = [_normalize_dynamic_metric_aggregate(it) for it in baseline_items if isinstance(it, dict)]
    parsed["results"] = baseline_items

    # Optional: noise suppression for season-level queries
    semantic_for_context = semantic_list
    if intent.upper() == "PLAYER_PERFORMANCE" and entities_norm.get("season") and entities_norm.get("gameweek") is None:
        if any(isinstance(x, dict) and x.get("kind") == "player_season" for x in baseline_items):
            semantic_for_context = []

    # 5) Merge baseline + semantic
    llm_context = build_unified_llm_context(
        raw_query=query,
        canonical_query=canon_query,
        intent=intent,
        entities=entities_norm,
        baseline_items=baseline_items,
        semantic_items_list=semantic_for_context,
        model_name=model_name,
    )

    out: Dict[str, Any] = {
        "input": {
            "raw_query": query,
            "canonical_query": canon_query,
            "intent": intent,
            "entities": entities_norm,
            "model": model_name,
        },
        "baseline": {
            "cypher": {"query": cypher, "params": params, "returned_records": len(records)},
            "parsed": parsed,
        },
        "semantic": {
            "model": model_name,
            "top_k": semantic_top_k,
            "items": semantic_list,
        },
        "llm_context": llm_context,
    }

    if debug:
        out["debug"] = {
            "first_baseline_record_preview": records[0].data() if records else None,
            "semantic_top1": semantic_list[0] if semantic_list else None,
        }

    return out




# ---------------------------
# CLI
# ---------------------------
# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="(query + intent + entities) -> baseline cypher + semantic retrieval + merged llm_context"
#     )
#     parser.add_argument("--query", required=True, type=str)
#     parser.add_argument("--intent", required=True, type=str)
#     parser.add_argument("--entities", required=True, type=str)
#     parser.add_argument("--model", choices=list(MODEL_CONFIG.keys()), default="miniLM")
#     parser.add_argument("--semantic-top-k", type=int, default=10)
#     parser.add_argument("--config", type=str, default="../neo4j/config.txt")
#     parser.add_argument("--debug", action="store_true")
#     args = parser.parse_args()

#     try:
#         entities = json.loads(args.entities)
#         if not isinstance(entities, dict):
#             raise ValueError("entities must be a JSON object (dict).")
#     except Exception as e:
#         raise SystemExit(f"Failed to parse --entities JSON: {e}")

#     cfg = load_config(args.config)
#     driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))

#     try:
#         payload = run_connected_pipeline(
#             driver=driver,
#             query=args.query,
#             intent=args.intent,
#             entities=entities,
#             model_name=args.model,
#             semantic_top_k=args.semantic_top_k,
#             debug=args.debug,
#         )
#         print(json.dumps(payload, indent=2, ensure_ascii=False))
#     finally:
#         driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="(query + intent + entities) -> baseline cypher + semantic retrieval + merged llm_context"
    )
    parser.add_argument("--query", required=True, type=str)
    # parser.add_argument("--intent", required=True, type=str)
    # parser.add_argument("--entities", required=True, type=str)
    parser.add_argument("--model", choices=list(MODEL_CONFIG.keys()), default="miniLM")
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--config", type=str, default="../neo4j/config.txt")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # try:
    #     entities = json.loads(args.entities)
    #     if not isinstance(entities, dict):
    #         raise ValueError("entities must be a JSON object (dict).")
    # except Exception as e:
    #     raise SystemExit(f"Failed to parse --entities JSON: {e}")

    
    vocab = load_fpl_vocab("../neo4j/fpl_two_seasons.csv")
    pre = process_input(args.query, vocab)

    intent = pre["intent"]
    entities = pre["entities"]

    print(intent)
    print(entities)

    cfg = load_config(args.config)
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))

    try:
        payload = run_connected_pipeline(
            driver=driver,
            query=args.query,
            # intent=args.intent,
            intent=intent,
            entities=entities,
            model_name=args.model,
            semantic_top_k=args.semantic_top_k,
            debug=args.debug,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        driver.close()



if __name__ == "__main__":
    main()