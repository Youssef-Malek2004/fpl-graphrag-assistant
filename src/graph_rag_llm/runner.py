# src/graph_rag_llm/runner.py
import os
from dotenv import load_dotenv

from .models import LLMBackend
from .prompt_builder import build_messages
from .evidence_selector import select_evidence
from .context_builder import format_fpl_context_from_evidence


SUPPORTED_MODELS = {
    "DeepSeek": "deepseek/deepseek-chat-v3.1",
    "Llama 70B Free": "meta-llama/llama-3.3-70b-instruct:free",
    "Nova 2 Lite Free": "amazon/nova-2-lite-v1:free",
}

SUPPORTED_RETRIEVAL_MODES = ["baseline", "hybrid"]  # you said you only want these


def run_llm_from_context(
    llm_context: dict,
    retrieval_mode: str,   # "baseline" | "hybrid"
    model_name: str,
    api_key: str,
):
    if retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of {SUPPORTED_RETRIEVAL_MODES}")

    evidence = llm_context.get("evidence", [])
    selected = select_evidence(evidence, retrieval_mode)

    context_text = format_fpl_context_from_evidence(llm_context, selected)

    messages = build_messages(
        user_query=llm_context["query"],
        context_text=context_text,
        intent=llm_context["intent"],  # intent is a string now
    )

    backend = LLMBackend(api_key=api_key)
    answer, meta = backend.generate(model=model_name, messages=messages)

    return {
        "answer": answer,
        "meta": meta,
        "context_text": context_text,
        "selected_evidence": selected,
        "retrieval_mode": retrieval_mode,
        "model_name": model_name,
    }


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found. Put it in .env as OPENROUTER_API_KEY=...")

    # ✅ PASTE YOUR TEST CONTEXT HERE (use the INNER object, NOT wrapped in {"llm_context": ...})
    llm_context = {
        "query": "how did john stones do in gw 5 2021-22",
        "canonical_query": "how did john stones do in gw=5 season=2021-22",
        "intent": "PLAYER_PERFORMANCE",
        "entities": {
            "players": ["John Stones"],
            "gameweek": 5,
            "season": "2021-22",
            "season_alt": "2021/22"
        },
        "evidence": [
            {
                "source": "cypher",
                "confidence": 1.0,
                "item": {
                    "fixture": {
                        "kickoff_time": "2021-09-18 14:00:00+00:00",
                        "fixture_number": 45,
                        "team_h_score": 0,
                        "team_a_score": 0
                    },
                    "stats": {
                        "goals_scored": 0,
                        "assists": 0,
                        "bonus": 0,
                        "minutes": 0,
                        "total_points": 0
                    },
                    "kind": "player_gameweek",
                    "season": "2021-22",
                    "gameweek": 5,
                    "home_team": {"name": "Man City"},
                    "away_team": {"name": "Southampton"},
                    "player": {"player_element": 252, "player_name": "John Stones"},
                    "intent": "PLAYER_PERFORMANCE",
                },
                "semantic_match": None
            },
            {
                "source": "semantic",
                "confidence": 0.7993543148040771,
                "item": {
                    "kind": "player_gameweek",
                    "intent": "PLAYER_PERFORMANCE",
                    "player": {"player_name": "John Stones", "player_element": None},
                    "position": "FWD",
                    "season": "2021-22",
                    "gameweek": 29,
                    "fixture": {
                        "fixture_number": None,
                        "kickoff_time": None,
                        "team_h_score": None,
                        "team_a_score": None
                    },
                    "home_team": {"name": "Crystal Palace"},
                    "away_team": {"name": "Man City"},
                    "stats": {
                        "minutes": 90,
                        "total_points": 7,
                        "goals_scored": 0,
                        "assists": 0,
                        "bonus": 1
                    },
                    "semantic_score": 0.7993543148040771
                },
                "semantic_match": {"model": "miniLM", "score": 0.7993543148040771}
            },
            {
                "source": "semantic",
                "confidence": 0.7978322505950928,
                "item": {
                    "kind": "player_gameweek",
                    "intent": "PLAYER_PERFORMANCE",
                    "player": {"player_name": "John Stones", "player_element": None},
                    "position": "DEF",
                    "season": "2021-22",
                    "gameweek": 10,
                    "fixture": {
                        "fixture_number": None,
                        "kickoff_time": None,
                        "team_h_score": None,
                        "team_a_score": None
                    },
                    "home_team": {"name": "Man City"},
                    "away_team": {"name": "Crystal Palace"},
                    "stats": {
                        "minutes": 31,
                        "total_points": 1,
                        "goals_scored": 0,
                        "assists": 0,
                        "bonus": 0
                    },
                    "semantic_score": 0.7978322505950928
                },
                "semantic_match": {"model": "miniLM", "score": 0.7978322505950928}
            }
        ]
    }

    for mode in SUPPORTED_RETRIEVAL_MODES:
        for label, model_id in SUPPORTED_MODELS.items():
            try:
                out = run_llm_from_context(
                    llm_context=llm_context,
                    retrieval_mode=mode,
                    model_name=model_id,
                    api_key=api_key,
                )

                print("\n==============================")
                print(f"MODE: {mode} | MODEL: {label}")
                print("==============================\n")
                print(out["context_text"])
                print("\n--- ANSWER ---\n")
                print(out["answer"])
                print("\n--- META ---\n")
                print(out["meta"])

            except Exception as e:
                print("\n==============================")
                print(f"MODE: {mode} | MODEL: {label}  -> ERROR")
                print("==============================")
                print(str(e))
