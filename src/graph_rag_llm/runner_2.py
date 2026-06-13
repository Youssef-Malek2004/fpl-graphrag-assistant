# src/graph_rag_llm/runner.py
import os
import sys
import json
import argparse
from dotenv import load_dotenv

from neo4j import GraphDatabase

from models import LLMBackend
from prompt_builder import build_messages
from evidence_selector import select_evidence
from context_builder import format_fpl_context_from_evidence


SUPPORTED_MODELS = {
    "DeepSeek": "deepseek/deepseek-chat-v3.1",
    "Llama 70B Free": "meta-llama/llama-3.3-70b-instruct:free",
    "Nova 2 Lite Free": "amazon/nova-2-lite-v1:free",
}

SUPPORTED_RETRIEVAL_MODES = ["baseline", "hybrid"]  # only these


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
        intent=llm_context["intent"],
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


# ---------------------------
# Build llm_context using your connected pipeline
# ---------------------------
def build_llm_context_from_pipeline(
    *,
    query: str,
    config_path: str,
    model_name: str = "miniLM",
    semantic_top_k: int = 10,
    debug: bool = False,
    vocab_csv_path: str = "../neo4j/fpl_two_seasons.csv",
):
    """
    Uses:
      vocab = load_fpl_vocab(...)
      pre = process_input(query, vocab) -> intent/entities
      payload = run_connected_pipeline(...)
      return payload["llm_context"]
    """

    # Ensure imports work whether you run from repo root, src, etc.
    # If your project layout is "src/..." but you run from repo root,
    # this helps Python find top-level modules like pipeline.py.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Import your pipeline pieces (same ones used in your main)
    from src.semantic_query import load_config
    from src.pipeline import process_input, load_fpl_vocab
    from src.stage_two_pipeline import run_connected_pipeline

    vocab = load_fpl_vocab(vocab_csv_path)
    pre = process_input(query, vocab)

    intent = pre["intent"]
    entities = pre["entities"]

    cfg = load_config(config_path)
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))

    try:
        payload = run_connected_pipeline(
            driver=driver,
            query=query,
            intent=intent,
            entities=entities,
            model_name=model_name,
            semantic_top_k=semantic_top_k,
            debug=debug,
        )
    finally:
        driver.close()

    if "llm_context" not in payload:
        raise RuntimeError("Pipeline payload missing 'llm_context' key.")

    return payload["llm_context"], payload  # returning payload is handy for debugging


def main():
    parser = argparse.ArgumentParser(description="Run pipeline -> build llm_context -> call LLM")
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--openrouter-key", type=str, default=None)

    # These should match your connected pipeline knobs
    parser.add_argument("--config", type=str, default="../neo4j/config.txt")
    parser.add_argument("--vocab-csv", type=str, default="../neo4j/fpl_two_seasons.csv")
    parser.add_argument("--semantic-model", choices=["miniLM", "mpnet"], default="miniLM")
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    load_dotenv()
    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    print("here is the api key")
    print(api_key)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found. Put it in .env as OPENROUTER_API_KEY=...")

    llm_context, payload = build_llm_context_from_pipeline(
        query=args.query,
        config_path=args.config,
        model_name=args.semantic_model,
        semantic_top_k=args.semantic_top_k,
        debug=args.debug,
        vocab_csv_path=args.vocab_csv,
    )

    if args.debug:
        print("\n--- PIPELINE PAYLOAD (debug) ---\n")
        print(json.dumps(payload.get("input", {}), indent=2, ensure_ascii=False))
        print("\n(intent/entities from process_input are inside payload['input'])\n")

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


if __name__ == "__main__":
    main()
