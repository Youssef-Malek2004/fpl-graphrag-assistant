# src/graph_rag_llm/evidence_selector.py
from typing import List, Dict, Any

def select_evidence(evidence: List[Dict[str, Any]], retrieval_mode: str):
    """
    retrieval_mode: baseline | hybrid
    baseline -> only cypher
    hybrid   -> cypher first, then semantic
    """
    if retrieval_mode == "baseline":
        return [e for e in evidence if e.get("source") == "cypher"]

    if retrieval_mode == "hybrid":
        cypher = [e for e in evidence if e.get("source") == "cypher"]
        semantic = [e for e in evidence if e.get("source") == "semantic"]
        return cypher + semantic

    raise ValueError("retrieval_mode must be one of: baseline | hybrid")
