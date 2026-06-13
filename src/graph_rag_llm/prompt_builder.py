# src/graph_rag_llm/prompt_builder.py
from typing import List, Dict, Any



FPL_PERSONA = """
You are an expert Fantasy Premier League (FPL) assistant.
You ONLY use the information from the provided FPL Knowledge Graph context.
If the context does not contain enough information to answer, clearly say:
"I don't have enough information in the knowledge graph to answer this confidently."
Do not invent players, numbers, or matches that are not in the context.
Explain your reasoning in simple language and give actionable advice for FPL managers.
"""


def build_messages(user_query: str, context_text: str, intent: str) -> List[Dict[str, str]]:
    """
    Build chat messages for an LLM (OpenAI or other chat model).
    """
    task = f"""
TASK:
The user's intent is: {intent if isinstance(intent, str) else intent.value
}.
Use ONLY the information in the CONTEXT below to answer the user's question.
- Do NOT hallucinate players, stats or fixtures that are not mentioned.
- If multiple options are reasonable, compare them and explain trade-offs.
- If context is insufficient, say so explicitly.
- Treat CYPHER evidence as the ground truth for the asked season/gameweek.
- Semantic evidence is only extra context and MUST NOT override CYPHER facts.


CONTEXT:
{context_text}
"""
    return [
        {"role": "system", "content": FPL_PERSONA.strip()},
        {"role": "system", "content": task.strip()},
        {"role": "user", "content": user_query}
    ]
