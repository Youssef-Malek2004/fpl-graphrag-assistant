# src/input_embedding.py

from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 5) INPUT EMBEDDING
# ============================================================

EMBEDDING_MODELS = {
    "miniLM": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
}


def generate_embeddings(text: str) -> Dict[str, List[float]]:
    """
    Generate embeddings using multiple models.
    Returns a dict: {model_name: embedding_vector}
    """
    embeddings = {}

    for model_name, model in EMBEDDING_MODELS.items():
        vec = model.encode([text], convert_to_numpy=True)[0]
        embeddings[model_name] = vec.tolist()

    return embeddings


# ============================================================
#  VALIDATING EMBEDDINGS (EXACT SAME AS NOTEBOOK)
# ============================================================

def _cos(a, b):
    return cosine_similarity(
        np.array(a).reshape(1, -1),
        np.array(b).reshape(1, -1)
    )[0][0]


if __name__ == "__main__":
    q1 = "Haaland or Kane?"
    q2 = "Compare Haaland vs Kane"
    q3 = "Show me Arsenal defenders"

    e1 = generate_embeddings(q1)["miniLM"]
    e2 = generate_embeddings(q2)["miniLM"]
    e3 = generate_embeddings(q3)["miniLM"]

    print("Similar:", _cos(e1, e2))
    print("Different:", _cos(e1, e3))
