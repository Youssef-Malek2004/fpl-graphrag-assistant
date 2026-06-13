# FPL Graph-RAG Assistant ⚽

A **GraphRAG question-answering system** over Fantasy Premier League data. Ask in plain English — _"compare Salah and Saka last season"_, _"who should I pick at midfield gameweek 10?"_ — and the system parses the question, walks a **Neo4j knowledge graph**, retrieves grounded evidence with a **hybrid Cypher + vector search**, and has an LLM answer **only** from that retrieved context.

Built on two full Premier League seasons modeled as a knowledge graph of players, teams, fixtures, gameweeks, and positions.

> **Authorship.** Originally a team project for the Advanced Concepts Lab (GUC, "Team 28"). This is a cleaned standalone showcase; I was the primary contributor (60+ commits), and built most of `src/` — the NLU pipeline, the Cypher/vector retrieval, and the Graph-RAG LLM layer. Secrets and personal data have been removed.

---

## Why GraphRAG (not plain vector RAG)

FPL questions are inherently **relational and numeric**: "midfielders who played _against_ Arsenal and scored > 6 points in the _last 3_ gameweeks." Flattening that into text chunks for vector similarity loses the structure. Modeling players ↔ fixtures ↔ teams ↔ gameweeks as a **graph** lets the system answer with precise traversals, then fall back to semantic similarity for fuzzy/under-specified queries — a **hybrid** retrieval strategy.

---

## Architecture

```mermaid
flowchart TD
    Q[Natural-language question] --> CLEAN[Text normalization<br/>regex + Porter stemming + FPL synonyms]
    CLEAN --> NER[Entity extraction<br/>spaCy EntityRuler over FPL vocab]
    NER --> INTENT[Intent classification<br/>recommend / compare / lookup ...]
    INTENT --> EMB[Query embedding<br/>SentenceTransformer]

    EMB --> RET{Hybrid retrieval}
    NER --> RET
    RET -->|structured| CYPHER[Manual Cypher builder<br/>entity + intent → graph traversal]
    RET -->|semantic| VEC[Vector similarity search<br/>+ numeric-constraint filtering]
    CYPHER --> KG[(Neo4j Knowledge Graph)]
    VEC --> KG

    KG --> EVID[Evidence selection<br/>baseline | hybrid]
    EVID --> CTX[Context builder<br/>graph evidence → grounded prompt]
    CTX --> LLM[LLM backend<br/>OpenRouter: DeepSeek v3.1 / Llama-3.3-70B]
    LLM --> A[Grounded answer]
```

### The knowledge graph

Built by [`neo4j/create_kg.py`](neo4j/create_kg.py) from `neo4j/fpl_two_seasons.csv`, with uniqueness constraints on every entity:

| Node | Key | Relationships |
|---|---|---|
| `Season` | `season_name` | `HAS_GW` → Gameweek |
| `Gameweek` | (season, GW) | `HAS_FIXTURE` → Fixture |
| `Fixture` | (season, fixture) | `HAS_HOME_TEAM` / `HAS_AWAY_TEAM` → Team |
| `Team` | `name` | home/away in fixtures |
| `Player` | (name, element) | `PLAYED_IN` → Fixture, `PLAYS_AS` → Position |
| `Position` | name | groups players |

A **vector index** over node embeddings supports the semantic-search path alongside exact Cypher traversals.

### Retrieval pipeline (two stages)

1. **NLU** ([`src/pipeline.py`](src/pipeline.py)) — `clean_text` → `extract_entities` → `classify_intent` → `should_generate_embedding` → query embedding. Builds an FPL vocabulary (players/teams/positions/seasons/gameweeks) straight from the dataset.
2. **Retrieval + answer** ([`src/stage_two_pipeline.py`](src/stage_two_pipeline.py)) — `build_cypher` produces a targeted graph query from the parsed entities + intent; `filtered_similarity_search` / `vector_search_relationships` add semantic recall with numeric-constraint extraction; results are parsed, evidence is selected (`baseline` vs `hybrid`), and the Graph-RAG layer ([`src/graph_rag_llm/`](src/graph_rag_llm/)) formats grounded context and calls the LLM.

---

## Repository layout

| Path | What's there |
|---|---|
| `src/` | NLU pipeline, entity extraction, intent classification, embeddings, Cypher builder, semantic/vector search |
| `src/graph_rag_llm/` | The Graph-RAG layer — `models` (LLM backend), `prompt_builder`, `evidence_selector`, `context_builder`, `runner`, Streamlit `app.py` |
| `neo4j/` | `create_kg.py` (graph build + constraints), Cypher queries, local setup `guide.txt` |
| `scripts/` | The predictive-modeling side — `data_cleaning`, `feature_engineering`, `model_training`, `explainability`, `data_visualization` |
| `tf_models/` | `FFNNRegressorModel` — a TensorFlow feed-forward regressor for player-points prediction |
| `notebooks/` | End-to-end analysis notebooks |
| `data/`, `neo4j/fpl_two_seasons.csv` | Two seasons of public FPL data |

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start Neo4j locally (see neo4j/guide.txt for the full docker command)
docker run -d --name neo4j-fpl -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-neo4j-password neo4j:5

# 3. Configure
cp .env.example .env          # add your OpenRouter API key
#   set Neo4j URI / user / password in neo4j/config.txt

# 4. Build the knowledge graph
python neo4j/create_kg.py

# 5. Launch the Graph-RAG assistant (Streamlit UI)
streamlit run src/graph_rag_llm/app.py
```

---

## Highlights

- **True hybrid GraphRAG** — structured Cypher traversals for relational/numeric questions, plus vector similarity over graph-node embeddings for fuzzy ones, with selectable `baseline` vs `hybrid` evidence modes.
- **Real NLU front-end** — not just "embed the question": regex+stemming normalization with FPL-specific synonym rules, spaCy `EntityRuler` grounded in the dataset vocabulary, and intent classification that routes the query.
- **Grounded generation** — the LLM only ever sees retrieved graph evidence, reducing hallucination; swappable OpenRouter models (DeepSeek v3.1, Llama-3.3-70B).
- **Beyond retrieval** — a parallel ML track (feature engineering → TensorFlow regressor → explainability) for points prediction.
- **Interactive UI** — a Streamlit assistant with graph visualizations.

---

## Tech Stack

**Knowledge graph:** Neo4j (Cypher, constraints, vector index) ·
**NLU:** spaCy (EntityRuler), NLTK (Porter stemmer), SentenceTransformers ·
**LLM:** OpenRouter (DeepSeek v3.1, Llama-3.3-70B) ·
**ML:** TensorFlow, scikit-learn, pandas ·
**UI:** Streamlit, Plotly, NetworkX

## License

[MIT](LICENSE). FPL data is public.
