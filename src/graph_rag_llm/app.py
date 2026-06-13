import streamlit as st
import os
import sys
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from neo4j import GraphDatabase

# Add project root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import your pipeline components
from src.graph_rag_llm.models import LLMBackend
from src.graph_rag_llm.prompt_builder import build_messages
from src.graph_rag_llm.evidence_selector import select_evidence
from src.graph_rag_llm.context_builder import format_fpl_context_from_evidence
from src.semantic_query import load_config
from src.pipeline import process_input, load_fpl_vocab
from src.stage_two_pipeline import run_connected_pipeline

# Page config
st.set_page_config(
    page_title="FPL Graph-RAG Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #37003c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #00ff87;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .evidence-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #37003c;
        margin: 0.5rem 0;
    }
    .cypher-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        margin: 0.5rem 0;
    }
    /* Align text input and selectbox vertically */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }
    /* Ensure input widgets align at the bottom */
    div[data-testid="column"] > div[data-baseweb] {
        margin-top: auto;
    }
</style>
""", unsafe_allow_html=True)

# Constants
SUPPORTED_MODELS = {
    "DeepSeek": "deepseek/deepseek-chat-v3.1",
    "Llama 70B Free": "meta-llama/llama-3.3-70b-instruct:free",
    "Nova 2 Lite Free": "amazon/nova-2-lite-v1:free",
    "Nemotron 30B Free": "nvidia/nemotron-3-nano-30b-a3b:free",
    "Hermes 3 405B (Free)": "nousresearch/hermes-3-llama-3.1-405b:free",
}

RETRIEVAL_MODES = {
    "Baseline (Cypher Only)": "baseline",
    "Hybrid (Cypher + Embeddings)": "hybrid"
}

EMBEDDING_MODELS = ["miniLM", "mpnet"]

# Sample queries for quick testing
SAMPLE_QUERIES = [
    "What were Mohamed Salah's gameweek stats in GW5 of 2021-22?",
    "Show me Mohamed Salah's overall performance in 2021-22 season",
    "Who were the top 10 forwards by points in 2021-22?",
    "Top 10 midfielders by goals in 2021-22",
    "Who were the top performing players in gameweek 10 of 2021-22?",
    "Compare Mohamed Salah and John Stones in gameweek 5 of 2021-22",
    "Compare between the overall season of Bukayo Saka and Harry Kane in season 2021-2022.",
    "Compare Liverpool and Manchester City in 2021-22",
    "Show me Liverpool's top 5 players in 2021-22",
    "Who was Liverpool's top 2 defenders in gameweek 13 of 2021-22?",
    "Recommend 5 defenders for gameweek 10 in 2021-22",
    "What was the average points for midfielders in 2021-22?",
    "Total goals scored in 2021-22 season",
    "Who were the top 3 consistent players with accordance to points in 2021-22?",
]

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'llm_context' not in st.session_state:
    st.session_state.llm_context = None
if 'pipeline_payload' not in st.session_state:
    st.session_state.pipeline_payload = None


@st.cache_resource
def load_vocab_cached():
    """Load FPL vocabulary (cached)"""
    vocab_path = os.path.join(repo_root, "neo4j", "fpl_two_seasons.csv")
    return load_fpl_vocab(vocab_path)


@st.cache_resource
def get_neo4j_driver():
    """Get Neo4j driver (cached)"""
    config_path = os.path.join(repo_root, "neo4j", "config.txt")
    cfg = load_config(config_path)
    return GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))

def get_smart_default_season(entities: dict, user_selected_default: str) -> str:
    """
    Intelligently determine the season to use.
    
    Priority:
    1. Explicit season in entities (user mentioned specific season)
    2. If user said "this season" or "current season" -> use latest (2022-23)
    3. Otherwise, use user's selected default from sidebar
    """
    
    # If season already exists and is valid, use it
    if entities.get("season") and entities.get("season") not in [None, "unknown"]:
        return entities["season"]
    
    # Check for temporal keywords that mean "current/latest"
    # This should ideally be done in entity extraction, but we can patch here
    raw_query = entities.get("raw_query", "").lower()
    current_indicators = ["this season", "current season", "now", "recent", "latest"]
    
    if any(indicator in raw_query for indicator in current_indicators):
        return "2022-23"  # Your most recent season
    
    # Otherwise use the sidebar default
    return user_selected_default

def build_llm_context_from_pipeline(
    query: str,
    semantic_model: str,
    semantic_top_k: int,
    default_season: str = "2022-23",  # ✅ Changed default
) -> tuple:
    """Build LLM context using the pipeline"""
    vocab = load_vocab_cached()
    pre = process_input(query, vocab)
    
    intent = pre["intent"]
    entities = pre["entities"]
    
    # ✅ Use smart season detection
    if not entities.get("season") or entities.get("season") is None:
        smart_season = get_smart_default_season(entities, default_season)
        
        # Handle "Both Seasons" option
        if smart_season == "Both Seasons":
            entities["season"] = "2022-23"
            entities["season_alt"] = "2022/23"
        else:
            entities["season"] = smart_season
            # Convert between formats
            if "-" in smart_season:
                entities["season_alt"] = smart_season.replace("-", "/")
            elif "/" in smart_season:
                entities["season_alt"] = smart_season.replace("/", "-")
    
    config_path = os.path.join(repo_root, "neo4j", "config.txt")
    driver = get_neo4j_driver()
    
    payload = run_connected_pipeline(
        driver=driver,
        query=query,
        intent=intent,
        entities=entities,
        model_name=semantic_model,
        semantic_top_k=semantic_top_k,
        debug=False,
    )
    
    return payload["llm_context"], payload


def run_llm_query(
    llm_context: Dict[str, Any],
    retrieval_mode: str,
    model_name: str,
    api_key: str,
) -> Dict[str, Any]:
    """Run LLM query with selected retrieval mode and model"""
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


def visualize_graph(evidence: List[Dict[str, Any]]):
    """Create a graph visualization of the evidence"""
    G = nx.Graph()
    
    # Add nodes and edges from evidence
    for e in evidence:
        item = e.get("item", {})
        player = item.get("player", {})
        player_name = player.get("player_name", "Unknown")
        home_team = item.get("home_team", {}).get("name", "")
        away_team = item.get("away_team", {}).get("name", "")
        
        # Add player node
        if player_name and player_name != "Unknown":
            G.add_node(player_name, node_type="player", color="#00ff87")
        
        # Add team nodes and edges
        if home_team:
            G.add_node(home_team, node_type="team", color="#37003c")
            if player_name and player_name != "Unknown":
                G.add_edge(player_name, home_team, relation="plays_for")
        
        if away_team:
            G.add_node(away_team, node_type="team", color="#37003c")
    
    if len(G.nodes()) == 0:
        return None
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color="#888"),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(G.nodes[node].get('color', '#888'))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=20,
            color=node_color,
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=400
        )
    )
    
    return fig


def display_evidence_details(evidence: List[Dict[str, Any]]):
    """Display detailed evidence in an organized way"""
    cypher_evidence = [e for e in evidence if e.get("source") == "cypher"]
    semantic_evidence = [e for e in evidence if e.get("source") == "semantic"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Cypher Evidence (Baseline)")
        if cypher_evidence:
            for i, e in enumerate(cypher_evidence):
                item = e.get("item", {})
                player = item.get("player", {})
                stats = item.get("stats", {})
                fixture = item.get("fixture", {})
                
                with st.expander(f"📊 {player.get('player_name', 'Unknown')} - GW{item.get('gameweek', 'N/A')}"):
                    st.write(f"**Season:** {item.get('season', 'N/A')}")
                    st.write(f"**Position:** {item.get('position', 'N/A')}")
                    st.write(f"**Fixture:** {item.get('home_team', {}).get('name', '')} vs {item.get('away_team', {}).get('name', '')}")
                    st.write(f"**Score:** {fixture.get('team_h_score', '-')}-{fixture.get('team_a_score', '-')}")
                    
                    st.markdown("**Performance:**")
                    stat_cols = st.columns(3)
                    stat_cols[0].metric("Points", stats.get('total_points', 0))
                    stat_cols[1].metric("Minutes", stats.get('minutes', 0))
                    stat_cols[2].metric("Goals", stats.get('goals_scored', 0))
                    
                    stat_cols2 = st.columns(3)
                    stat_cols2[0].metric("Assists", stats.get('assists', 0))
                    stat_cols2[1].metric("Bonus", stats.get('bonus', 0))
                    stat_cols2[2].metric("Confidence", f"{e.get('confidence', 1.0):.2f}")
        else:
            st.info("No Cypher evidence found")
    
    with col2:
        st.markdown("### 🔍 Semantic Evidence (Embeddings)")
        if semantic_evidence:
            for i, e in enumerate(semantic_evidence):
                item = e.get("item", {})
                player = item.get("player", {})
                stats = item.get("stats", {})
                score = e.get("confidence", 0)
                
                with st.expander(f"🎲 {player.get('player_name', 'Unknown')} - Similarity: {score:.3f}"):
                    st.write(f"**Season:** {item.get('season', 'N/A')}")
                    st.write(f"**Gameweek:** {item.get('gameweek', 'N/A')}")
                    st.write(f"**Position:** {item.get('position', 'N/A')}")
                    st.write(f"**Fixture:** {item.get('home_team', {}).get('name', '')} vs {item.get('away_team', {}).get('name', '')}")
                    
                    st.markdown("**Stats:**")
                    stat_cols = st.columns(2)
                    stat_cols[0].metric("Points", stats.get('total_points', 0))
                    stat_cols[1].metric("Minutes", stats.get('minutes', 0))
        else:
            st.info("No semantic evidence found (or baseline mode selected)")
def update_query_from_sample():
    """Callback to update the main query text input from the selectbox value."""
    # st.session_state.sample_selector holds the selected value from the selectbox
    selected_query = st.session_state.get("sample_selector", "")
    if selected_query: # Check if a valid query was selected (not the empty string)
        st.session_state.query_input = selected_query

def main():
    # Header
    st.markdown('<div class="main-header">⚽ FPL Graph-RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown("*Powered by Neo4j Knowledge Graph + LLM*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=os.environ.get("OPENROUTER_API_KEY", ""),
            help="Enter your OpenRouter API key"
        )
        
        st.divider()
        
        # Model selection
        st.subheader("🤖 LLM Model")
        selected_model_name = st.selectbox(
            "Choose LLM",
            list(SUPPORTED_MODELS.keys()),
            help="Select which language model to use"
        )
        
        st.divider()
        
        # Retrieval configuration
        st.subheader("🔍 Retrieval Settings")
        retrieval_mode_name = st.selectbox(
            "Retrieval Mode",
            list(RETRIEVAL_MODES.keys()),
            help="Baseline uses only Cypher queries, Hybrid adds semantic embeddings"
        )
        # Season Selection
        st.subheader("📅 Default Season")
        st.caption("⚠️ Only used if query doesn't specify a season")

        available_seasons = ["2022-23", "2021-22", "Both Seasons"]  # Most recent first
        default_season = st.selectbox(
            "Fallback season",
            available_seasons,
            index=0,  # Default to most recent (2022-23)
            help="This season is used ONLY when your query doesn't mention a specific season (e.g., 'Show me Salah's form')"
            )

        # Store in session state
        st.session_state.default_season = default_season

        
        semantic_model = st.selectbox(
            "Embedding Model",
            EMBEDDING_MODELS,
            help="Choose embedding model for semantic search"
        )
        
        semantic_top_k = st.slider(
            "Semantic Top-K",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of semantic search results to retrieve"
        )
        
        st.divider()
        
        # Compare models option
        compare_models = st.checkbox(
            "📊 Compare All Models",
            help="Run query on all available models side-by-side"
        )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎮 Query Interface", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        # Query input
        st.subheader("💬 Ask Your FPL Question")
        
        # Initialize session state for query if not exists
        if 'query_input' not in st.session_state:
            st.session_state.query_input = ""
    
        col1, col2 = st.columns([3, 1])
        with col1:
            # Text input synced with session state
            query = st.text_input(
                "Enter your question",
                placeholder="e.g., How many assists did Bukayo Saka make in gameweek 12?",
                label_visibility="collapsed",
                key="query_input"
                )
        
        with col2:
            selected_sample = st.selectbox(
                "Quick Select",
                [""] + SAMPLE_QUERIES,
                key="sample_selector",
                on_change=update_query_from_sample,
                help="Choose a sample query or type your own",
                label_visibility="collapsed"
            )
            

        # Run query button
        if st.button("🚀 Run Query", type="primary"):
            if not query:
                st.error("Please enter a query!")
                return
            
            if not api_key:
                st.error("Please enter your OpenRouter API key in the sidebar!")
                return
            
            with st.spinner("🔄 Processing your query..."):
                try:
                    # Build pipeline context
                    llm_context, pipeline_payload = build_llm_context_from_pipeline(
                        query=query,
                        semantic_model=semantic_model,
                        semantic_top_k=semantic_top_k,
                        default_season=st.session_state.get('default_season', '2021-22'),
                    )
                    
                    st.session_state.llm_context = llm_context
                    st.session_state.pipeline_payload = pipeline_payload
                    
                    # Run LLM
                    if compare_models:
                        results = {}
                        for model_label, model_id in SUPPORTED_MODELS.items():
                            try:
                                results[model_label] = run_llm_query(
                                    llm_context=llm_context,
                                    retrieval_mode=RETRIEVAL_MODES[retrieval_mode_name],
                                    model_name=model_id,
                                    api_key=api_key,
                                )
                            except Exception as e:
                                results[model_label] = {"error": str(e)}
                        st.session_state.results = results
                    else:
                        result = run_llm_query(
                            llm_context=llm_context,
                            retrieval_mode=RETRIEVAL_MODES[retrieval_mode_name],
                            model_name=SUPPORTED_MODELS[selected_model_name],
                            api_key=api_key,
                        )
                        st.session_state.results = {selected_model_name: result}
                    
                    st.success("✅ Query completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
        
        # Display results
        if st.session_state.results:
            st.divider()
            
            # Display answers
            st.markdown("## 🎯 LLM Answers")
            
            if len(st.session_state.results) > 1:
                # Multiple models - show in columns
                cols = st.columns(len(st.session_state.results))
                for idx, (model_name, result) in enumerate(st.session_state.results.items()):
                    with cols[idx]:
                        st.markdown(f"### {model_name}")
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.markdown(result["answer"])
                            
                            with st.expander("📈 Metadata"):
                                meta = result["meta"]
                                col1, col2 = st.columns(2)
                                col1.metric("Latency", f"{meta['latency']:.2f}s")
                                col2.metric("Total Tokens", meta['total_tokens'])
                                col1.metric("Prompt Tokens", meta['prompt_tokens'])
                                col2.metric("Completion Tokens", meta['completion_tokens'])
            else:
                # Single model
                model_name, result = list(st.session_state.results.items())[0]
                if "error" not in result:
                    st.markdown(result["answer"])
                    
                    # Metadata
                    with st.expander("📈 Performance Metrics"):
                        meta = result["meta"]
                        cols = st.columns(4)
                        cols[0].metric("⏱️ Latency", f"{meta['latency']:.2f}s")
                        cols[1].metric("📝 Total Tokens", meta['total_tokens'])
                        cols[2].metric("📤 Prompt Tokens", meta['prompt_tokens'])
                        cols[3].metric("📥 Completion Tokens", meta['completion_tokens'])
                else:
                    st.error(f"Error: {result['error']}")
            
            st.divider()
            
            # Display context and evidence
            if st.session_state.llm_context and st.session_state.pipeline_payload:
                st.markdown("## 🔍 Retrieved Knowledge Graph Context")
                
                # Tabs for different views
                ctx_tab1, ctx_tab2, ctx_tab3, ctx_tab4 = st.tabs([
                    "📋 Context Summary",
                    "🎯 Evidence Details",
                    "💾 Cypher Query",
                    "🕸️ Graph Visualization"
                ])
                
                with ctx_tab1:
                    st.markdown("### Query Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    llm_ctx = st.session_state.llm_context
                    entities = llm_ctx.get("entities", {})
                    
                    col1.metric("Intent", llm_ctx.get("intent", "Unknown"))
                    col2.metric("Evidence Count", len(llm_ctx.get("evidence", [])))
                    col3.metric("Retrieval Mode", RETRIEVAL_MODES[retrieval_mode_name])
                    
                    st.markdown("### Extracted Entities")
                    ent_cols = st.columns(4)
                    ent_cols[0].write(f"**Players:** {', '.join(entities.get('players', [])) or 'None'}")
                    ent_cols[1].write(f"**Teams:** {', '.join(entities.get('teams', [])) or 'None'}")
                    ent_cols[2].write(f"**Season:** {entities.get('season', 'N/A')}")
                    ent_cols[3].write(f"**Gameweek:** {entities.get('gameweek', 'N/A')}")
                    
                    # Show formatted context
                    st.markdown("### Formatted Context for LLM")
                    context_text = list(st.session_state.results.values())[0].get("context_text", "")
                    st.text_area("Context", context_text, height=300)
                
                with ctx_tab2:
                    evidence = st.session_state.llm_context.get("evidence", [])
                    display_evidence_details(evidence)
                
                with ctx_tab3:
                    st.markdown("### Executed Cypher Query")
                    baseline = st.session_state.pipeline_payload.get("baseline", {})
                    cypher_info = baseline.get("cypher", {})
                    
                    st.markdown("**Query:**")
                    st.code(cypher_info.get("query", "N/A"), language="cypher")
                    
                    st.markdown("**Parameters:**")
                    st.json(cypher_info.get("params", {}))
                    
                    st.metric("Records Returned", cypher_info.get("returned_records", 0))
                
                with ctx_tab4:
                    evidence = st.session_state.llm_context.get("evidence", [])
                    fig = visualize_graph(evidence)
                    
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("No graph data available for visualization")
    
    with tab2:
        st.markdown("## 📊 System Analytics")
        
        if st.session_state.results and len(st.session_state.results) > 1:
            st.markdown("### Model Performance Comparison")
            
            # Create comparison dataframe
            comp_data = []
            for model_name, result in st.session_state.results.items():
                if "error" not in result:
                    meta = result["meta"]
                    comp_data.append({
                        "Model": model_name,
                        "Latency (s)": round(meta["latency"], 2),
                        "Total Tokens": meta["total_tokens"],
                        "Prompt Tokens": meta["prompt_tokens"],
                        "Completion Tokens": meta["completion_tokens"],
                    })
            
            if comp_data:
                df = pd.DataFrame(comp_data)
                st.dataframe(df,width="stretch")
                
                # Visualize latency
                fig = go.Figure(data=[
                    go.Bar(x=df["Model"], y=df["Latency (s)"], marker_color='#37003c')
                ])
                fig.update_layout(
                    title="Model Latency Comparison",
                    xaxis_title="Model",
                    yaxis_title="Latency (seconds)",
                    height=400
                )
                st.plotly_chart(fig, width="stretch")
                
                # Token usage
                fig2 = go.Figure(data=[
                    go.Bar(name='Prompt', x=df["Model"], y=df["Prompt Tokens"], marker_color='#00ff87'),
                    go.Bar(name='Completion', x=df["Model"], y=df["Completion Tokens"], marker_color='#37003c')
                ])
                fig2.update_layout(
                    title="Token Usage by Model",
                    xaxis_title="Model",
                    yaxis_title="Tokens",
                    barmode='stack',
                    height=400
                )
                st.plotly_chart(fig2,width="stretch")
        else:
            st.info("Run a query with 'Compare All Models' enabled to see analytics")
    
    with tab3:
        st.markdown("## ℹ️ About FPL Graph-RAG")
        
        st.markdown("""
        ### What is Graph-RAG?
        
        Graph-RAG (Retrieval-Augmented Generation) combines the power of:
        - **Knowledge Graphs** (Neo4j) for structured data storage and retrieval
        - **Vector Embeddings** for semantic similarity search
        - **Large Language Models** for natural language understanding and generation
        
        ### How It Works
        
        1. **Input Processing**: Your query is analyzed to extract intent and entities
        2. **Baseline Retrieval**: Cypher queries fetch exact matches from the knowledge graph
        3. **Semantic Retrieval**: Embedding models find semantically similar information
        4. **Context Building**: Both retrieval methods are combined into unified context
        5. **LLM Generation**: The language model generates an answer grounded in the KG data
        
        ### Retrieval Modes
        
        - **Baseline**: Uses only Cypher queries for deterministic, exact matching
        - **Hybrid**: Combines Cypher with semantic embeddings for broader context
        
        ### Available Models
        
        - **DeepSeek**: High-performance reasoning model
        - **Llama 70B**: Open-source large language model
        - **Nova 2 Lite**: Fast and efficient model
        
        ### Data Source
        
        This system uses Fantasy Premier League data including:
        - Player performance statistics
        - Team information
        - Fixture data
        - Gameweek-by-gameweek analysis
        """)
        
        st.divider()
        


if __name__ == "__main__":
    load_dotenv()
    main()