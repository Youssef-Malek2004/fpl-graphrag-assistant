from typing import Dict, Any, List

def format_fpl_context_from_evidence(
    llm_context: Dict[str, Any],
    selected_evidence: List[Dict[str, Any]],
) -> str:
    """
    Turns llm_context + filtered evidence into readable LLM grounding text.
    Cypher evidence is authoritative. Semantic evidence is contextual.
    """

    intent = llm_context.get("intent")
    entities = llm_context.get("entities", {})
    season = entities.get("season")
    gw = entities.get("gameweek")
    players = entities.get("players", [])

    lines = [
        "Context from the Fantasy Premier League Knowledge Graph.",
        f"Intent: {intent}",
        f"Entities: players={players}, season={season}, gameweek={gw}",
        "",
        "EVIDENCE (Cypher is authoritative; Semantic is contextual only):",
        ""
    ]

    if not selected_evidence:
        lines.append("NO_EVIDENCE_FOUND")
        return "\n".join(lines)

    # Split for readability
    cypher_items = [e for e in selected_evidence if e.get("source") == "cypher"]
    semantic_items = [e for e in selected_evidence if e.get("source") == "semantic"]

    # 1) Cypher section (truth)
    if cypher_items:
        lines.append("== CYPHER (Baseline) ==")
        for e in cypher_items:
            item = e.get("item", {})
            kind = item.get("kind", "unknown")
            p = item.get("player", {})
            
            # ✅ Handle different item kinds
            if kind == "player_season":
                # Season aggregation - use "aggregates" field
                agg = item.get("aggregates", {})
                lines.append(
                    f"- Player: {p.get('player_name')} | Season: {item.get('season')}"
                )
                lines.append(
                    f"  Season Totals: minutes={agg.get('total_minutes')}, "
                    f"points={agg.get('total_points')}, "
                    f"goals={agg.get('total_goals')}, "
                    f"assists={agg.get('total_assists')}, "
                    f"clean_sheets={agg.get('total_clean_sheets')}"
                )
                lines.append(f"  Confidence: {e.get('confidence', 1.0)}")
                lines.append("")
                
            elif kind == "player_gameweek":
                # Gameweek-specific - use "stats" field
                fx = item.get("fixture", {})
                st = item.get("stats", {})
                ht = (item.get("home_team") or {}).get("name")
                at = (item.get("away_team") or {}).get("name")

                lines.append(
                    f"- Player: {p.get('player_name')} | Season: {item.get('season')} | GW: {item.get('gameweek')}"
                )
                lines.append(
                    f"  Fixture: {ht} vs {at} | Kickoff: {fx.get('kickoff_time')} | Score: {fx.get('team_h_score')}-{fx.get('team_a_score')}"
                )
                lines.append(
                    f"  Stats: minutes={st.get('minutes')}, points={st.get('total_points')}, "
                    f"goals={st.get('goals_scored')}, assists={st.get('assists')}, bonus={st.get('bonus')}"
                )
                lines.append(f"  Confidence: {e.get('confidence', 1.0)}")
                lines.append("")
                
            else:
                # Generic fallback for other kinds (team queries, recommendations, etc.)
                lines.append(f"- Kind: {kind} | Player: {p.get('player_name', 'N/A')}")
                lines.append(f"  Season: {item.get('season', 'N/A')} | GW: {item.get('gameweek', 'N/A')}")
                
                # Try both stats and aggregates
                if "stats" in item:
                    st = item.get("stats", {})
                    lines.append(
                        f"  Stats: minutes={st.get('minutes')}, points={st.get('total_points')}, "
                        f"goals={st.get('goals_scored')}, assists={st.get('assists')}"
                    )
                elif "aggregates" in item:
                    agg = item.get("aggregates", {})
                    lines.append(f"  Aggregates: {agg}")
                
                lines.append(f"  Confidence: {e.get('confidence', 1.0)}")
                lines.append("")

    # 2) Semantic section (neighbors)
    if semantic_items:
        lines.append("== SEMANTIC (Embeddings neighbors - contextual) ==")
        for e in semantic_items:
            item = e.get("item", {})
            p = (item.get("player") or {})
            st = (item.get("stats") or {})
            ht = (item.get("home_team") or {}).get("name")
            at = (item.get("away_team") or {}).get("name")
            score = (e.get("semantic_match") or {}).get("score", e.get("confidence"))

            lines.append(
                f"- Neighbor: {p.get('player_name')} | Season: {item.get('season')} | "
                f"GW: {item.get('gameweek')} | SemanticScore: {score}"
            )
            lines.append(
                f"  Fixture: {ht} vs {at} | Stats: minutes={st.get('minutes')}, "
                f"points={st.get('total_points')}"
            )
            lines.append("")

    return "\n".join(lines)