# cypher_manual_queries_builder.py
from typing import Dict, Any, List, Tuple, Optional


# ============================================================================
# GENERAL INTENTS (5 main categories)
# ============================================================================
# 1. PLAYER_PERFORMANCE_ANALYSIS
# 2. TEAM_ANALYSIS
# 3. RECOMMENDATION
# 4. PLAYER_COMPARISON
# 5. AGGREGATION


def _first_position(entities: Dict[str, Any]) -> Optional[str]:
    if entities.get("position"):
        return entities["position"]
    if isinstance(entities.get("positions"), list) and entities["positions"]:
        return entities["positions"][0]
    return None


def _metric_field(metric: Optional[str]) -> str:
    metric_map = {
        "goals_scored": "goals_scored",
        "assists": "assists",
        "clean_sheets": "clean_sheets",
        "saves": "saves",
        "bonus": "bonus",
        "yellow_cards": "yellow_cards",
        "red_cards": "red_cards",
        "minutes": "minutes",
        "points": "total_points",
        "total_points": "total_points",
    }
    if not metric:
        return "total_points"
    return metric_map.get(metric.lower(), "total_points")


def build_cypher(intent: str, entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Build Cypher query based on general intent and extracted entities.
    All queries MUST return a single column named `item` (a map) for stability.
    """
    intent = (intent or "").upper()

    if intent == "PLAYER_PERFORMANCE_ANALYSIS":
        return _build_player_performance_query(entities)
    elif intent == "TEAM_ANALYSIS":
        return _build_team_analysis_query(entities)
    elif intent == "RECOMMENDATION":
        return _build_recommendation_query(entities)
    elif intent == "PLAYER_COMPARISON":
        return _build_comparison_query(entities)
    elif intent == "AGGREGATION":
        return _build_aggregation_query(entities)
    else:
        raise ValueError(f"Unknown intent: {intent}")


# ============================================================================
# INTENT 1: PLAYER_PERFORMANCE_ANALYSIS
# ============================================================================
def _build_player_performance_query(entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    players = entities.get("players", []) or []
    season = entities.get("season")
    season_alt = entities.get("season_alt", season)
    gameweek = entities.get("gameweek")
    position = _first_position(entities)
    metric = entities.get("metric")

    if len(players) == 1 and gameweek is not None and season:
        return _player_gameweek_stats(players[0], int(gameweek), season, season_alt)

    if len(players) == 1 and season and gameweek is None:
        return _player_season_stats(players[0], season, season_alt)

    if position and season and metric and gameweek is None:
        return _top_by_metric_position_season(position, season, season_alt, metric, int(entities.get("limit", 10)))

    if position and season and not metric and gameweek is None:
        return _top_players_by_position_season(position, season, season_alt, int(entities.get("limit", 10)))

    limit = entities.get("limit")
    limit = int(limit) if limit not in (None, "") else 10

    if (gameweek is not None) and season and (not players) and (not position):
        return _top_points_gameweek(int(gameweek), season, season_alt, limit)

    raise ValueError(f"Cannot determine specific query for PLAYER_PERFORMANCE_ANALYSIS with entities: {entities}")


def _player_gameweek_stats(player_name: str, gameweek: int, season: str, season_alt: str) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (p:Player)
    WHERE toLower(p.player_name) = toLower($player_name)

    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]

    MATCH (s)-[:HAS_GW]->(gw:Gameweek {GW_number: $gameweek})
    MATCH (gw)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)

    MATCH (p)-[pi:PLAYED_IN]->(f)

    RETURN {
      kind: "player_gameweek",
      intent: "PLAYER_PERFORMANCE_ANALYSIS",
      player: p { .player_name, .player_element },
      season: s.season_name,
      gameweek: gw.GW_number,
      fixture: f { .fixture_number, .kickoff_time, .team_h_score, .team_a_score },
      home_team: home { .name },
      away_team: away { .name },
      stats: pi { .minutes, .total_points, .goals_scored, .assists, .bonus }
    } AS item
    """
    return query, {"player_name": player_name, "season": season, "season_alt": season_alt, "gameweek": gameweek}


def _player_season_stats(player_name: str, season: str, season_alt: str) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (p:Player)
    WHERE toLower(p.player_name) = toLower($player_name)

    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]

    MATCH (s)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p)-[pi:PLAYED_IN]->(f)

    WITH p, s,
         sum(coalesce(pi.total_points, 0)) AS total_points,
         sum(coalesce(pi.goals_scored, 0)) AS total_goals,
         sum(coalesce(pi.assists, 0)) AS total_assists,
         sum(coalesce(pi.clean_sheets, 0)) AS total_clean_sheets,
         sum(coalesce(pi.minutes, 0)) AS total_minutes

    RETURN {
      kind: "player_season",
      intent: "PLAYER_PERFORMANCE_ANALYSIS",
      player: p { .player_name, .player_element },
      season: s.season_name,
      aggregates: {
        total_points: total_points,
        total_goals: total_goals,
        total_assists: total_assists,
        total_clean_sheets: total_clean_sheets,
        total_minutes: total_minutes
      }
    } AS item
    """
    return query, {"player_name": player_name, "season": season, "season_alt": season_alt}


def _top_players_by_position_season(position: str, season: str, season_alt: str, limit: int) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    WHERE toLower(pos.name) = toLower($position)

    MATCH (p)-[pi:PLAYED_IN]->(f)
    WITH p, s, sum(coalesce(pi.total_points, 0)) AS total_points

    RETURN {
      kind: "top_players_position_season",
      intent: "PLAYER_PERFORMANCE_ANALYSIS",
      player: p { .player_name, .player_element },
      season: s.season_name,
      position: $position,
      aggregates: { total_points: total_points }
    } AS item
    ORDER BY total_points DESC
    LIMIT $limit
    """
    return query, {"position": position, "season": season, "season_alt": season_alt, "limit": limit}


def _top_by_metric_position_season(
    position: str,
    season: str,
    season_alt: str,
    metric: str,
    limit: int
) -> Tuple[str, Dict[str, Any]]:

    field = _metric_field(metric)

    query = f"""
    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]

    MATCH (s)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    WHERE toLower(pos.name) = toLower($position)

    MATCH (p)-[pi:PLAYED_IN]->(f)

    WITH p, s, sum(coalesce(pi.{field}, 0)) AS metric_total

    RETURN {{
      kind: "top_players_metric_position_season",
      intent: "PLAYER_PERFORMANCE_ANALYSIS",
      player: p {{ .player_name, .player_element }},
      season: s.season_name,
      position: $position,
      metric: $metric,
      aggregates: {{
        metric_name: $metric,
        metric_total: metric_total
      }}
    }} AS item
    ORDER BY metric_total DESC
    LIMIT $limit
    """

    return query, {
        "position": position,
        "season": season,
        "season_alt": season_alt,
        "metric": metric,
        "limit": limit
    }




def _top_points_gameweek(gameweek: int, season: str, season_alt: str, limit: int) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(gw:Gameweek {GW_number: $gameweek})
    MATCH (gw)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (p:Player)-[pi:PLAYED_IN]->(f)

    RETURN {
      kind: "top_players_gameweek",
      intent: "PLAYER_PERFORMANCE_ANALYSIS",
      player: p { .player_name, .player_element },
      season: s.season_name,
      gameweek: gw.GW_number,
      fixture: f { .fixture_number, .kickoff_time, .team_h_score, .team_a_score },
      stats: pi { .minutes, .total_points, .goals_scored, .assists, .bonus }
    } AS item
    ORDER BY coalesce(pi.total_points, 0) DESC
    LIMIT $limit
    """
    return query, {"season": season, "season_alt": season_alt, "gameweek": gameweek, "limit": limit}


# ============================================================================
# INTENT 2: TEAM_ANALYSIS
# ============================================================================
def _build_team_analysis_query(entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    teams = entities.get("teams", []) or []
    season = entities.get("season")
    season_alt = entities.get("season_alt", season)
    gameweek = entities.get("gameweek")
    position = _first_position(entities)

    if not teams:
        raise ValueError("TEAM_ANALYSIS requires at least one team")

    if len(teams) >= 2 and season:
        return _compare_teams_season(teams, season, season_alt)

    team = teams[0]

    if season and (gameweek is None):
        return _team_player_ranking_season(team, season, season_alt, int(entities.get("limit", 10)))

    if season and (gameweek is not None) and position:
        return _team_position_top_gameweek(team, position, int(gameweek), season, season_alt, int(entities.get("limit", 1)))


    raise ValueError(f"Cannot determine specific query for TEAM_ANALYSIS with entities: {entities}")


def _team_player_ranking_season(team: str, season: str, season_alt: str, limit: int) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (t:Team)
    WHERE toLower(t.name) = toLower($team)

    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
    MATCH (p:Player)-[pi:PLAYED_IN]->(f)

    WITH p, t, s, sum(coalesce(pi.total_points, 0)) AS total_points

    RETURN {
      kind: "team_top_players_season",
      intent: "TEAM_ANALYSIS",
      team: t { .name },
      player: p { .player_name, .player_element },
      season: s.season_name,
      aggregates: { total_points: total_points }
    } AS item
    ORDER BY total_points DESC
    LIMIT $limit
    """
    return query, {"team": team, "season": season, "season_alt": season_alt, "limit": limit}


def _team_position_top_gameweek(team: str, position: str, gameweek: int, season: str, season_alt: str, limit: int) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (t:Team)
    WHERE toLower(t.name) = toLower($team)

    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(gw:Gameweek {GW_number: $gameweek})
    MATCH (gw)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)

    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    WHERE toLower(pos.name) = toLower($position)

    MATCH (p)-[pi:PLAYED_IN]->(f)

    RETURN {
      kind: "team_best_position_gameweek",
      intent: "TEAM_ANALYSIS",
      team: t { .name },
      player: p { .player_name, .player_element },
      season: s.season_name,
      gameweek: gw.GW_number,
      fixture: f { .fixture_number, .kickoff_time, .team_h_score, .team_a_score },
      position: $position,
      stats: pi { .minutes, .total_points, .goals_scored, .assists, .bonus }
    } AS item
    ORDER BY coalesce(pi.total_points, 0) DESC
    LIMIT $limit
    """
    return query, {"team": team, "position": position, "season": season, "season_alt": season_alt, "gameweek": gameweek, "limit": limit}


# ============================================================================
# INTENT 3: RECOMMENDATION
# ============================================================================
def _build_recommendation_query(entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    position = _first_position(entities)
    season = entities.get("season")
    season_alt = entities.get("season_alt", season)
    gameweek = entities.get("gameweek")

    limit = entities.get("limit")
    limit = int(limit) if limit not in (None, "") else 10

    if season and (gameweek is not None) and position:
        return _recommend_position_for_gameweek(position, int(gameweek) - 1, season, season_alt, limit)


    raise ValueError(f"Cannot determine specific query for RECOMMENDATION with entities: {entities}")


def _recommend_position_for_gameweek(position: str, gameweek: int, season: str, season_alt: str, limit: int) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(gw:Gameweek {GW_number: $gameweek})
    MATCH (gw)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    WHERE toLower(pos.name) = toLower($position)

    MATCH (p)-[pi:PLAYED_IN]->(f)

    RETURN {
      kind: "recommend_position_gameweek",
      intent: "RECOMMENDATION",
      season: s.season_name,
      gameweek: gw.GW_number,
      position: $position,
      player: p { .player_name, .player_element },
      fixture: f { .fixture_number, .kickoff_time, .team_h_score, .team_a_score },
      stats: pi { .minutes, .total_points, .goals_scored, .assists, .bonus }
    } AS item
    ORDER BY coalesce(pi.total_points, 0) DESC
    LIMIT $limit
    """
    return query, {"position": position, "season": season, "season_alt": season_alt, "gameweek": gameweek, "limit": limit}



# ============================================================================
# INTENT 4: PLAYER_COMPARISON
# ============================================================================
def _build_comparison_query(entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    players = entities.get("players", []) or []
    teams = entities.get("teams", []) or []
    season = entities.get("season")
    season_alt = entities.get("season_alt", season)
    gameweek = entities.get("gameweek")

    if len(players) >= 2 and season:
        if gameweek is not None:
            return _compare_players_gameweek(players, int(gameweek), season, season_alt)
        return _compare_players_season(players, season, season_alt)

    raise ValueError(f"Cannot determine specific query for PLAYER_COMPARISON with entities: {entities}")


def _compare_players_gameweek(players: List[str], gameweek: int, season: str, season_alt: str) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (p:Player)
    WHERE toLower(p.player_name) IN $player_names_lower

    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(gw:Gameweek {GW_number: $gameweek})
    MATCH (gw)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (p)-[pi:PLAYED_IN]->(f)

    RETURN {
      kind: "compare_players_gameweek",
      intent: "PLAYER_COMPARISON",
      season: s.season_name,
      gameweek: gw.GW_number,
      player: p { .player_name, .player_element },
      fixture: f { .fixture_number, .kickoff_time, .team_h_score, .team_a_score },
      stats: pi { .minutes, .total_points, .goals_scored, .assists, .bonus }
    } AS item
    ORDER BY coalesce(pi.total_points, 0) DESC
    """
    return query, {
        "player_names_lower": [p.lower() for p in players],
        "season": season,
        "season_alt": season_alt,
        "gameweek": gameweek,
    }


def _compare_players_season(players: List[str], season: str, season_alt: str) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (p:Player)
    WHERE toLower(p.player_name) IN $player_names_lower

    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (p)-[pi:PLAYED_IN]->(f)

    WITH p, s,
         sum(coalesce(pi.total_points, 0)) AS total_points,
         sum(coalesce(pi.goals_scored, 0)) AS total_goals,
         sum(coalesce(pi.assists, 0)) AS total_assists,
         sum(coalesce(pi.minutes, 0)) AS total_minutes

    RETURN {
      kind: "compare_players_season",
      intent: "PLAYER_COMPARISON",
      season: s.season_name,
      player: p { .player_name, .player_element },
      aggregates: {
        total_points: total_points,
        total_goals: total_goals,
        total_assists: total_assists,
        total_minutes: total_minutes
      }
    } AS item
    ORDER BY total_points DESC
    """
    return query, {
        "player_names_lower": [p.lower() for p in players],
        "season": season,
        "season_alt": season_alt,
    }


def _compare_teams_season(teams: List[str], season: str, season_alt: str) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (t:Team)
    WHERE toLower(t.name) IN $team_names_lower

    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
    MATCH (p:Player)-[pi:PLAYED_IN]->(f)

    WITH t, s, sum(coalesce(pi.total_points, 0)) AS total_team_points, count(DISTINCT p) AS player_count

    RETURN {
      kind: "compare_teams_season",
      intent: "PLAYER_COMPARISON",
      season: s.season_name,
      team: t { .name },
      aggregates: {
        total_team_points: total_team_points,
        player_count: player_count
      }
    } AS item
    ORDER BY total_team_points DESC
    """
    return query, {
        "team_names_lower": [t.lower() for t in teams],
        "season": season,
        "season_alt": season_alt,
    }


# ============================================================================
# INTENT 5: AGGREGATION
# ============================================================================
# Replace your _build_aggregation_query function in cypher_manual_queries_builder.py

def _build_aggregation_query(entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Build aggregation queries.
    Handles misclassified queries by redirecting to appropriate intent.
    """
    
    players = entities.get("players", [])
    
    # EDGE CASE 1: If specific players mentioned, this is PLAYER_PERFORMANCE, not AGGREGATION
    # E.g., "How many assists did Bukayo Saka make in gameweek 12?"
    if len(players) > 0:
        gameweek = entities.get("gameweek")
        season = entities.get("season")
        
        # Add default season if missing
        if not season:
            entities["season"] = "2023-24"
            entities["season_alt"] = "2023/24"
        
        return _build_player_performance_query(entities)
    
    # EDGE CASE 2: Multiple teams mentioned = TEAM_ANALYSIS
    teams = entities.get("teams", [])
    if len(teams) >= 2:
        season = entities.get("season")
        if not season:
            entities["season"] = "2023-24"
            entities["season_alt"] = "2023/24"
        return _compare_teams_season(teams, entities["season"], entities["season_alt"])
    
    # Continue with normal aggregation logic
    season = entities.get("season")
    season_alt = entities.get("season_alt", season)
    metric = entities.get("metric")
    position = _first_position(entities)

    # Default season if missing
    if not season:
        season = "2023-24"
        season_alt = "2023/24"

    agg_raw = entities.get("aggregate") or ""
    aggregate = agg_raw.strip().upper()

    limit = int(entities.get("limit") or 10)
    min_appearances = int(entities.get("min_appearances", entities.get("min_apps", 10)))

    metric_l = (metric or "").strip().lower()
    
    # Most consistent players (average points)
    if aggregate in {"AVG", "AVERAGE"} and metric_l in {"", "points", "total_points"} and position is None:
        return _most_consistent_players(season, season_alt, min_appearances, limit)

    # Average points by position
    if position and aggregate in {"AVG", "AVERAGE"}:
        return _average_points_by_position(position, season, season_alt)

    # Total metric for season
    if metric and (not position) and aggregate in {"SUM", "TOTAL", "OVERALL", "COUNT", ""}:
        return _total_metric_season(season, season_alt, metric)

    # If we got here and have a position, try top players by position
    if position:
        return _top_players_by_position_season(position, season, season_alt, limit)

    raise ValueError(
        f"Cannot determine specific query for AGGREGATION with entities: {entities}\n"
        f"Hint: AGGREGATION queries should not mention specific players. "
        f"If asking about a specific player, this should be PLAYER_PERFORMANCE_ANALYSIS."
    )


def _average_points_by_position(position: str, season: str, season_alt: str) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)

    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    WHERE toLower(pos.name) = toLower($position)

    MATCH (p)-[pi:PLAYED_IN]->(f)

    WITH s,
         avg(coalesce(pi.total_points, 0)) AS avg_points,
         count(pi) AS total_appearances

    RETURN {
      kind: "avg_points_by_position",
      intent: "AGGREGATION",
      season: s.season_name,
      position: $position,
      aggregates: { avg_points: avg_points, total_appearances: total_appearances }
    } AS item
    """
    return query, {"position": position, "season": season, "season_alt": season_alt}

def _most_consistent_players(
    season: str,
    season_alt: str,
    min_appearances: int,
    limit: int
) -> Tuple[str, Dict[str, Any]]:
    query = """
    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player)-[pi:PLAYED_IN]->(f)

    WITH s, p,
         count(pi) AS appearances,
         avg(toFloat(coalesce(pi.total_points, 0))) AS avg_points,
         sum(coalesce(pi.total_points, 0)) AS total_points

    WHERE appearances >= $min_appearances

    RETURN {
      kind: "most_consistent_players",
      intent: "AGGREGATION",
      season: s.season_name,
      min_appearances: $min_appearances,
      player: p { .player_name, .player_element },
      aggregates: {
        appearances: appearances,
        avg_points_per_appearance: avg_points,
        total_points: total_points
      }
    } AS item
    ORDER BY avg_points DESC, appearances DESC, total_points DESC
    LIMIT $limit
    """
    return query, {
        "season": season,
        "season_alt": season_alt,
        "min_appearances": int(min_appearances),
        "limit": int(limit),
    }



def _total_metric_season(season: str, season_alt: str, metric: str) -> Tuple[str, Dict[str, Any]]:
    field = _metric_field(metric)
    query = f"""
    MATCH (s:Season)
    WHERE s.season_name IN [$season, $season_alt]
    MATCH (s)-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player)-[pi:PLAYED_IN]->(f)

    WITH s,
         sum(coalesce(pi.{field}, 0)) AS total_metric,
         count(DISTINCT p) AS player_count

    RETURN {{
      kind: "total_metric_season",
      intent: "AGGREGATION",
      season: s.season_name,
      metric: $metric,
      aggregates: {{ total_metric: total_metric, player_count: player_count }}
    }} AS item
    """
    return query, {"season": season, "season_alt": season_alt, "metric": metric}
