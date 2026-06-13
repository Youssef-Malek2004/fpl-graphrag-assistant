import pandas as pd
from neo4j import GraphDatabase

BATCH_SIZE = 1000

def create_constraints(session):
    constraints = [
        # Season unique
        """
        CREATE CONSTRAINT season_unique IF NOT EXISTS
        FOR (s:Season)
        REQUIRE s.season_name IS UNIQUE
        """,

        # Gameweek unique
        """
        CREATE CONSTRAINT gameweek_unique IF NOT EXISTS
        FOR (gw:Gameweek)
        REQUIRE (gw.season, gw.GW_number) IS UNIQUE
        """,

        # Fixture unique
        """
        CREATE CONSTRAINT fixture_unique IF NOT EXISTS
        FOR (f:Fixture)
        REQUIRE (f.season, f.fixture_number) IS UNIQUE
        """,

        # Team unique
        """
        CREATE CONSTRAINT team_unique IF NOT EXISTS
        FOR (t:Team)
        REQUIRE t.name IS UNIQUE
        """,

        # Player unique
        """
        CREATE CONSTRAINT player_unique IF NOT EXISTS
        FOR (p:Player)
        REQUIRE (p.player_name, p.player_element) IS UNIQUE
        """,

        # Position unique
        """
        CREATE CONSTRAINT position_unique IF NOT EXISTS
        FOR (pos:Position)
        REQUIRE pos.name IS UNIQUE
        """
    ]

    for c in constraints:
        session.run(c)
    print("Constraints created (or already existed).")



def load_config(path="config.txt"):
    config = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            config[key] = value
    return config


def load_fpl_batch(tx, rows):
    """
    rows: list[dict]
    Uses UNWIND to insert many records in a single transaction.
    """
    query = """
    UNWIND $rows AS row

    MERGE (s:Season {season_name: row.season})

    MERGE (gw:Gameweek {season: row.season, GW_number: row.GW})
    MERGE (s)-[:HAS_GW]->(gw)

    MERGE (f:Fixture {season: row.season, fixture_number: row.fixture})
    ON CREATE SET
        f.kickoff_time  = row.kickoff_time,
        f.team_a_score  = row.team_a_score,
        f.team_h_score  = row.team_h_score
    MERGE (gw)-[:HAS_FIXTURE]->(f)

    MERGE (home:Team {name: row.home_team})
    MERGE (away:Team {name: row.away_team})
    MERGE (f)-[:HAS_HOME_TEAM]->(home)
    MERGE (f)-[:HAS_AWAY_TEAM]->(away)

    MERGE (p:Player {player_name: row.name, player_element: row.element})
    MERGE (pos:Position {name: row.position})
    MERGE (p)-[:PLAYS_AS]->(pos)

    MERGE (p)-[pi:PLAYED_IN]->(f)
    SET pi.minutes          = row.minutes,
        pi.goals_scored     = row.goals_scored,
        pi.assists          = row.assists,
        pi.total_points     = row.total_points,
        pi.bonus            = row.bonus,
        pi.clean_sheets     = row.clean_sheets,
        pi.goals_conceded   = row.goals_conceded,
        pi.own_goals        = row.own_goals,
        pi.penalties_saved  = row.penalties_saved,
        pi.penalties_missed = row.penalties_missed,
        pi.yellow_cards     = row.yellow_cards,
        pi.red_cards        = row.red_cards,
        pi.saves            = row.saves,
        pi.bps              = row.bps,
        pi.influence        = row.influence,
        pi.creativity       = row.creativity,
        pi.threat           = row.threat,
        pi.ict_index        = row.ict_index,
        pi.form             = row.form
    """

    tx.run(query, rows=rows)


def main():
    cfg = load_config()
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))

    df = pd.read_csv("fpl_two_seasons.csv")

    df = df.fillna(0)

    records = []
    for _, r in df.iterrows():
        records.append({
            "season": r["season"],
            "GW": int(r["GW"]),
            "fixture": int(r["fixture"]),
            "kickoff_time": str(r["kickoff_time"]),
            "team_a_score": int(r["team_a_score"]),
            "team_h_score": int(r["team_h_score"]),
            "home_team": r["home_team"],
            "away_team": r["away_team"],

            "name": r["name"],
            "element": int(r["element"]),
            "position": r["position"],

            "minutes": int(r["minutes"]),
            "goals_scored": int(r["goals_scored"]),
            "assists": int(r["assists"]),
            "total_points": int(r["total_points"]),
            "bonus": int(r["bonus"]),
            "clean_sheets": int(r["clean_sheets"]),
            "goals_conceded": int(r["goals_conceded"]),
            "own_goals": int(r["own_goals"]),
            "penalties_saved": int(r["penalties_saved"]),
            "penalties_missed": int(r["penalties_missed"]),
            "yellow_cards": int(r["yellow_cards"]),
            "red_cards": int(r["red_cards"]),
            "saves": int(r["saves"]),
            "bps": int(r["bps"]),
            "influence": float(r["influence"]),
            "creativity": float(r["creativity"]),
            "threat": float(r["threat"]),
            "ict_index": float(r["ict_index"]),
            "form": float(r["form"]),
        })

    with driver.session() as session:
        create_constraints(session)

        total = len(records)
        for i in range(0, total, BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            session.execute_write(load_fpl_batch, batch)
            print(f"Inserted {min(i + BATCH_SIZE, total)}/{total} rows")

    driver.close()
    print("KG Construction Complete!")



if __name__ == "__main__":
    main()
