from neo4j import GraphDatabase
from cypher_manual_queries_builder import build_cypher
from manual_parsers import parse_results_for_llm
import json


def load_config(path="../neo4j/config.txt"):
    config = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split("=")
            config[key] = value
    return config


def test_flexible_system():
    cfg = load_config()
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))

    # ========================================================================
    # 13 TEST CASES covering all intents and query variations
    # ========================================================================

    test_cases = [
        # ===== PLAYER_PERFORMANCE INTENT =====
        {
            "description": "Query 1: Single player gameweek stats",
            "intent": 'PLAYER_PERFORMANCE_ANALYSIS',
            "entities": {
                "players": ["Mohamed Salah"],
                "gameweek": 5,
                "season": "2021-22"
            }
        },
        {
            "description": "Query 2: Single player season aggregation",
            "intent": 'PLAYER_PERFORMANCE_ANALYSIS',
            "entities": {
                "players": ["Mohamed Salah"],
                "season": "2021-22"
            }
        },
        {
            "description": "Query 3: Top players by position (total points)",
            "intent": 'PLAYER_PERFORMANCE_ANALYSIS',
            "entities": {
                "position": "FWD",
                "season": "2021-22",
                "limit": 10,
            }
        },
        {
            "description": "Query 4: Top goal scorers by position",
            "intent": 'PLAYER_PERFORMANCE_ANALYSIS',
            "entities": {
                "position": "MID",
                "season": "2021-22",
                "metric": "goals",
                "limit": 10
            }
        },
        {
            "description": "Query 5: Top performers in a gameweek",
            "intent": 'PLAYER_PERFORMANCE_ANALYSIS',
            "entities": {
                "gameweek": 10,
                "season": "2021-22",
                "limit": 10
            }
        },

        # ===== COMPARISON INTENT =====
        {
            "description": "Query 6: Compare two players in a gameweek",
            "intent": 'PLAYER_COMPARISON',
            "entities": {
                "players": ["Mohamed Salah", "John Stones"],
                "gameweek": 5,
                "season": "2021-22"
            }
        },
        {
            "description": "Query 7: Compare two players across season",
            "intent": 'PLAYER_COMPARISON',
            "entities": {
                "players": ["Mohamed Salah", "John Stones"],
                "season": "2021-22"
            }
        },

        # ===== TEAM_ANALYSIS INTENT =====
        {
            "description": "Query 8: Compare two teams' performance",
            "intent": 'TEAM_ANALYSIS',
            "entities": {
                "teams": ["Liverpool", "Manchester City"],
                "season": "2021-22"
            }
        },

        {
            "description": "Query 9: Top players from a specific team",
            "intent": 'TEAM_ANALYSIS',
            "entities": {
                "teams": ["Liverpool"],
                "season": "2021-22",
                "limit": 5
            }
        },
        {
            "description": "Query 10: Team's best player by position in gameweek",
            "intent": 'TEAM_ANALYSIS',
            "entities": {
                "teams": ["Liverpool"],
                "position": "GK",
                "gameweek": 13,
                "season": "2021-22",
                "limit": 1
            }
        },

        # ===== RECOMMENDATION INTENT =====
        {
            "description": "Query 11: Recommend players for position in gameweek",
            "intent": 'RECOMMENDATION',
            "entities": {
                "position": "DEF",
                "gameweek": 10,
                "season": "2021-22",
                "limit": 5
            }
        },

        # ===== AGGREGATION INTENT =====
        {
            "description": "Query 12: average points for forwards in 2021-22",
            "intent": 'AGGREGATION',
            "entities": {
                "position": "MID",
                "season": "2021-22",
                "aggregate": "AVG"
            }
        },
        {
            "description": "Query 13: total goals in 2021-22",
            "intent": "AGGREGATION",
            "entities": {
                "season": "2021-22",
                "metric": "goals_scored",
                "aggregate": "SUM"
            }
        },
        {
            "description": "Query 14: Consistent average points for 5 players in 2021-22",
            "intent": "AGGREGATION",
            "entities": {
                "season": "2021-22",
                "aggregate": "AVG",
                "metric": "points",
                "min_appearances": 10,
                "limit": 3
            }
        }
    ]

    print(f"\n{'=' * 80}")
    print(f"TESTING FLEXIBLE INTENT SYSTEM WITH {len(test_cases)} QUERIES")
    print(f"{'=' * 80}\n")

    successful = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"{'=' * 80}")

        entities = test_case['entities']

        # Classify intent from entities
        intent = test_case['intent']
        print(f"Classified Intent: {intent}")
        print(f"Entities: {entities}")

        try:
            # Build query
            query, params = build_cypher(intent, entities)
            print(f"\nGenerated Cypher Query:")
            print(query)
            print(f"\nParameters: {params}")

            # Execute query
            with driver.session() as session:
                result = session.run(query, params)
                records = list(result)
                print(f"\n✓ SUCCESS: Returned {len(records)} records")
                successful += 1

                # Show first result (raw data)
                if records:
                    print("\nFirst result (sample):")
                    record_dict = dict(records[0])
                    # Print first few keys to avoid clutter
                    for key in list(record_dict.keys())[:3]:
                        print(f"  {key}: {record_dict[key]}")

                payload = parse_results_for_llm(intent, entities, records)
                json_str = json.dumps(payload, indent=2, ensure_ascii=False)
                print("\nParsed JSON for LLM:")
                print(json_str)

        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 80}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful / len(test_cases) * 100:.1f}%")

    driver.close()


if __name__ == "__main__":
    # Run all test cases
    test_flexible_system()