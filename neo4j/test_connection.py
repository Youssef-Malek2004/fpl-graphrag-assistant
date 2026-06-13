from neo4j import GraphDatabase

def load_config(path="config.txt"):
    config = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split("=")
            config[key] = value
    return config

def test_query():
    cfg = load_config()
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))

    query = "RETURN 'Neo4j is connected successfully!' AS message"

    with driver.session() as session:
        result = session.run(query)
        for record in result:
            print(record["message"])

    driver.close()

if __name__ == "__main__":
    test_query()
