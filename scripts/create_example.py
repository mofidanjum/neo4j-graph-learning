import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.environ["NEO4J_URI"]
USERNAME = os.environ["NEO4J_USERNAME"]
PASSWORD = os.environ["NEO4J_PASSWORD"]


def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j successfully.")

        create_cypher = """
        MERGE (p:Person {name: "Tom Hanks"})
        MERGE (m:Movie {title: "Forrest Gump"})
        MERGE (p)-[r:ACTED_IN {role: "Forrest"}]->(m)
        RETURN p, r, m
        """
        _, summary, _ = driver.execute_query(create_cypher)
        print("Nodes created:", summary.counters.nodes_created)
        print("Relationships created:", summary.counters.relationships_created)

        read_cypher = """
        MATCH (p:Person {name: "Tom Hanks"})-[r:ACTED_IN]->(m:Movie)
        RETURN m.title AS title, r.role AS role
        """
        records, _, _ = driver.execute_query(read_cypher)
        for record in records:
            print(record["title"], "-", record["role"])
    finally:
        driver.close()


if __name__ == "__main__":
    main()
