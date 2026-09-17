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

        acted_in_cypher = """
        UNWIND $rows AS row
        MERGE (p:Person {id: row.person_id})
        SET p.name = row.name, p.birth_year = row.birth_year, p.gender = row.gender
        MERGE (m:Movie {id: row.movie_id})
        SET m.title = row.movie_title
        MERGE (p)-[:ACTED_IN]->(m)
        """
        acted_in_rows = [
            {"person_id": 1, "name": "Tom Hanks", "birth_year": 1956, "gender": "M", "movie_id": 101, "movie_title": "Forrest Gump"},
            {"person_id": 1, "name": "Tom Hanks", "birth_year": 1956, "gender": "M", "movie_id": 102, "movie_title": "Cast Away"},
            {"person_id": 2, "name": "Robin Wright", "birth_year": 1966, "gender": "F", "movie_id": 101, "movie_title": "Forrest Gump"},
            {"person_id": 3, "name": "Tom Cruise", "birth_year": 1962, "gender": "M", "movie_id": 103, "movie_title": "Top Gun"},
        ]
        _, acted_in_summary, _ = driver.execute_query(acted_in_cypher, rows=acted_in_rows)
        print("ACTED_IN — nodes created:", acted_in_summary.counters.nodes_created)
        print("ACTED_IN — relationships created:", acted_in_summary.counters.relationships_created)

        directed_cypher = """
        UNWIND $rows AS row
        MERGE (p:Person {id: row.person_id})
        SET p.name = row.name, p.birth_year = row.birth_year, p.gender = row.gender
        MERGE (m:Movie {id: row.movie_id})
        SET m.title = row.movie_title
        MERGE (p)-[:DIRECTED]->(m)
        """
        directed_rows = [
            {"person_id": 4, "name": "Robert Zemeckis", "birth_year": 1951, "gender": "M", "movie_id": 101, "movie_title": "Forrest Gump"},
            {"person_id": 4, "name": "Robert Zemeckis", "birth_year": 1951, "gender": "M", "movie_id": 102, "movie_title": "Cast Away"},
            {"person_id": 5, "name": "Tony Scott", "birth_year": 1944, "gender": "M", "movie_id": 103, "movie_title": "Top Gun"},
        ]
        _, directed_summary, _ = driver.execute_query(directed_cypher, rows=directed_rows)
        print("DIRECTED — nodes created:", directed_summary.counters.nodes_created)
        print("DIRECTED — relationships created:", directed_summary.counters.relationships_created)

        read_cypher = """
        MATCH (p:Person)-[r]->(m:Movie)
        RETURN p.name AS person, type(r) AS relationship, m.title AS movie
        ORDER BY m.title, relationship
        """
        records, _, _ = driver.execute_query(read_cypher)
        for record in records:
            print(record["person"], "-", record["relationship"], "->", record["movie"])
    finally:
        driver.close()


if __name__ == "__main__":
    main()
