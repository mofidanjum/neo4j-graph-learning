import os

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.environ["NEO4J_URI"]
USERNAME = os.environ["NEO4J_USERNAME"]
PASSWORD = os.environ["NEO4J_PASSWORD"]
HUGGINGFACE_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

POSTER_EMBEDDING_DIMENSIONS = 512
HF_CLIP_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"


def poster_url_for(title):
    slug = title.lower().replace(" ", "-")
    return f"https://picsum.photos/seed/{slug}/300/450"


def embed_poster(poster_url):
    image_bytes = requests.get(poster_url, timeout=10).content
    response = requests.post(
        HF_CLIP_MODEL_URL,
        headers={"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"},
        data=image_bytes,
        timeout=30,
    )
    response.raise_for_status()
    embedding = response.json()
    # Some models return one vector per image patch; average them into a single vector.
    if isinstance(embedding[0], list):
        embedding = [sum(values) / len(values) for values in zip(*embedding)]
    return embedding


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

        movies = {row["movie_id"]: row["movie_title"] for row in acted_in_rows + directed_rows}

        poster_rows = []
        for movie_id, title in movies.items():
            poster_url = poster_url_for(title)
            print(f"Embedding poster for {title} via Hugging Face API...")
            embedding = embed_poster(poster_url)
            poster_rows.append({"movie_id": movie_id, "poster_url": poster_url, "embedding": embedding})

        poster_cypher = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        SET m.poster_url = row.poster_url, m.poster_embedding = row.embedding
        """
        driver.execute_query(poster_cypher, rows=poster_rows)
        print(f"Set poster_url and poster_embedding on {len(poster_rows)} movies.")

        driver.execute_query(f"""
        CREATE VECTOR INDEX movie_poster_embedding IF NOT EXISTS
        FOR (m:Movie) ON (m.poster_embedding)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {POSTER_EMBEDDING_DIMENSIONS},
          `vector.similarity_function`: 'cosine'
        }}}}
        """)
        print("Ensured vector index movie_poster_embedding exists.")

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
