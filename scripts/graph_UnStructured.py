# This script turns one plain-text file into a small knowledge graph in Neo4j.
#
# The steps are:
#   1. Read the file's text.
#   2. Split the text into smaller chunks.
#   3. Save the file (as a Document node) and its chunks (as Chunk nodes) in Neo4j.
#   4. Turn each chunk's text into a vector ("embedding") so we can search by meaning later.
#   5. Create a vector index, so that search is fast.
#   6. Ask Claude to read each chunk and pull out entities (things it mentions),
#      then save those as nodes in Neo4j too.

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

# Load the secrets (passwords, API keys) from the .env file into environment variables.
load_dotenv()
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
HUGGINGFACE_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# The file we're turning into a graph.
FILE_PATH = "data/asciidoc/courses/30-days/modules/1-introduction/lessons/day-1-introduction/lesson.adoc"

# How the text gets split into chunks, and how big each chunk's embedding vector is.
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_SIZE = 384  # must match the embedding model below


# --- Step 1: read the file ---

with open(FILE_PATH, encoding="utf-8") as file:
    full_text = file.read()

print(f"Read {len(full_text)} characters from {FILE_PATH}")


# --- Step 2: split the text into chunks ---

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunk_texts = splitter.split_text(full_text)

print(f"Split into {len(chunk_texts)} chunks")


# --- Set up the two AI clients we'll use later (embeddings + entity extraction) ---

huggingface_client = InferenceClient(token=HUGGINGFACE_API_TOKEN)
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)


def get_embedding(text):
    # Turns text into a list of numbers (a vector) that represents its meaning.
    vector = huggingface_client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
    return vector.tolist()


def get_entities(text):
    # Asks Claude to read the text and return the entities it finds, as JSON,
    # so we can save them into Neo4j.
    prompt = f"""Read the text below and extract the important entities mentioned
(concepts, tools, people, places, etc.).

Respond with ONLY JSON in exactly this shape, nothing else:
{{
  "entities": [{{"name": "...", "type": "..."}}]
}}

Text:
{text}
"""
    response = claude_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    # Claude's reply can include a "thinking" block before its actual answer,
    # but the text answer always comes last, so grab the last item.
    answer = response.content[-1].text.strip()

    # Claude sometimes wraps its answer in a ```json ... ``` code fence even when told
    # not to. Remove that if it's there, so we're left with just the JSON.
    if answer.startswith("```"):
        answer = answer.split("\n", 1)[1]
        answer = answer.rsplit("```", 1)[0]

    return json.loads(answer)


# --- Connect to Neo4j and do the rest of the work ---

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
try:
    driver.verify_connectivity()
    print("Connected to Neo4j.")

    # --- Step 3: save the Document and its Chunks ---

    chunk_rows = []
    for i, chunk_text in enumerate(chunk_texts):
        chunk_rows.append({"index": i, "text": chunk_text})

    driver.execute_query(
        """
        MERGE (d:Document {path: $path})
        WITH d
        UNWIND $chunks AS row
        MERGE (c:Chunk {path: $path, index: row.index})
        SET c.text = row.text
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        path=FILE_PATH,
        chunks=chunk_rows,
    )
    print(f"Saved 1 Document and {len(chunk_rows)} Chunks.")

    # --- Step 4: embed each chunk and save its vector ---

    embedding_rows = []
    for row in chunk_rows:
        print(f"Embedding chunk {row['index']}...")
        embedding_rows.append({"index": row["index"], "embedding": get_embedding(row["text"])})

    driver.execute_query(
        """
        UNWIND $chunks AS row
        MATCH (c:Chunk {path: $path, index: row.index})
        SET c.embedding = row.embedding
        """,
        path=FILE_PATH,
        chunks=embedding_rows,
    )
    print("Saved embeddings.")

    # --- Step 5: create the vector index (safe to run every time) ---

    driver.execute_query(f"""
        CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {EMBEDDING_SIZE},
          `vector.similarity_function`: 'cosine'
        }}}}
    """)
    print("Vector index ready.")

    # --- Step 6: extract entities from each chunk, and save them ---

    entity_rows = []
    for row in chunk_rows:
        print(f"Extracting entities from chunk {row['index']}...")
        result = get_entities(row["text"])

        for entity in result["entities"]:
            entity_rows.append({
                "name": entity["name"],
                "type": entity["type"].lower(),  # lowercase so "Tool" and "tool" don't become two different nodes
                "chunk_index": row["index"],
            })

    # Save each entity, linked back to the chunk it was found in.
    driver.execute_query(
        """
        UNWIND $entities AS row
        MERGE (e:Entity {name: row.name, type: row.type})
        WITH e, row
        MATCH (c:Chunk {path: $path, index: row.chunk_index})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """,
        path=FILE_PATH,
        entities=entity_rows,
    )
    print(f"Saved {len(entity_rows)} entities.")
finally:
    driver.close()
