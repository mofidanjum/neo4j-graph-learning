# Neo4j Graph Learning Sandbox

A small, self-contained project for learning Neo4j and the Cypher query language.

## 1. Get a Neo4j instance running

Pick one:

- **Neo4j Aura Free (easiest, no install)** — create a free cloud instance at https://console.neo4j.io. Note the connection URI, username, and password it gives you.
- **Neo4j Desktop** — install from https://neo4j.com/download, create a local database, and start it.
- **Docker** (if you have Docker Desktop installed):
  ```
  docker compose up -d
  ```
  This starts Neo4j at `bolt://localhost:7687` with browser UI at http://localhost:7474 (user: `neo4j`, password: `learn1234`).

## 2. Configure connection details

Copy `.env.example` to `.env` and fill in your URI/user/password:

```
cp .env.example .env
```

## 3. Set up Python environment

```
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## 4. Run the lessons in order

```
python scripts/01_connect.py
python scripts/02_create_sample_graph.py
python scripts/03_basic_queries.py
python scripts/04_relationships_and_traversal.py
```

Each script is commented to explain the Cypher concept it demonstrates:

1. **01_connect.py** — verify the driver can reach your database.
2. **02_create_sample_graph.py** — build a small movie/actor graph with `CREATE` and `MERGE`.
3. **03_basic_queries.py** — `MATCH`, `WHERE`, `RETURN`, filtering and sorting.
4. **04_relationships_and_traversal.py** — pattern matching across relationships, variable-length paths, aggregation.

## 5. Explore visually

Open the Neo4j Browser (Aura console, Neo4j Desktop, or http://localhost:7474 for Docker) and try:

```cypher
MATCH (n) RETURN n LIMIT 100
```

to see the graph you built.
