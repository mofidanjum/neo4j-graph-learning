import os

from dotenv import load_dotenv
from neo4j import GraphDatabase, Result

load_dotenv()

URI = os.environ["NEO4J_URI"]
USERNAME = os.environ["NEO4J_USERNAME"]
PASSWORD = os.environ["NEO4J_PASSWORD"]


def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j successfully.")

        count_records,_ ,_ = driver.execute_query(
            "RETURN COUNT {()} AS count"
        )
        print("Node count:", count_records[0]["count"])

        cypher = """
        MATCH (p:Patient {name: $patient_name})-[r:HAD_VISIT]->(v:Visit)
        RETURN p.name as name, r.id as id , v.type as type
        """
        records, summary, keys = driver.execute_query(
            cypher, patient_name="Mohammad578 Nikolaus26"
        )
        for record in records:
            print(record["name"], record["id"], record["type"])

        df = driver.execute_query(
            cypher,
            patient_name="Mohammad578 Nikolaus26",
            result_transformer_=Result.to_df,
        )
        print(df)

        single_create_cypher = """
        MERGE (p:Person {name: $actor})
        MERGE (m:Movie {title: $movie})
        MERGE (p)-[r:ACTED_IN {role: $role}]->(m)
        RETURN p, r, m
        """
        _, single_create_summary, _ = driver.execute_query(
            single_create_cypher,
            actor="Tom Hanks",
            movie="Forrest Gump",
            role="Forrest",
        )
        print("Nodes created (single):", single_create_summary.counters.nodes_created)
        print("Relationships created (single):", single_create_summary.counters.relationships_created)

        create_cypher = """
        UNWIND [
          {actor: "Tom Hanks", movie: "Forrest Gump", role: "Forrest"},
          {actor: "Tom Hanks", movie: "Cast Away", role: "Chuck Noland"},
          {actor: "Robin Wright", movie: "Forrest Gump", role: "Jenny"}
        ] AS row
        MERGE (p:Person {name: row.actor})
        MERGE (m:Movie {title: row.movie})
        MERGE (p)-[r:ACTED_IN {role: row.role}]->(m)
        RETURN p, r, m
        """
        _, create_summary, _ = driver.execute_query(create_cypher)
        print("Nodes created:", create_summary.counters.nodes_created)
        print("Relationships created:", create_summary.counters.relationships_created)

        read_cypher = """
        MATCH (p:Person {name: $actor})-[r:ACTED_IN]->(m:Movie)
        RETURN m.title AS title, r.role AS role
        """
        acted_records, _, _ = driver.execute_query(read_cypher, actor="Tom Hanks")
        for record in acted_records:
            print(record["title"], "-", record["role"])

        # SET: update a property on an existing node
        set_cypher = """
        MATCH (p:Person {name: $actor})
        SET p.birth_year = 1956
        RETURN p.name AS name, p.birth_year AS birth_year
        """
        set_records, _, _ = driver.execute_query(set_cypher, actor="Tom Hanks")
        for record in set_records:
            print(record["name"], "born", record["birth_year"])

        # DELETE: remove a single relationship (node stays, only the edge goes)
        delete_rel_cypher = """
        MATCH (:Person {name: $actor})-[r:ACTED_IN {role: $role}]->(:Movie {title: $movie})
        DELETE r
        """
        _, delete_rel_summary, _ = driver.execute_query(
            delete_rel_cypher, actor="Robin Wright", role="Jenny", movie="Forrest Gump"
        )
        print("Relationships deleted:", delete_rel_summary.counters.relationships_deleted)

        # DETACH DELETE: remove a node and any relationships still attached to it
        detach_delete_cypher = """
        MATCH (p:Person {name: $actor})
        DETACH DELETE p
        """
        _, detach_summary, _ = driver.execute_query(
            detach_delete_cypher, actor="Robin Wright"
        )
        print("Nodes deleted:", detach_summary.counters.nodes_deleted)


    finally:
        driver.close()


if __name__ == "__main__":
    main()
