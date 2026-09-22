# This does the same thing as graph_UnStructured.py (turn one text file into a
# knowledge graph in Neo4j), but uses Neo4j's official "neo4j-graphrag" library
# instead of writing every step by hand. Compare the two files to see what the
# library does for you: chunking, embedding, entity extraction, and writing to
# Neo4j all happen inside SimpleKGPipeline.run_async(), instead of us writing
# five separate steps ourselves.

import asyncio
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from neo4j import GraphDatabase
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import AnthropicLLM
from neo4j_graphrag.retrievers import Text2CypherRetriever, VectorCypherRetriever
from neo4j_graphrag.schema import get_schema

load_dotenv()
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
HUGGINGFACE_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

# A second file this time, so it doesn't collide with what graph_UnStructured.py already saved.
FILE_PATH = "data/asciidoc/courses/30-days/modules/1-introduction/lessons/day-2-running-neo4j/lesson.adoc"


# The library needs an "Embedder" object to turn chunk text into vectors. It ships
# with OpenAIEmbeddings built in, but we're using Hugging Face, so we write a small
# class that plugs our existing embedding call into the shape the library expects:
# a class with one method, embed_query(text), that returns a list of numbers.
class HuggingFaceEmbedder(Embedder):
    def __init__(self, client):
        self.client = client

    def embed_query(self, text):
        return self.client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2").tolist()


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# AnthropicLLM is the library's built-in wrapper for calling Claude - this replaces
# our own get_entities() function and its prompt/JSON-parsing/code-fence-stripping code.

# thinking is disabled because the library's response parser doesn't handle Claude's
# "thinking" blocks yet - it only reads the first content block, which errors if
# thinking puts a ThinkingBlock there instead of text.
llm = AnthropicLLM(
    model_name="claude-sonnet-5",
    model_params={"max_tokens": 1024, "thinking": {"type": "disabled"}},
)

embedder = HuggingFaceEmbedder(InferenceClient(token=HUGGINGFACE_API_TOKEN))

# FixedSizeSplitter is the library's own splitter (this is also its default if
# you don't pass text_splitter at all) - splits by raw character count, with
# approximate=True to avoid cutting words in half at chunk boundaries.
text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)

# SimpleKGPipeline is the library's all-in-one pipeline: it reads the file, splits it
# into chunks, embeds each chunk, asks the LLM to extract entities/relationships, and
# writes everything into Neo4j - 5 of our 6 steps from graph_UnStructured.py, bundled.
# It does NOT create the vector index for us - that's a separate step below.
#
# perform_entity_resolution=True also fixes the duplicate-entity problem we ran into
# by hand (e.g. "Konigsberg bridge problem" vs "Euler's Konigsberg Bridge Problem") -
# the library merges similar entities automatically after extraction.
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    text_splitter=text_splitter,
    from_file=False,  # we're passing raw text below, not a file path
    perform_entity_resolution=True,
)

# The library's built-in file loader only supports .pdf/.md/.markdown, not .adoc.
# Since .adoc is just plain text anyway, we read it ourselves and pass the text
# directly instead of a file path.
with open(FILE_PATH, encoding="utf-8") as file:
    file_text = file.read()

# The pipeline's run method is async, so we use asyncio.run() to actually execute it.
result = asyncio.run(kg_builder.run_async(text=file_text))
print(result)

# The pipeline saved embeddings onto Chunk nodes but didn't index them - normally
# we'd create one here, but graph_UnStructured.py already created a vector index
# named "chunk_embedding" on this exact (Chunk, embedding) combination - Neo4j
# treats a second index on the same label+property as redundant and won't create
# it, so we just reuse the existing one instead (see CHUNK_VECTOR_INDEX below).
CHUNK_VECTOR_INDEX = "chunk_embedding"

print("Done. Check Neo4j for the new nodes from this file.")

# Print which entities got created. Extracted entities are connected to their
# source Chunk via a FROM_CHUNK relationship (the library's default naming -
# different from MENTIONED_IN, which is what our own hand-written script used).
#
# Note: since we passed raw text (not a file path), the library made up a
# placeholder Document path ("document.txt") instead of using our real FILE_PATH.
# DISTINCT matters here too - one entity can link to several chunks, which would
# otherwise print it once per chunk instead of once overall.
entity_records, _, _ = driver.execute_query(
    """
    MATCH (e)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(:Document {path: "document.txt"})
    RETURN DISTINCT labels(e) AS labels, e.name AS name
    """
)
print(f"\nEntities created ({len(entity_records)}):")
for record in entity_records:
    print(record["labels"], record["name"])

# --- VectorCypherRetriever: vector search + graph traversal in one query ---
#
# "node" refers to whatever the vector search matched (a Chunk, here). The
# retrieval_query runs FOR EACH match, so we can pull in extra graph context.
# We must include node.text too - without it, the LLM only sees entity names
# and has no actual passage content to answer from.
retrieval_query = """
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(entity)
WITH node, collect(DISTINCT entity.name) AS entities
RETURN node.text AS text, entities
"""

retriever = VectorCypherRetriever(
    driver,
    index_name=CHUNK_VECTOR_INDEX,
    retrieval_query=retrieval_query,
    embedder=embedder,
)

# --- GraphRAG: the full RAG loop ---
#
# GraphRAG.search() calls our retriever internally to fetch context, then hands
# that context + the question to the LLM to generate a real answer - one call
# does both retrieval and generation.
rag = GraphRAG(retriever=retriever, llm=llm)

query_text = input("Ask a question: ")
response = rag.search(
    query_text=query_text,
    retriever_config={"top_k": 5},
    return_context=True,
)
print(f"\nGraphRAG answer for: {query_text!r}")
print(response.answer)

# --- Text2CypherRetriever: no vector search, just LLM-written Cypher ---
#
# Good for precise/structured questions (counts, filters) that vector similarity
# can't answer well - e.g. "how many patients take insulin?" instead of "find
# text similar to insulin." get_schema() describes your database's labels,
# relationships and properties so the LLM knows what it can query against.
neo4j_schema = get_schema(driver)

text2cypher_retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=neo4j_schema,
)

cypher_question = input("\nAsk a structured question (e.g. a count or filter): ")
cypher_results = text2cypher_retriever.search(query_text=cypher_question)
print("\nText2CypherRetriever results:")
for item in cypher_results.items:
    print(item.content)

driver.close()
