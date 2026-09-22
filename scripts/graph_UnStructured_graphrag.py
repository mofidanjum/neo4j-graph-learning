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
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import AnthropicLLM

load_dotenv()
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
HUGGINGFACE_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]
EMBEDDING_SIZE = 384  # must match the embedding model in HuggingFaceEmbedder below

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

# The pipeline saved embeddings onto Chunk nodes but didn't index them - do that now
# (IF NOT EXISTS equivalent via fail_if_exists=False, so this is safe to rerun).
create_vector_index(
    driver,
    name="graphrag_chunk_embedding",
    label="Chunk",
    embedding_property="embedding",
    dimensions=EMBEDDING_SIZE,
    similarity_fn="cosine",
    fail_if_exists=False,
)

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

driver.close()
