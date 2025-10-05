"""
index_kb.py
-----------
Creates and stores vector embeddings for the KB using Pinecone via LangChain.
"""

import os
import json
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv(override=True)

# ========== CONFIG ==========
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "agentic-rag-kb"
EMBEDDING_MODEL = "models/gemini-embedding-001"  # Gemini embedding model
DATA_PATH = "data/self_critique_loop_dataset.json"
METRIC = "cosine"
# =============================

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# =============================

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found. Please set it in your environment.")
else:
  logger.info("PINECONE API KEY found")


def get_embedding_dimension():
    """
    Generate a temporary embedding vector to infer model output dimension.
    """
    logger.info(f"Checking embedding dimension for model: {EMBEDDING_MODEL}")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    test_vec = embeddings.embed_query("test sentence for dimension check")
    dim = len(test_vec)
    logger.info(f"Detected embedding dimension: {dim}")
    return dim


def create_index(delete_index=True):
    logger.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    dim = get_embedding_dimension()


    # Always delete index if it exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:

        logger.info(f"Creating fresh Pinecone index '{PINECONE_INDEX_NAME}' with dimension {dim}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    elif delete_index:
        logger.warning(f"Deleting existing index '{PINECONE_INDEX_NAME}' before recreation...")
        pc.delete_index(PINECONE_INDEX_NAME)
    else:
        logger.info(f"Found Pinecone Index {PINECONE_INDEX_NAME}..")

def index_kb():
    logger.info("Loading KB data...")
    with open(DATA_PATH, "r") as f:
        kb_data = json.load(f)

    texts = [item["answer_snippet"] for item in kb_data]
    metadatas = [
        {
            "doc_id": item["doc_id"],
            "question": item["question"],
            "source": item["source"],
            "confidence": item["confidence_indicator"],
            "last_updated": item["last_updated"]
        }
        for item in kb_data
    ]
    ids = [item["doc_id"] for item in kb_data]

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    logger.info("Upserting texts into Pinecone...")
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    logger.info("✅ KB indexing complete.")

if __name__ == "__main__":
    create_index()
    index_kb()