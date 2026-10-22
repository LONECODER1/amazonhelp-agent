"""
Ingestion pipeline for AmazonHelp Customer Support RAG Agent.

Steps:
  1. Load data/sample_amazon_dataset.csv using CSVLoader
  2. Chunk using RecursiveCharacterTextSplitter
  3. Embed using BAAI/bge-small-en-v1.5 via HuggingFace Inference API
  4. Store in local Qdrant (collection: amazon_support)

Usage:
  python -m src.ingestion.ingest
  OR
  python src/ingestion/ingest.py
"""

import os
import sys
import logging
from pathlib import Path

# Ensure project root is in path when run directly
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "sample_amazon_dataset.csv"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "amazon_support")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 50  # 50 docs per batch to avoid HTTP keep-alive timeouts on cloud clusters


from langsmith import traceable


@traceable(name="run_ingestion", tags=["ingestion", "qdrant"], metadata={"component": "data_indexer"})
def run_ingestion():
    """Full ingestion pipeline: load → chunk → embed → upsert to Qdrant."""
    import time
    from langchain_community.document_loaders import CSVLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    from src.utils.llm import get_embeddings
    from src.utils.vector_store import (
        QDRANT_HOST,
        QDRANT_PORT,
        get_qdrant_client,
    )

    client = get_qdrant_client()

    # ── 1. Check if already ingested or partially ingested ───────────────────
    force_recreate = "--force" in sys.argv or "--recreate" in sys.argv

    try:
        info = client.get_collection(QDRANT_COLLECTION)
        existing_points = getattr(info, "points_count", 0) or getattr(info, "vectors_count", 0) or 0
    except Exception:
        existing_points = 0

    if existing_points >= 3000 and not force_recreate:
        logger.info(
            "Collection '%s' already contains %d vectors (complete). Skipping ingestion. "
            "Pass --force to re-ingest.",
            QDRANT_COLLECTION,
            existing_points,
        )
        return

    if 0 < existing_points < 3000 or force_recreate:
        logger.warning(
            "Collection '%s' has %d points (incomplete). Deleting and recreating for clean ingestion.",
            QDRANT_COLLECTION,
            existing_points,
        )
        try:
            client.delete_collection(QDRANT_COLLECTION)
        except Exception as e:
            logger.warning("Could not delete existing collection: %s", e)

    # ── 2. Load CSV ───────────────────────────────────────────────────────────
    logger.info("Loading CSV from: %s", DATA_PATH)
    loader = CSVLoader(
        file_path=str(DATA_PATH),
        encoding="utf-8",
        csv_args={"delimiter": ","},
    )
    documents = loader.load()
    logger.info("Loaded %d raw documents", len(documents))

    # ── 3. Chunk ─────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    # ── 4. Create collection if not exists ────────────────────────────────────
    embeddings = get_embeddings()

    # Determine embedding dimension from a test embed
    logger.info("Resolving embedding dimension…")
    test_vec = embeddings.embed_query("test")
    dim = len(test_vec)
    logger.info("Embedding dimension: %d", dim)

    try:
        client.get_collection(QDRANT_COLLECTION)
        logger.info("Collection '%s' already exists. Reusing.", QDRANT_COLLECTION)
    except Exception:
        logger.info("Creating collection '%s'…", QDRANT_COLLECTION)
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # ── 5. Upsert in batches ─────────────────────────────────────────────────
    logger.info("Upserting %d chunks in batches of %d…", len(chunks), BATCH_SIZE)
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info("Batch %d / %d (%d docs)…", batch_num, total_batches, len(batch))

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                vector_store.add_documents(documents=batch)
                break
            except Exception as e:
                logger.warning(
                    "Batch %d attempt %d failed with error (%s). Retrying in %ds…",
                    batch_num,
                    attempt,
                    e,
                    attempt * 2,
                )
                if attempt == max_retries:
                    raise
                time.sleep(attempt * 2)

    # Verify
    info = client.get_collection(QDRANT_COLLECTION)
    count = getattr(info, "vectors_count", getattr(info, "points_count", 0))
    logger.info(
        "✅ Ingestion complete! Collection '%s' now has %s vectors.",
        QDRANT_COLLECTION,
        count,
    )


if __name__ == "__main__":
    run_ingestion()

