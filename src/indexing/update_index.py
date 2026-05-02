import json
import logging
from pathlib import Path
from src.core.vector_store import VectorStore

#  LOGGING 
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    #  PATHS 
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"
    INDEXED_PATH = PROJECT_ROOT / "data" / "indexed_chunks.json"

    #  LOAD CHUNKS ----------------
    if not CHUNKS_PATH.exists():
        logger.error("[ERROR] chunks.json not found")
        return

    logger.info("[INFO] Loading chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    #  INIT VECTOR STORE 
    store = VectorStore()

    #  ensure DB is initialized
    if store.db is None:
        logger.info("[INFO] Initializing new vector DB...")

        store.db = store._init_db() 

    #  LOAD INDEXED IDS 
    indexed_ids = set()

    if INDEXED_PATH.exists():
        with open(INDEXED_PATH, "r", encoding="utf-8") as f:
            indexed_ids = set(json.load(f))

    logger.info(f"[INFO] Already indexed: {len(indexed_ids)}")

    # FILTER NEW CHUNKS 
    new_texts = []
    new_metadatas = []
    new_ids = []

    skipped = 0

    for chunk in chunks:
        cid = chunk["chunk_id"]

        if cid in indexed_ids:
            skipped += 1
            continue

        new_texts.append(chunk["text"])
        new_metadatas.append({
            "doc_id": chunk["doc_id"],
            "chunk_id": cid
        })
        new_ids.append(cid)

    logger.info(f"[INFO] Skipped duplicates: {skipped}")

    #  ADD TO VECTOR DB 
    if not new_texts:
        logger.info("[INFO] No new chunks to index ")
        return

    logger.info(f"[INFO] Adding {len(new_texts)} new chunks...")

    store.db.add_texts(
        texts=new_texts,
        metadatas=new_metadatas,
        ids=new_ids
    )

    #  SAVE UPDATED IDS 
    indexed_ids.update(new_ids)

    with open(INDEXED_PATH, "w", encoding="utf-8") as f:
        json.dump(list(indexed_ids), f, indent=2)

    logger.info("[INFO] Index updated successfully ")


if __name__ == "__main__":
    main()