import json
import os
import pickle
from pathlib import Path
from src.chunking import chunk_documents
from src.vector_store import VectorStore
from rank_bm25 import BM25Okapi

USE_SAVED_CHUNKS = True

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BM25_PATH = DATA_DIR / "bm25.pkl"


def run_retrieval(query=None):

    # ---------------- LOAD DOCUMENTS ----------------
    CHUNKS_PATH = DATA_DIR / "chunks.json"

    if USE_SAVED_CHUNKS and CHUNKS_PATH.exists():
        print("[INFO] Loading saved chunks...")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        print("[INFO] Running chunking...")
        chunks = chunk_documents(str(DATA_DIR / "raw_cleaned.json"))

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Total chunks: {len(chunks)}")

    # ---------------- BM25 (DISK CACHE ) ----------------
    rebuild_bm25 = True
    if BM25_PATH.exists() and CHUNKS_PATH.exists():
        if os.path.getmtime(BM25_PATH) > os.path.getmtime(CHUNKS_PATH):
            rebuild_bm25 = False

    if not rebuild_bm25:
        print("[INFO] Loading BM25 from disk ")
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
    else:
        print("[INFO] Building BM25 index...")
        tokenized_corpus = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)

        print("[INFO] BM25 saved")

    # ---------------- VECTOR STORE ----------------
    store = VectorStore()
    store.build_index(chunks)

    # ---------------- READ QUERY ----------------
    if not query:
        with open(DATA_DIR / "query.txt", "r", encoding="utf-8") as f:
            query = f.read().strip()

    if not query:
        raise ValueError("Query is empty!")

    query_lower = query.lower()  #  optimize

    # ---------------- CLEAN QUERY ----------------
    stopwords = {
    "what","is","the","a","an","of","in","on","for",
    "to","with","and","by","from","at","as"
    }
    tokenized_query = [
        word for word in query_lower.split()
        if word not in stopwords
    ]

    # ---------------- BM25 SEARCH ----------------
    bm25_scores = bm25.get_scores(tokenized_query)

    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1

    bm25_results = []
    for i, score in enumerate(bm25_scores):

        #  optimized string check (no repeated lower())
        if query_lower in chunks[i]["text"].lower():
            score *= 1.2

        bm25_results.append({
            "doc_id": chunks[i]["doc_id"],
            "chunk_id": chunks[i]["chunk_id"],
            "score": float(score) / max_bm25,
            "text": chunks[i]["text"]
        })

    bm25_results = sorted(bm25_results, key=lambda x: x["score"], reverse=True)[:5]

    # ---------------- VECTOR SEARCH ----------------
    vector_results = store.search(query, top_k=3)

    # ---------------- NORMALIZE VECTOR SCORES ----------------
    vector_scores = [r["score"] for r in vector_results]

    max_vec = max(vector_scores) if vector_scores else 1
    min_vec = min(vector_scores) if vector_scores else 0

    for r in vector_results:
        if max_vec - min_vec != 0:
            # Chroma returns distance (lower is better), so we invert it
            r["score"] = 1.0 - ((r["score"] - min_vec) / (max_vec - min_vec))
        else:
            r["score"] = 1.0

    # ---------------- HYBRID FUSION ----------------
    alpha = 0.85
    beta = 0.15

    combined_dict = {}

    for r in vector_results:
        combined_dict[r["chunk_id"]] = {
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "score": alpha * r["score"]
        }

    for r in bm25_results:
        if r["chunk_id"] in combined_dict:
            combined_dict[r["chunk_id"]]["score"] += beta * r["score"]
        else:
            combined_dict[r["chunk_id"]] = {
                "doc_id": r["doc_id"],
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "score": beta * r["score"]
            }

    # ---------------- FINAL RESULTS ----------------
    results = sorted(
        combined_dict.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:5]

    # ---------------- SAVE JSON ----------------
    with open(OUTPUT_DIR / "retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("[INFO] Retrieval results saved to outputs/retrieval_results.json")

    return results


if __name__ == "__main__":
    run_retrieval()