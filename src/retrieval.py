import json
import os
from pathlib import Path
from chunking import chunk_documents
from vector_store import VectorStore
from rank_bm25 import BM25Okapi

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD DOCUMENTS ----------------
chunks = chunk_documents(str(DATA_DIR / "raw_cleaned.json"))
print(f"[INFO] Total chunks created: {len(chunks)}")

# ---------------- BM25 SETUP ----------------
tokenized_corpus = [c["text"].lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized_corpus)

# ---------------- BUILD VECTOR STORE ----------------
store = VectorStore()
store.build_index(chunks)

# ---------------- READ QUERY ----------------
with open(DATA_DIR / "query.txt", "r", encoding="utf-8") as f:
    query = f.read().strip()

if not query:
    raise ValueError("Query file is empty!")

# ---------------- CLEAN QUERY ----------------
stopwords = {"what", "is", "the", "a", "an", "of"}
tokenized_query = [
    word for word in query.lower().split()
    if word not in stopwords
]

# ---------------- BM25 SEARCH ----------------
bm25_scores = bm25.get_scores(tokenized_query)

max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1

bm25_results = []
for i, score in enumerate(bm25_scores):

    # 🔥 Exact match boost
    if query.lower() in chunks[i]["text"].lower():
        score *= 1.2

    bm25_results.append({
        "doc_id": chunks[i]["doc_id"],
        "chunk_id": chunks[i]["chunk_id"],
        "score": float(score) / max_bm25,
        "text": chunks[i]["text"]
    })

bm25_results = sorted(bm25_results, key=lambda x: x["score"], reverse=True)[:5]

# ---------------- VECTOR SEARCH ----------------
vector_results = store.search(query, top_k=5)

# ---------------- NORMALIZE VECTOR SCORES ----------------
vector_scores = [r["score"] for r in vector_results]

max_vec = max(vector_scores) if vector_scores else 1
min_vec = min(vector_scores) if vector_scores else 0

for r in vector_results:
    if max_vec - min_vec != 0:
        r["score"] = (r["score"] - min_vec) / (max_vec - min_vec)
    else:
        r["score"] = 0.0

# ---------------- HYBRID WEIGHTED FUSION ----------------
alpha = 0.7  # vector weight
beta = 0.3   # BM25 weight

combined_dict = {}

# Vector results
for r in vector_results:
    combined_dict[r["chunk_id"]] = {
        "doc_id": r["doc_id"],
        "chunk_id": r["chunk_id"],
        "text": r["text"],
        "score": alpha * r["score"]
    }

# BM25 results
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

# ---------------- OUTPUT ----------------
print("\n=== TOP RETRIEVED CHUNKS ===\n")

for i, r in enumerate(results, 1):
    preview = r["text"][:200].replace("\n", " ") + "..."
    print(f"[{i}] Document  : {r['doc_id']}")
    print(f"    Chunk ID : {r['chunk_id']}")
    print(f"    Score    : {round(r['score'], 3)}")
    print(f"    Preview  : {preview}")
    print("-" * 60)

# ---------------- SAVE JSON ----------------
with open(OUTPUT_DIR / "retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n[INFO] Retrieval results saved to outputs/retrieval_results.json")