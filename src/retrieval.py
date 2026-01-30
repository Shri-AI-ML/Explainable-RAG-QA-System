import json
from chunking import chunk_documents
from vector_store import VectorStore

# ---------------- LOAD DOCUMENTS ----------------
chunks = chunk_documents("data/raw_cleaned.json")
print(f"[INFO] Total chunks created: {len(chunks)}")

# ---------------- BUILD VECTOR STORE ----------------
store = VectorStore()
store.build_index(chunks)

# ---------------- READ QUERY ----------------
with open("data/query.txt", "r", encoding="utf-8") as f:
    query = f.read().strip()

if not query:
    raise ValueError("Query file is empty!")

# ---------------- RETRIEVE ----------------
results = store.search(query, top_k=5)

# ---------------- REMOVE DUPLICATES ----------------
seen = set()
clean_results = []

for r in results:
    if r["chunk_id"] not in seen:
        seen.add(r["chunk_id"])
        clean_results.append(r)

results = clean_results

# ---------------- CLEAN CONSOLE OUTPUT ----------------
print("\n=== TOP RETRIEVED CHUNKS ===\n")

for i, r in enumerate(results, 1):
    preview = r["text"][:200].replace("\n", " ") + "..."
    print(f"[{i}] Document  : {r['doc_id']}")
    print(f"    Chunk ID : {r['chunk_id']}")
    print(f"    Score    : {round(r['score'], 3)}")
    print(f"    Preview  : {preview}")
    print("-" * 60)

# ---------------- SAVE JSON FOR ARYAN ----------------
with open("outputs/retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n[INFO] Retrieval results saved to outputs/retrieval_results.json")
