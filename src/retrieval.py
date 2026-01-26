from chunking import load_and_chunk
from vector_store import VectorStore

chunks = load_and_chunk("data/dummy.json")

store = VectorStore()
store.build_index(chunks)

query = "What is Machine Learning ?"
results = store.search(query)

for r in results:
    print(r)
