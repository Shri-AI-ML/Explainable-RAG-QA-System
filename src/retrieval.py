from chunking import load_and_chunk
from vector_store import VectorStore



chunks = load_and_chunk("data/dummy.json")

store = VectorStore()
store.build_index(chunks)


with open("data/querry.txt", "r") as f:
    question = f.read().strip()

if not question:
    raise ValueError("Question file is empty!")

results = store.search(question)

for r in results:
    print(r)
