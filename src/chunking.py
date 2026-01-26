import json

def chunk_text(text, chunk_size=40):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def load_and_chunk(path):
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    chunked_docs = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}_chunk_{idx}",
                "text": chunk
            })

    return chunked_docs
