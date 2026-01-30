import json

def chunk_documents(path, chunk_size=250, overlap=50):
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    chunks = []

    for idx, doc in enumerate(docs):
        # 🔒 SAFE doc_id handling (NO key error possible)
        doc_id = f"wiki_{idx+1}"
        text = doc.get("text", "")

        if not text.strip():
            continue

        words = text.split()
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                "doc_id": doc_id,
                "text": chunk_text
            })

            start += chunk_size - overlap
            chunk_idx += 1

    return chunks
