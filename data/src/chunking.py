from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def chunk_documents(path, chunk_size=500, overlap=100):
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []

    for idx, doc in enumerate(docs):
        doc_id = f"wiki_{idx+1}"
        text = doc.get("text", "")

        if not text.strip():
            continue

        split_texts = splitter.split_text(text)

        for chunk_idx, chunk in enumerate(split_texts):
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                "doc_id": doc_id,
                "text": chunk
            })

    return chunks