from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os


class VectorStore:
    def __init__(self):
        self.persist_dir = "chroma_db"

        # 🔥 simple load (no singleton, no overhead)
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )

    def build_index(self, chunks):
        # 💀 NEVER rebuild again
        return

    def search(self, query, top_k=3):
        # 🔥 DIRECT FAST SEARCH (no MMR)
        results = self.db.similarity_search_with_score(query, k=top_k)

        formatted = []

        for doc, score in results:
            formatted.append({
                "doc_id": doc.metadata["doc_id"],
                "chunk_id": doc.metadata["chunk_id"],
                "score": float(score),
                "text": doc.page_content
            })

        return formatted