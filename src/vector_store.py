import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = []
        self.metadata = []

    def build_index(self, chunks):
        texts = [c["text"] for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.metadata = chunks

    def search(self, query, top_k=5):
        query_emb = self.model.encode(query)

        scores = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "doc_id": self.metadata[idx]["doc_id"],
                "chunk_id": self.metadata[idx]["chunk_id"],
                "score": float(scores[idx]),
                "text": self.metadata[idx]["text"]
            })

        return results
