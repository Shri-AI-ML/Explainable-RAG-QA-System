from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None

    def build_index(self, chunks):
        texts = [c["text"] for c in chunks]
        metadatas = [
            {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"]}
            for c in chunks
        ]

        self.db = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding,
            metadatas=metadatas,
            persist_directory="chroma_db"
        )

    def search(self, query, top_k=5):
    # Step 1: Get results with scores
        scored_results = self.db.similarity_search_with_score(query, k=10)

        # Step 2: Apply MMR for diversity
        mmr_results = self.db.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=10
        )

        # Step 3: Map scores
        score_map = {
            doc.page_content: score
            for doc, score in scored_results
        }

        formatted_results = []

        for doc in mmr_results:
            formatted_results.append({
                "doc_id": doc.metadata["doc_id"],
                "chunk_id": doc.metadata["chunk_id"],
                "score": float(score_map.get(doc.page_content, 0.0)),
                "text": doc.page_content
            })

        return formatted_results