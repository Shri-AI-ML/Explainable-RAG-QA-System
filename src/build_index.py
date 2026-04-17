from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

class VectorStore:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.persist_dir = "chroma_db"

        # 🔥 If DB exists → load
        if os.path.exists(self.persist_dir):
            self.db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding
            )
            print("[INFO] Loaded existing vector DB")
        else:
            self.db = None

    def build_index(self, chunks):
        if self.db is not None:
            return  # 💀 skip rebuild

        texts = [c["text"] for c in chunks]
        metadatas = [
            {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"]}
            for c in chunks
        ]

        self.db = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding,
            metadatas=metadatas,
            persist_directory=self.persist_dir
        )

        print("[INFO] Vector DB created and saved")