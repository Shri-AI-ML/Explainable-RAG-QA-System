from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
import hashlib
import json


class VectorStore:
    def __init__(self):
        self.persist_dir = "chroma_db"

        #  load embedding model
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = None

    # ---------------- HASH FUNCTION ----------------
    def get_chunks_hash(self, chunks):
        data = json.dumps(chunks, sort_keys=True).encode()
        return hashlib.md5(data).hexdigest()

    # ---------------- BUILD INDEX ----------------
    def build_index(self, chunks):

        hash_file = os.path.join(self.persist_dir, "hash.txt")
        new_hash = self.get_chunks_hash(chunks)

        #  check if DB exists + hash same
        if os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                old_hash = f.read().strip()

            if old_hash == new_hash:
                print("[INFO] Using existing vector DB ")

                self.db = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embedding
                )
                return

        #  rebuild required
        print("[INFO] Rebuilding vector DB...")

        # delete old DB safely
        if os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
            except PermissionError:
                print("[WARNING] DB in use, skipping delete (restart may be needed)")

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

        # save new hash
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(hash_file, "w") as f:
            f.write(new_hash)

        print("[INFO] Vector DB rebuilt ")

    # ---------------- SEARCH ----------------
    def search(self, query, top_k=3):
        if self.db is None:
            raise ValueError("Vector DB not initialized!")

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