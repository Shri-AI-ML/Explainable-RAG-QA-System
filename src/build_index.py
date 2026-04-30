import hashlib
import json
import shutil


def build_index(self, chunks):

    hash_file = os.path.join(self.persist_dir, "hash.txt")

    #  create hash of chunks
    data = json.dumps(chunks, sort_keys=True).encode()
    new_hash = hashlib.md5(data).hexdigest()

    #  check if DB exists + hash match
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            old_hash = f.read().strip()

        if old_hash == new_hash:
            print("[INFO] Using existing vector DB ")
            return  # ✔️ reuse DB

    #  rebuild required
    print("[INFO] Rebuilding vector DB...")

    #  IMPORTANT: delete only if DB not loaded
    if os.path.exists(self.persist_dir):
        try:
            shutil.rmtree(self.persist_dir)
        except PermissionError:
            print("[WARNING] DB in use → restart required")
            return

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

    #  save hash
    os.makedirs(self.persist_dir, exist_ok=True)
    with open(hash_file, "w") as f:
        f.write(new_hash)

    print("[INFO] Vector DB rebuilt ")