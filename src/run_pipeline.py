import json
import time
from llm_client import GroqClient
from explain import ExplainabilityModule
from retrieval import run_retrieval  # 💀 NEW IMPORT

def main():
    start_total = time.time()

    # ---------- RUN RETRIEVAL DIRECTLY ----------
    print("[INFO] Running retrieval step...")
    t1 = time.time()

    retrieved_chunks = run_retrieval()  # 💀 NO subprocess

    print("[TIME] Retrieval:", round(time.time() - t1, 2), "sec")

    # ---------- QUERY ----------
    with open("data/query.txt") as f:
        query = f.read().strip()

    # ---------- LLM ----------
    llm = GroqClient()

    t2 = time.time()

    answer = llm.generate_answer(query, retrieved_chunks)

    print("[TIME] LLM:", round(time.time() - t2, 2), "sec")

    # ---------- EXPLAINABILITY ----------
    t3 = time.time()

    explainer = ExplainabilityModule()
    report = explainer.generate_explainability_report(
        query=query,
        answer=answer,
        retrieved_chunks=retrieved_chunks
    )

    print("[TIME] Explainability:", round(time.time() - t3, 2), "sec")

    # ---------- OUTPUT ----------
    print(json.dumps(report, indent=2))
    explainer.visualize_evidence()

    print("[TIME] TOTAL:", round(time.time() - start_total, 2), "sec")


if __name__ == "__main__":
    main()


## delete data/bm25.pkl