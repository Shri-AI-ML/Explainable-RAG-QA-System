import json
import subprocess
from llm_client import GeminiClient
from explain import ExplainabilityModule

RETRIEVAL_JSON = "outputs/retrieval_results.json"


def main():
    # ---------- RUN RETRIEVAL SCRIPT ----------
    print("[INFO] Running retrieval step...")
    subprocess.run(
        ["python", "src/retrieval.py"],
        check=True
    )

    # ---------- LOAD RETRIEVAL RESULTS ----------
    with open(RETRIEVAL_JSON, "r", encoding="utf-8") as f:
        retrieved_chunks = json.load(f)

    # ---------- QUERY ----------
    query = "What is the current state of AI?"

    # ---------- LLM ----------
    llm = GeminiClient()
    answer = llm.generate_answer(query, retrieved_chunks)

    # ---------- EXPLAINABILITY ----------
    explainer = ExplainabilityModule()
    report = explainer.generate_explainability_report(
        query=query,
        answer=answer,
        retrieved_chunks=retrieved_chunks
    )

    print(json.dumps(report, indent=2))
    explainer.visualize_evidence()


if __name__ == "__main__":
    main()
