from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.retrieval import run_retrieval
from src.llm_client import GroqClient

app = FastAPI()

llm = GroqClient()

# 💀 request body
class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "RAG API running 💀"}


@app.post("/query")
def query_api(request: QueryRequest):

    query = request.query

    # 🔥 run retrieval with dynamic query
    retrieved_chunks = run_retrieval(query)

    answer = llm.generate_answer(query, retrieved_chunks)

    return {
    "question": query,
    "answer": answer,
    "sources": [
        {
            "doc": c["doc_id"],
            "preview": c["text"][:120]
        }
        for c in retrieved_chunks
    ]
}