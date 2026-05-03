from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.retrieval import run_retrieval
from src.llm_client import GroqClient

app = FastAPI()


#  request body
class QueryRequest(BaseModel):
    query: str


#  HEALTH CHECK 
@app.get("/")
def home():
    return {"message": "RAG API running "}


#  QUERY ENDPOINT 
@app.post("/query")
def query_api(request: QueryRequest):
    try:
        query = request.query

        #  run retrieval with dynamic query
        retrieved_chunks = run_retrieval(query)

        #  Generate answer
        llm = GroqClient()
        answer = llm.generate_answer(query, retrieved_chunks)

        return {
            "question": query,
            "answer": answer,
            "sources": [
                {
                    "doc": c.get("doc_id", ""),
                    "preview": c.get("text", "")[:120]
                }
                for c in retrieved_chunks
            ]
        }

    except Exception as e:
        return {
            "error": str(e)
        }