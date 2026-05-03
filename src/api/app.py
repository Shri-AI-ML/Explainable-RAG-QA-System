from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.retrieval import run_retrieval
from src.llm_client import GroqClient

app = FastAPI()


#  REQUEST MODEL 
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

        # Lazy load everything (important for Render)
        llm = GroqClient()

        # Retrieval
        retrieved_chunks = run_retrieval(query)

        #  Generate answer
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