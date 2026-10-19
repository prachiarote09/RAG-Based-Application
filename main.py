
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.search import RAGSearch


app = FastAPI(
    title="RAG API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize RAG Search using ChromaDB
rag = RAGSearch(persist_dir="./chroma_db")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1)


class QueryResponse(BaseModel):
    query: str
    summary: str
    top_k: int


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/search", response_model=QueryResponse)
def search(request: QueryRequest):
    try:
        summary = rag.search_and_summarize(
            request.query,
            top_k=request.top_k
        )

        return QueryResponse(
            query=request.query,
            summary=summary,
            top_k=request.top_k
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/")
def root():
    return {
        "message": "RAG API - Use /docs for API documentation"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
