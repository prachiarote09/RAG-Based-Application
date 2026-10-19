from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import ChromaVectorStore
from src.search import RAGSearch

app = FastAPI(
    title="RAG API",
    version="1.0.0"
)
# Example usage

if __name__ == "__main__":

    # Initialize RAG Search using ChromaDB
    rag = RAGSearch(persist_dir="./chroma_db")

    # Example query
    query = "What are the Types of SECURITY?"

    print(f"\n[USER QUERY] {query}\n")

    # Get the answer using retrieved document context
    answer = rag.search_and_summarize(query, top_k=3)

    print(f"[ANSWER]\n{answer}\n")
