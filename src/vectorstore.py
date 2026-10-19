
import os
from typing import Any, List

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding import EmbeddingPipeline
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = VECTORSTORE_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        collection_name: str = "documents"
    ):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.model = SentenceTransformer(embedding_model)

        print(f"[INFO] Loaded embedding model: {embedding_model}")
        print(f"[INFO] ChromaDB path: {self.persist_dir}")

    def build_from_documents(self, documents: List[Any]):
        print(
            f"[INFO] Building vector store from "
            f"{len(documents)} documents..."
        )

        pipeline = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = pipeline.chunk_documents(documents)

        if not chunks:
            print("[INFO] No chunks found.")
            return

        embeddings = pipeline.embed_chunks(chunks)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        metadatas = [
            {
                "source": str(
                    chunk.metadata.get("source", "unknown")
                )
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(
            f"[INFO] Stored {len(chunks)} chunks in ChromaDB."
        )

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Searching for: {query_text}")

        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode(
            [query_text]
        ).astype(np.float32)

        result_count = min(
            max(top_k, 1),
            self.collection.count()
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=result_count,
            include=["documents", "metadatas", "distances"]
        )

        matches = []

        ids = results["ids"][0]
        texts = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, metadata, distance in zip(
            ids, texts, metadatas, distances
        ):
            matches.append({
                "id": chunk_id,
                "distance": distance,
                "metadata": {
                    "text": text,
                    **(metadata or {})
                }
            })

        return matches
