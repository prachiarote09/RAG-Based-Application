
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.vectorstore import ChromaVectorStore

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b"
    ):
        # Initialize ChromaDB vector store
        self.vectorstore = ChromaVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model
        )

        # Build the vector store if the collection is empty
        if self.vectorstore.collection.count() == 0:
            print("[INFO] ChromaDB collection is empty. Building vector store...")

            from src.data_loader import load_all_documents

            documents = load_all_documents("data")

            if documents:
                self.vectorstore.build_from_documents(documents)
            else:
                print("[WARNING] No documents found in the data directory.")

        else:
            print(
                f"[INFO] Loaded existing ChromaDB collection with "
                f"{self.vectorstore.collection.count()} chunks."
            )

        # Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.7
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        # Retrieve relevant document chunks from ChromaDB
        results = self.vectorstore.query(
            query_text=query,
            top_k=top_k
        )

        # Extract text from retrieved chunks
        texts = [
            result["metadata"].get("text", "")
            for result in results
            if result.get("metadata")
        ]

        # Combine retrieved chunks into context
        context = "\n\n".join(
            text for text in texts if text.strip()
        )

        if not context:
            return "No relevant documents found."

        # Create prompt for the LLM
        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using the provided document context.

Instructions:
- Give a clear and concise answer.
- Use the supplied context as your source of information.
- Do not invent facts that are not supported by the context.
- If the answer is not available in the context, clearly say so.

User question:
{query}

Retrieved document context:
{context}

Answer:
"""

        # Generate the final answer using Groq
        response = self.llm.invoke(prompt)

        return response.content
