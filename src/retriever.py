import logging
import numpy as np
import faiss
from langchain.schema import Document
from src.embedder import Embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: Embedder):
        """
        Retriever wraps the Embedder and handles similarity search.
        Accepts an already-initialized Embedder instance.
        """
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Embed the query and retrieve top-k most similar chunks from FAISS.
        Returns a list of dicts with keys: content, metadata, score.
        """
        if self.embedder.index is None:
            logger.warning("No FAISS index loaded. Please index documents first.")
            return []

        if not query.strip():
            logger.warning("Empty query received.")
            return []

        # Embed the query
        query_embedding = self.embedder.model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Adjust k if fewer chunks exist
        actual_k = min(k, len(self.embedder.chunks))

        # Search FAISS index
        scores, indices = self.embedder.index.search(query_embedding, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.embedder.chunks[idx]
            results.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "score": float(score)
            })

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
        return results

    def retrieve_with_threshold(self, query: str, k: int = 5, threshold: float = 0.2) -> list[dict]:
        """
        Same as retrieve() but filters out chunks below a similarity threshold.
        """
        results = self.retrieve(query, k)
        filtered = [r for r in results if r["score"] >= threshold]
        logger.info(f"{len(filtered)} chunks passed threshold {threshold} (from {len(results)} retrieved)")
        return filtered

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM prompt.
        Includes source filename and page number for each chunk.
        """
        if not retrieved_chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk["metadata"].get("source", "unknown")
            page = chunk["metadata"].get("page", "?")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{chunk['content']}"
            )

        return "\n\n---\n\n".join(context_parts)