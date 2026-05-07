import os
import pickle
import logging
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "faiss_index/index.faiss"
FAISS_PKL_PATH = "faiss_index/index.pkl"


class Embedder:
    def __init__(self):
        logger.info("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []  # stores all Document objects
        self.indexed_files = {}  # filename -> chunk count
        logger.info("Model loaded successfully.")

    def embed_documents(self, new_chunks: list[Document], filename: str) -> None:
        """
        Embed a list of Document chunks and add them to the FAISS index.
        Saves index to disk after embedding.
        """
        if not new_chunks:
            logger.warning(f"No chunks to embed for {filename}")
            return

        # Extract text content
        texts = [chunk.page_content for chunk in new_chunks]

        logger.info(f"Embedding {len(texts)} chunks from {filename}...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create or update FAISS index
        dimension = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        self.indexed_files[filename] = self.indexed_files.get(filename, 0) + len(new_chunks)

        logger.info(f"Added {len(new_chunks)} chunks. Total chunks in index: {len(self.chunks)}")
        self._save_index()

    def _save_index(self) -> None:
        """Save FAISS index and chunk metadata to disk."""
        os.makedirs("faiss_index", exist_ok=True)
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(FAISS_PKL_PATH, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "indexed_files": self.indexed_files
            }, f)
        logger.info("Index saved to disk.")

    def load_index(self) -> bool:
        """
        Load existing FAISS index from disk.
        Returns True if loaded successfully, False if not found.
        """
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_PKL_PATH):
            logger.info("Loading existing FAISS index from disk...")
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_PKL_PATH, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.indexed_files = data["indexed_files"]
            logger.info(f"Loaded index with {len(self.chunks)} chunks.")
            return True
        else:
            logger.info("No existing index found.")
            return False

    def is_file_indexed(self, filename: str) -> bool:
        """Check if a file has already been indexed (cache check)."""
        return filename in self.indexed_files

    def get_index_info(self) -> dict:
        """Return info about the current index."""
        return {
            "total_chunks": len(self.chunks),
            "indexed_files": self.indexed_files,
            "index_loaded": self.index is not None
        }