import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.document_processor import DocumentProcessor
from src.embedder import Embedder
from src.retriever import Retriever
from src.llm_handler import LLMHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Global components ────────────────────────────────────────────────────────
embedder = None
retriever = None
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, retriever, llm
    logger.info("Starting up RAG API...")
    embedder = Embedder()
    embedder.load_index()
    retriever = Retriever(embedder)
    llm = LLMHandler()
    logger.info("RAG API ready.")
    yield
    logger.info("Shutting down RAG API...")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Q&A API",
    description="REST API for querying PDF documents using RAG architecture",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─── Request/Response Models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    k: int = 5
    conversation_history: list[dict] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_used: int
    question: str


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    total_chunks: int
    indexed_files: dict


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API status and index info."""
    info = embedder.get_index_info()
    return HealthResponse(
        status="ok",
        total_chunks=info["total_chunks"],
        indexed_files=info["indexed_files"]
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a PDF document.
    Returns indexing status and chunk count.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    filename = file.filename

    if embedder.is_file_indexed(filename):
        return {
            "filename": filename,
            "status": "already_indexed",
            "message": f"{filename} is already indexed.",
            "chunk_count": embedder.indexed_files.get(filename, 0)
        }

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        processor = DocumentProcessor()
        with open(tmp_path, "rb") as f:
            result = processor.process_pdf(f, filename)

        if result["status"] != "success" or result["chunk_count"] == 0:
            raise HTTPException(
                status_code=422,
                detail=f"Could not extract text from {filename}. Make sure it is a text-based PDF."
            )

        embedder.embed_documents(result["chunks"], filename)

        return {
            "filename": filename,
            "status": "success",
            "pages": result["pages"],
            "chunk_count": result["chunk_count"],
            "message": f"Successfully indexed {filename}"
        }

    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question against indexed documents.
    Returns answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    info = embedder.get_index_info()
    if info["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF first."
        )

    chunks = retriever.retrieve(request.question, k=request.k)
    result = llm.generate_answer(
        question=request.question,
        retrieved_chunks=chunks,
        conversation_history=request.conversation_history
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        question=request.question
    )


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all indexed documents with chunk counts."""
    info = embedder.get_index_info()
    return [
        DocumentInfo(filename=fname, chunk_count=count)
        for fname, count in info["indexed_files"].items()
    ]