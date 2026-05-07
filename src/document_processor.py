import pdfplumber
from pypdf import PdfReader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, file, filename: str) -> list[tuple[int, str]]:
        """
        Extract text from a PDF file page by page.
        Returns a list of (page_number, text) tuples.
        Tries pdfplumber first, falls back to pypdf.
        """
        pages = []

        try:
            logger.info(f"Extracting text from {filename} using pdfplumber...")
            with pdfplumber.open(file) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append((page_num, text.strip()))
                    else:
                        pages.append((page_num, ""))

            # Check if pdfplumber got meaningful text
            total_text = " ".join([t for _, t in pages])
            if len(total_text.strip()) < 50:
                raise ValueError("pdfplumber extracted insufficient text, trying fallback...")

            logger.info(f"pdfplumber extracted {len(pages)} pages from {filename}")
            return pages

        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}. Falling back to pypdf...")
            pages = []

            try:
                # Reset file pointer if possible
                if hasattr(file, 'seek'):
                    file.seek(0)

                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append((page_num, text.strip()))
                    else:
                        pages.append((page_num, ""))

                logger.info(f"pypdf extracted {len(pages)} pages from {filename}")
                return pages

            except Exception as e2:
                logger.error(f"Both extractors failed for {filename}: {e2}")
                return []

    def chunk_text(self, pages: list[tuple[int, str]], filename: str) -> list[Document]:
        """
        Split extracted pages into chunks.
        Each chunk is a LangChain Document with metadata:
        - source: filename
        - page: page number
        - chunk_index: position of chunk within the document
        - timestamp: when it was processed
        """
        all_chunks = []
        chunk_index = 0
        timestamp = datetime.now().isoformat()

        for page_num, text in pages:
            if not text.strip():
                continue

            # Split the page text into chunks
            splits = self.text_splitter.split_text(text)

            for split in splits:
                if split.strip():
                    doc = Document(
                        page_content=split.strip(),
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "timestamp": timestamp
                        }
                    )
                    all_chunks.append(doc)
                    chunk_index += 1

        logger.info(f"Created {len(all_chunks)} chunks from {filename}")
        return all_chunks

    def process_pdf(self, file, filename: str) -> dict:
        """
        Full pipeline: extract + chunk a PDF.
        Returns a dict with chunks and metadata summary.
        """
        pages = self.extract_text_from_pdf(file, filename)

        if not pages:
            return {
                "filename": filename,
                "pages": 0,
                "chunks": [],
                "chunk_count": 0,
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }

        chunks = self.chunk_text(pages, filename)
        non_empty_pages = sum(1 for _, t in pages if t.strip())

        return {
            "filename": filename,
            "pages": len(pages),
            "non_empty_pages": non_empty_pages,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }