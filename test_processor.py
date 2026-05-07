from src.document_processor import DocumentProcessor

processor = DocumentProcessor()

# Test with any real PDF you have on your laptop
# Replace the path below with an actual PDF file path
pdf_path = "sample_docs/test.pdf"

with open(pdf_path, "rb") as f:
    result = processor.process_pdf(f, "test.pdf")

print(f"Status     : {result['status']}")
print(f"Pages      : {result['pages']}")
print(f"Chunks     : {result['chunk_count']}")
print(f"First chunk: {result['chunks'][0].page_content[:200]}")
print(f"Metadata   : {result['chunks'][0].metadata}")