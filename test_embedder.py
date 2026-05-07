from src.document_processor import DocumentProcessor
from src.embedder import Embedder

# Step 1: Process the PDF
processor = DocumentProcessor()
with open("sample_docs/test.pdf", "rb") as f:
    result = processor.process_pdf(f, "test.pdf")

print(f"Chunks created: {result['chunk_count']}")

# Step 2: Embed and index
embedder = Embedder()
embedder.embed_documents(result["chunks"], "test.pdf")

# Step 3: Check index info
info = embedder.get_index_info()
print(f"Total chunks in index : {info['total_chunks']}")
print(f"Indexed files         : {info['indexed_files']}")
print(f"Index loaded          : {info['index_loaded']}")

# Step 4: Test cache check
print(f"Is test.pdf indexed?  : {embedder.is_file_indexed('test.pdf')}")

# Step 5: Test reload from disk
embedder2 = Embedder()
loaded = embedder2.load_index()
print(f"Reloaded from disk    : {loaded}")
print(f"Chunks after reload   : {embedder2.get_index_info()['total_chunks']}")