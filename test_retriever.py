from src.embedder import Embedder
from src.retriever import Retriever

# Load existing index from disk
embedder = Embedder()
loaded = embedder.load_index()

if not loaded:
    print("No index found! Run test_embedder.py first.")
    exit()

print(f"Index loaded with {embedder.get_index_info()['total_chunks']} chunks\n")

# Initialize retriever
retriever = Retriever(embedder)

# Test query — change this to something relevant to your test.pdf
query = "What is this document about?"

print(f"Query: {query}\n")
results = retriever.retrieve(query, k=3)

for i, result in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(f"Score    : {result['score']:.4f}")
    print(f"Source   : {result['metadata']['source']}, Page {result['metadata']['page']}")
    print(f"Content  : {result['content'][:200]}")
    print()

# Test formatted context
print("=== Formatted Context for LLM ===")
context = retriever.format_context(results)
print(context[:500])