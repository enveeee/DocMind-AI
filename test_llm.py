from src.embedder import Embedder
from src.retriever import Retriever
from src.llm_handler import LLMHandler

# Load index
embedder = Embedder()
embedder.load_index()

retriever = Retriever(embedder)
llm = LLMHandler()

# Ask a question relevant to your test.pdf
question = "What topics are covered in this document?"

# Retrieve chunks
chunks = retriever.retrieve(question, k=5)
print(f"Retrieved {len(chunks)} chunks\n")

# Generate answer
result = llm.generate_answer(question, chunks)

print(f"Answer:\n{result['answer']}\n")
print("Sources:")
for s in result["sources"]:
    print(f"  - {s['filename']}, Page {s['page']} (score: {s['score']:.4f})")
print(f"\nChunks used: {result['chunks_used']}")