import streamlit as st
import os
import tempfile
from src.document_processor import DocumentProcessor
from src.embedder import Embedder
from src.retriever import Retriever
from src.llm_handler import LLMHandler

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }

    .user-message {
        background-color: #1e3a5f;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        color: white;
        text-align: right;
    }
    .assistant-message {
        background-color: #1a1f2e;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 20%;
        color: #e0e0e0;
        border-left: 3px solid #4a9eff;
    }
    .source-box {
        background-color: #12161f;
        border-radius: 8px;
        padding: 8px 12px;
        margin-top: 8px;
        font-size: 0.85em;
        color: #888;
        border: 1px solid #2a2f3e;
    }
    .warning-box {
        background-color: #3d2b00;
        border-radius: 8px;
        padding: 8px 12px;
        color: #ffaa00;
        font-size: 0.85em;
        margin-top: 6px;
    }
    .doc-badge {
        background-color: #1e3a2f;
        border-radius: 6px;
        padding: 4px 10px;
        color: #4caf50;
        font-size: 0.85em;
        margin: 3px 0;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ──────────────────────────────────────────────────────
def init_session_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "embedder" not in st.session_state:
        st.session_state.embedder = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = {}


# ─── Initialize Components ───────────────────────────────────────────────────
@st.cache_resource
def load_components():
    embedder = Embedder()
    existing = embedder.load_index()
    retriever = Retriever(embedder)
    llm = LLMHandler()
    return embedder, retriever, llm


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar(embedder, retriever, llm):
    with st.sidebar:
        st.title("📄 RAG Document Q&A")
        st.markdown("---")

        # Upload section
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to query"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name

                if embedder.is_file_indexed(filename):
                    st.success(f"✅ {filename} already indexed")
                    continue

                with st.spinner(f"Processing {filename}..."):
                    processor = DocumentProcessor()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        with open(tmp_path, "rb") as f:
                            result = processor.process_pdf(f, filename)

                        if result["status"] == "success" and result["chunk_count"] > 0:
                            embedder.embed_documents(result["chunks"], filename)
                            st.session_state.indexed_docs[filename] = {
                                "pages": result["pages"],
                                "chunks": result["chunk_count"]
                            }
                            st.success(f"✅ {filename} indexed! ({result['chunk_count']} chunks)")
                        else:
                            st.error(f"❌ Failed to process {filename}")
                    finally:
                        os.unlink(tmp_path)

        st.markdown("---")

        # Indexed documents list
        st.subheader("📚 Indexed Documents")
        index_info = embedder.get_index_info()

        if index_info["indexed_files"]:
            for fname, chunk_count in index_info["indexed_files"].items():
                st.markdown(f"""
                <div class="doc-badge">
                    📄 {fname}<br>
                    <small>{chunk_count} chunks</small>
                </div>
                """, unsafe_allow_html=True)
            st.caption(f"Total chunks: {index_info['total_chunks']}")
        else:
            st.info("No documents indexed yet. Upload a PDF to get started.")

        st.markdown("---")

        # Settings
        st.subheader("⚙️ Settings")
        top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)

        st.markdown("---")

        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

        st.markdown("---")
        st.caption("Built with LangChain · FAISS · Groq · Streamlit")

    return top_k


# ─── Chat Interface ───────────────────────────────────────────────────────────
def render_chat(retriever, llm, top_k):
    st.title("💬 Ask Your Documents")

    index_info = retriever.embedder.get_index_info()
    if not index_info["indexed_files"]:
        st.info("👈 Upload a PDF document from the sidebar to get started.")
        return

    # Display conversation history
    for entry in st.session_state.conversation_history:
        # User message
        st.markdown(f"""
        <div class="user-message">
            🧑 {entry['question']}
        </div>
        """, unsafe_allow_html=True)

        # Assistant message
        st.markdown(f"""
        <div class="assistant-message">
            🤖 {entry['answer']}
        </div>
        """, unsafe_allow_html=True)

        # Sources
        if entry.get("sources"):
            with st.expander("📎 Sources Used", expanded=False):
                for src in entry["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        📄 <b>{src['filename']}</b> — Page {src['page']}
                        &nbsp;&nbsp;|&nbsp;&nbsp; Score: {src['score']:.4f}
                    </div>
                    """, unsafe_allow_html=True)

        # Low coverage warning
        if entry.get("chunks_used", 5) < 3:
            st.markdown("""
            <div class="warning-box">
                ⚠️ Low source coverage — answer may be incomplete
            </div>
            """, unsafe_allow_html=True)

    # Query input
    st.markdown("---")
    with st.form(key="query_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_question = st.text_input(
                "Ask a question about your documents",
                placeholder="e.g. What are the main topics covered?",
                label_visibility="collapsed"
            )
        with col2:
            submit = st.form_submit_button("Send 🚀", use_container_width=True)

    if submit and user_question.strip():
        with st.spinner("Thinking..."):
            # Retrieve chunks
            chunks = retriever.retrieve(user_question, k=top_k)

            # Generate answer
            result = llm.generate_answer(
                question=user_question,
                retrieved_chunks=chunks,
                conversation_history=st.session_state.conversation_history
            )

            # Save to history
            st.session_state.conversation_history.append({
                "question": user_question,
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_used": result["chunks_used"]
            })

            # Keep only last 5 exchanges
            if len(st.session_state.conversation_history) > 5:
                st.session_state.conversation_history = \
                    st.session_state.conversation_history[-5:]

        st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    init_session_state()

    with st.spinner("Loading AI components..."):
        embedder, retriever, llm = load_components()

    top_k = render_sidebar(embedder, retriever, llm)
    render_chat(retriever, llm, top_k)


if __name__ == "__main__":
    main()