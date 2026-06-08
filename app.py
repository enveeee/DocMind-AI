import streamlit as st
import os
import tempfile
import json
import time
from datetime import datetime
from src.document_processor import DocumentProcessor
from src.embedder import Embedder
from src.retriever import Retriever
from src.llm_handler import LLMHandler

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ────────────────────────────────────────────────────────────────────
def load_css(dark_mode: bool):
    bg        = "#080c14" if dark_mode else "#f4f6fa"
    bg2       = "#0d1117" if dark_mode else "#ffffff"
    bg3       = "#0a0f1a" if dark_mode else "#eef1f7"
    border    = "#1e2535" if dark_mode else "#d0d7de"
    text_main = "#e6edf3" if dark_mode else "#1a1f2e"
    text_sub  = "#8b949e" if dark_mode else "#57606a"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * {{ font-family:'Inter',sans-serif; }}
    .stApp {{ background:{bg}; }}
    .main .block-container {{ padding-top:1rem; padding-bottom:2rem; }}

    section[data-testid="stSidebar"] {{
        background:linear-gradient(180deg,{bg2} 0%,{bg3} 100%);
        border-right:1px solid {border};
    }}

    .app-header {{ text-align:center; padding:1.5rem 0 0.5rem; }}
    .app-header h1 {{
        font-size:2.2rem; font-weight:700;
        background:linear-gradient(135deg,#58a6ff,#a78bfa,#f472b6);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }}
    .app-header p {{ color:{text_sub}; font-size:0.95rem; }}

    .stats-bar {{ display:flex; gap:10px; justify-content:center; margin:0.8rem 0; flex-wrap:wrap; }}
    .stat-chip {{
        background:{bg2}; border:1px solid {border}; border-radius:20px;
        padding:5px 14px; font-size:0.78rem; color:{text_sub};
        display:flex; align-items:center; gap:5px;
    }}
    .stat-chip span {{ color:#58a6ff; font-weight:600; }}

    .landing {{
        max-width:600px; margin:3rem auto; text-align:center; padding:2rem;
        background:{bg2}; border:1px solid {border}; border-radius:20px;
    }}
    .landing h2 {{ color:{text_main}; font-size:1.6rem; margin-bottom:0.5rem; }}
    .landing p  {{ color:{text_sub}; font-size:0.95rem; margin-bottom:1.5rem; }}
    .feature-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:1rem; }}
    .feature-item {{
        background:{bg3}; border:1px solid {border}; border-radius:12px;
        padding:12px; text-align:left; font-size:0.82rem; color:{text_sub};
    }}
    .feature-item b {{ color:{text_main}; display:block; margin-bottom:3px; }}

    .user-bubble {{ display:flex; justify-content:flex-end; margin:1rem 0 0.3rem; animation:slideR .3s ease; }}
    .user-bubble-inner {{
        background:linear-gradient(135deg,#1d4ed8,#2563eb);
        border-radius:18px 18px 4px 18px;
        padding:12px 18px; max-width:70%; color:white;
        font-size:0.93rem; line-height:1.55;
        box-shadow:0 4px 15px rgba(37,99,235,.3);
    }}
    .bot-bubble {{ display:flex; justify-content:flex-start; margin:0.3rem 0; animation:slideL .3s ease; }}
    .bot-avatar {{
        width:34px; height:34px;
        background:linear-gradient(135deg,#7c3aed,#a78bfa);
        border-radius:50%; display:flex; align-items:center; justify-content:center;
        font-size:1rem; margin-right:10px; flex-shrink:0; margin-top:2px;
    }}
    .bot-bubble-inner {{
        background:{bg2}; border:1px solid {border};
        border-radius:4px 18px 18px 18px;
        padding:14px 18px; max-width:75%; color:{text_main};
        font-size:0.93rem; line-height:1.6;
        box-shadow:0 4px 15px rgba(0,0,0,.2);
    }}

    .conf-high   {{ background:#0a2a1a; color:#3fb950; border:1px solid #238636; border-radius:8px; padding:3px 10px; font-size:0.75rem; font-weight:600; display:inline-block; margin:4px 0 4px 44px; }}
    .conf-medium {{ background:#2a1f00; color:#e3b341; border:1px solid #9e6a03; border-radius:8px; padding:3px 10px; font-size:0.75rem; font-weight:600; display:inline-block; margin:4px 0 4px 44px; }}
    .conf-low    {{ background:#2a0a0a; color:#f85149; border:1px solid #da3633; border-radius:8px; padding:3px 10px; font-size:0.75rem; font-weight:600; display:inline-block; margin:4px 0 4px 44px; }}

    .sources-label {{ font-size:0.72rem; color:{text_sub}; text-transform:uppercase; letter-spacing:.05em; margin:6px 0 4px 44px; }}
    .source-card {{
        background:{bg3}; border:1px solid {border}; border-left:3px solid #58a6ff;
        border-radius:8px; padding:7px 12px; margin:3px 0 3px 44px;
        font-size:0.8rem; color:{text_sub}; display:flex; align-items:center; gap:8px;
    }}
    .score-badge {{ background:{border}; border-radius:10px; padding:2px 7px; font-size:0.73rem; color:#58a6ff; margin-left:auto; }}
    .chunk-preview {{ font-size:0.75rem; color:{text_sub}; margin:2px 0 2px 44px; font-style:italic; border-left:2px solid {border}; padding-left:8px; }}
    .low-cov {{ background:#1a1200; border:1px solid #3d2e00; border-radius:8px; padding:5px 12px; margin:3px 0 3px 44px; font-size:0.78rem; color:#e3b341; }}

    .suggestions-label {{ font-size:0.85rem; color:{text_sub}; margin:1rem 0 0.5rem; font-weight:500; }}

    .doc-card {{
        background:{bg2}; border:1px solid {border}; border-radius:10px;
        padding:10px 13px; margin:5px 0;
    }}
    .doc-name {{ color:{text_main}; font-size:0.83rem; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .doc-meta {{ color:{text_sub}; font-size:0.73rem; margin-top:2px; }}

    .empty-state {{ text-align:center; padding:3rem 2rem; color:{text_sub}; }}
    .empty-state .icon {{ font-size:3rem; margin-bottom:1rem; }}
    .empty-state h3 {{ color:{text_main}; font-size:1.2rem; }}

    @keyframes slideR {{ from{{opacity:0;transform:translateX(20px)}} to{{opacity:1;transform:translateX(0)}} }}
    @keyframes slideL {{ from{{opacity:0;transform:translateX(-20px)}} to{{opacity:1;transform:translateX(0)}} }}
    @keyframes blink  {{ 0%,100%{{opacity:1}} 50%{{opacity:0}} }}
    .typing-dot {{ display:inline-block; animation:blink 1s infinite; font-size:1.2rem; margin:0 2px; }}

    div[data-testid="stFileUploader"] {{ background:{bg2}; border:1px dashed {border}; border-radius:10px; padding:0.4rem; }}
    .stButton>button {{
        background:linear-gradient(135deg,#1d4ed8,#7c3aed) !important;
        color:white !important; border:none !important;
        border-radius:10px !important; font-weight:500 !important;
    }}
    .stButton>button:hover {{ opacity:.85 !important; }}
    div[data-testid="stTextInput"] input {{
        background:{bg2} !important; border:1px solid {border} !important;
        border-radius:12px !important; color:{text_main} !important;
        padding:12px 16px !important; font-size:0.93rem !important;
    }}
    div[data-testid="stTextInput"] input:focus {{
        border-color:#58a6ff !important;
        box-shadow:0 0 0 3px rgba(88,166,255,.1) !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "conversation_history": [],
        "dark_mode": True,
        "page": "landing",
        "active_filter": "All",
        "show_summary": {},
        "pending_question": "",
        "trigger_send": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Load Components ─────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    embedder = Embedder()
    embedder.load_index()
    retriever = Retriever(embedder)
    llm = LLMHandler()
    return embedder, retriever, llm


# ─── Helpers ─────────────────────────────────────────────────────────────────
def confidence_label(chunks_used: int, avg_score: float) -> str:
    if avg_score >= 0.35 and chunks_used >= 4:
        return "high"
    elif avg_score >= 0.20 and chunks_used >= 2:
        return "medium"
    return "low"


def export_chat_txt() -> str:
    lines = [f"DocMind AI — Chat Export\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*50}\n"]
    for i, e in enumerate(st.session_state.conversation_history, 1):
        lines.append(f"Q{i}: {e['question']}")
        lines.append(f"A{i}: {e['answer']}")
        if e.get("sources"):
            for s in e["sources"]:
                lines.append(f"    Source: {s['filename']} p.{s['page']}")
        lines.append("")
    return "\n".join(lines)


def export_chat_pdf() -> bytes:
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "DocMind AI - Chat Export", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, datetime.now().strftime("%Y-%m-%d %H:%M"), ln=True)
        pdf.ln(4)
        for i, e in enumerate(st.session_state.conversation_history, 1):
            pdf.set_font("Helvetica", "B", 10)
            q = e['question'].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 7, f"Q{i}: {q}")
            pdf.set_font("Helvetica", "", 10)
            a = e['answer'].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, f"A: {a}")
            if e.get("sources"):
                pdf.set_font("Helvetica", "I", 8)
                for s in e["sources"]:
                    pdf.cell(0, 5, f"  Source: {s['filename']} p.{s['page']}", ln=True)
            pdf.ln(3)
        return bytes(pdf.output())
    except Exception:
        return b""


def delete_document(embedder: Embedder, filename: str):
    if filename not in embedder.indexed_files:
        return
    import faiss
    import numpy as np
    new_chunks = [c for c in embedder.chunks if c.metadata.get("source") != filename]
    del embedder.indexed_files[filename]
    embedder.chunks = []
    embedder.index = None
    if new_chunks:
        texts = [c.page_content for c in new_chunks]
        embeddings = embedder.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        embedder.index = faiss.IndexFlatIP(dim)
        embedder.index.add(embeddings)
        embedder.chunks = new_chunks
    embedder._save_index()


def generate_suggestions(llm, doc_names: list) -> list:
    try:
        from langchain.schema import HumanMessage, SystemMessage
        msgs = [
            SystemMessage(content="Reply ONLY with a JSON array of exactly 3 short question strings. No explanation, no markdown, no code blocks."),
            HumanMessage(content=f"Suggest 3 interesting questions a user might ask about documents named: {', '.join(doc_names)}")
        ]
        resp = llm.llm.invoke(msgs)
        raw = resp.content.strip().replace("```json", "").replace("```", "").strip()
        suggestions = json.loads(raw)
        return suggestions[:3] if isinstance(suggestions, list) else []
    except Exception:
        return [
            "What is the main topic of this document?",
            "Summarize the key points.",
            "What are the conclusions?"
        ]


def summarize_document(llm, retriever, filename: str) -> str:
    all_chunks = [
        {"content": c.page_content, "metadata": c.metadata, "score": 1.0}
        for c in retriever.embedder.chunks
        if c.metadata.get("source") == filename
    ][:8]
    if not all_chunks:
        return "No content found for this document."
    result = llm.generate_answer(
        question="Please provide a comprehensive summary covering all main topics and key points.",
        retrieved_chunks=all_chunks
    )
    return result["answer"]


# ─── Landing Page ─────────────────────────────────────────────────────────────
def render_landing():
    st.markdown("""
    <div class="landing">
        <div style="font-size:3rem;margin-bottom:0.5rem;">🧠</div>
        <h2>Welcome to DocMind AI</h2>
        <p>Upload your PDFs and have a real conversation with them.<br>
        Get accurate answers with exact source citations.</p>
        <div class="feature-grid">
            <div class="feature-item"><b>📄 PDF Upload</b>Multiple documents, instant indexing</div>
            <div class="feature-item"><b>🔍 Semantic Search</b>FAISS vector similarity retrieval</div>
            <div class="feature-item"><b>🤖 AI Answers</b>Groq Llama 3.1 with source grounding</div>
            <div class="feature-item"><b>📎 Citations</b>Exact page references for every answer</div>
            <div class="feature-item"><b>📊 Confidence</b>High / Medium / Low answer scoring</div>
            <div class="feature-item"><b>💾 Export</b>Download chat as PDF or text</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar(embedder, retriever, llm):
    with st.sidebar:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            <div style="font-size:1.2rem;font-weight:700;
            background:linear-gradient(135deg,#58a6ff,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🧠 DocMind AI</div>
            """, unsafe_allow_html=True)
        with col2:
            mode_icon = "☀️" if st.session_state.dark_mode else "🌙"
            if st.button(mode_icon, help="Toggle theme"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()

        if st.session_state.page == "chat":
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = "landing"
                st.rerun()

        st.divider()

        # Upload
        st.markdown("**📤 Upload Documents**")
        uploaded_files = st.file_uploader(
            "PDFs", type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            for uf in uploaded_files:
                fname = uf.name
                if embedder.is_file_indexed(fname):
                    st.success(f"✅ {fname} already indexed")
                    continue
                prog = st.progress(0, text=f"Processing {fname}…")
                processor = DocumentProcessor()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    prog.progress(30, text="Extracting text…")
                    with open(tmp_path, "rb") as f:
                        result = processor.process_pdf(f, fname)
                    prog.progress(65, text="Embedding chunks…")
                    if result["status"] == "success" and result["chunk_count"] > 0:
                        embedder.embed_documents(result["chunks"], fname)
                        prog.progress(100, text="Done!")
                        st.success(f"✅ {fname} — {result['chunk_count']} chunks")
                        if st.session_state.page == "landing":
                            st.session_state.page = "chat"
                    else:
                        st.error(f"❌ Could not extract text from {fname}")
                        prog.empty()
                finally:
                    os.unlink(tmp_path)

        st.divider()

        # Document list
        info = embedder.get_index_info()
        st.markdown(f"**📚 Documents ({len(info['indexed_files'])})**")

        if info["indexed_files"]:
            doc_names = list(info["indexed_files"].keys())
            filter_options = ["All"] + doc_names
            st.session_state.active_filter = st.selectbox(
                "Filter by document",
                filter_options,
                label_visibility="collapsed"
            )

            for fname, chunk_count in info["indexed_files"].items():
                st.markdown(f"""
                <div class="doc-card">
                    <div class="doc-name">📄 {fname}</div>
                    <div class="doc-meta">{chunk_count} chunks</div>
                </div>
                """, unsafe_allow_html=True)

                col_s, col_d = st.columns(2)
                with col_s:
                    if st.button("📋 Summarize", key=f"sum_{fname}", use_container_width=True):
                        with st.spinner("Summarizing…"):
                            summary = summarize_document(llm, retriever, fname)
                            st.session_state.show_summary[fname] = summary
                with col_d:
                    if st.button("🗑️ Delete", key=f"del_{fname}", use_container_width=True):
                        delete_document(embedder, fname)
                        if fname in st.session_state.show_summary:
                            del st.session_state.show_summary[fname]
                        st.rerun()

                if fname in st.session_state.show_summary:
                    with st.expander("📋 Summary", expanded=True):
                        st.write(st.session_state.show_summary[fname])
        else:
            st.caption("No documents yet.")

        st.divider()

        st.markdown("**⚙️ Settings**")
        top_k = st.slider("Chunks to retrieve", 1, 10, 5)

        st.divider()

        st.markdown("**💾 Export Chat**")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.session_state.conversation_history:
                st.download_button(
                    "📄 TXT",
                    data=export_chat_txt(),
                    file_name=f"docmind_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        with col_e2:
            if st.session_state.conversation_history:
                pdf_bytes = export_chat_pdf()
                if pdf_bytes:
                    st.download_button(
                        "📑 PDF",
                        data=pdf_bytes,
                        file_name=f"docmind_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

        st.divider()
        st.caption("Powered by LangChain · FAISS · Groq · Streamlit")

    return top_k


# ─── Chat Page ────────────────────────────────────────────────────────────────
def render_chat(embedder, retriever, llm, top_k):
    st.markdown("""
    <div class="app-header">
        <h1>🧠 DocMind AI</h1>
        <p>Upload documents · Ask questions · Get cited answers</p>
    </div>
    """, unsafe_allow_html=True)

    info = embedder.get_index_info()
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-chip">📄 Docs <span>{len(info['indexed_files'])}</span></div>
        <div class="stat-chip">🔢 Chunks <span>{info['total_chunks']}</span></div>
        <div class="stat-chip">💬 Messages <span>{len(st.session_state.conversation_history)}</span></div>
        <div class="stat-chip">🔍 Filter <span>{st.session_state.active_filter}</span></div>
        <div class="stat-chip">🤖 <span>Llama 3.1</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if not info["indexed_files"]:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📂</div>
            <h3>No documents uploaded yet</h3>
            <p>Upload PDFs from the sidebar to begin.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Suggestions (only when no chat history)
    if not st.session_state.conversation_history:
        st.markdown('<div class="suggestions-label">💡 Try asking:</div>',
                    unsafe_allow_html=True)
        suggestions = generate_suggestions(llm, list(info["indexed_files"].keys()))
        cols = st.columns(len(suggestions))
        for i, (col, sug) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_question = sug
                    st.session_state.trigger_send = True
                    st.rerun()

    # Conversation history
    for entry in st.session_state.conversation_history:
        st.markdown(f"""
        <div class="user-bubble">
            <div class="user-bubble-inner">{entry['question']}</div>
        </div>
        """, unsafe_allow_html=True)

        avg_score = sum(s.get("score", 0) for s in entry.get("sources", [])) / max(len(entry.get("sources", [])), 1)
        conf = confidence_label(entry.get("chunks_used", 0), avg_score)
        conf_labels = {"high": "🟢 High Confidence", "medium": "🟡 Medium Confidence", "low": "🔴 Low Confidence"}
        conf_html   = {"high": "conf-high", "medium": "conf-medium", "low": "conf-low"}

        st.markdown(f"""
        <div class="bot-bubble">
            <div class="bot-avatar">🧠</div>
            <div class="bot-bubble-inner">{entry['answer']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.button("📋 Copy", key=f"copy_{entry['question'][:20]}", 
                  help=entry['answer'], use_container_width=False)

        st.markdown(f'<span class="{conf_html[conf]}">{conf_labels[conf]}</span>',
                    unsafe_allow_html=True)

        if entry.get("sources"):
            st.markdown('<div class="sources-label">📎 Sources</div>', unsafe_allow_html=True)
            for src in entry["sources"]:
                preview = src.get("preview", "")
                st.markdown(f"""
                <div class="source-card">
                    📄 <b>{src['filename']}</b> &nbsp;·&nbsp; Page {src['page']}
                    <span class="score-badge">{src['score']:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
                if preview:
                    st.markdown(f'<div class="chunk-preview">"{preview[:120]}…"</div>',
                                unsafe_allow_html=True)

        if entry.get("chunks_used", 5) < 3:
            st.markdown('<div class="low-cov">⚠️ Low source coverage — answer may be incomplete</div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Input area ──────────────────────────────────────────────────────────
    st.divider()
    prefill = st.session_state.pending_question

    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "q",
            value=prefill,
            placeholder="Ask a question about your documents…",
            label_visibility="collapsed",
            key="chat_input"
        )
    with col2:
        send_clicked = st.button("Send ➤", use_container_width=True, key="send_btn")

    should_send = send_clicked or st.session_state.trigger_send
    question_to_ask = user_input.strip() or prefill.strip()

    if should_send and question_to_ask:
        st.session_state.pending_question = ""
        st.session_state.trigger_send = False

        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="bot-bubble">
            <div class="bot-avatar">🧠</div>
            <div class="bot-bubble-inner">
                <span class="typing-dot">●</span>
                <span class="typing-dot" style="animation-delay:.2s">●</span>
                <span class="typing-dot" style="animation-delay:.4s">●</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        active_filter = st.session_state.active_filter
        chunks = retriever.retrieve(question_to_ask, k=top_k)
        if active_filter != "All":
            filtered = [c for c in chunks if c["metadata"].get("source") == active_filter]
            chunks = filtered if filtered else chunks

        result = llm.generate_answer(
            question=question_to_ask,
            retrieved_chunks=chunks,
            conversation_history=st.session_state.conversation_history
        )

        chunk_map = {
            c["metadata"].get("source", "") + str(c["metadata"].get("page", "")): c["content"]
            for c in chunks
        }
        for src in result["sources"]:
            src["preview"] = chunk_map.get(src["filename"] + str(src["page"]), "")

        typing_placeholder.empty()

        st.session_state.conversation_history.append({
            "question": question_to_ask,
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks_used": result["chunks_used"]
        })
        if len(st.session_state.conversation_history) > 5:
            st.session_state.conversation_history = \
                st.session_state.conversation_history[-5:]

        st.rerun()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    embedder, retriever, llm = load_components()
    load_css(st.session_state.dark_mode)
    top_k = render_sidebar(embedder, retriever, llm)

    if st.session_state.page == "landing":
        render_landing()
    else:
        render_chat(embedder, retriever, llm, top_k)


if __name__ == "__main__":
    main()