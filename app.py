"""
Voice-Interactive Document RAG Assistant with Hybrid Search (BM25 + ChromaDB)
=============================================================================
Professional Streamlit Application featuring STT -> LLM -> TTS with Dual-Mode Output.
"""

import os
import sys
import re
import time
import tempfile
from pathlib import Path
import streamlit as st

# Set page layout and config first
st.set_page_config(
    page_title="Voice RAG Intelligence | PDF Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Modern Glassmorphic / Enterprise Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0px;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #38BDF8;
        margin-right: 8px;
    }
    .status-badge-green {
        background: rgba(52, 211, 153, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.3);
        color: #34D399;
    }

    /* Citation Card */
    .citation-card {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #818CF8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.94rem;
        line-height: 1.5;
        color: #E2E8F0;
    }
    .citation-header {
        font-weight: 600;
        color: #A5B4FC;
        margin-bottom: 6px;
    }

    /* Metric Box */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Modern Speak Button on the Right */
    div[data-testid="stButton"] button {
        height: 48px;
        margin-top: 0px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF !important;
        border: none;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45);
    }
</style>
""", unsafe_allow_html=True)

# Imports from modular src package
from src import config
from src.audio.stt import transcribe_from_microphone
from src.audio.tts import synthesize_audio_file
from src.rag.loader import load_pdf
from src.rag.splitter import split_documents
from src.rag.vectorstore import (
    build_vectorstore,
    get_or_create_vectorstore,
    reset_or_rebuild_vectorstore,
    delete_collection
)
from src.rag.chain import get_llm
from src.rag.hybrid_retriever import create_hybrid_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Working directory for audio caching
TEMP_DIR = Path(tempfile.gettempdir()) / "rag_audio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs" not in st.session_state:
    st.session_state.docs = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_pdf" not in st.session_state:
    default_path = config.DEFAULT_PDF_PATH
    st.session_state.current_pdf = str(default_path) if default_path.exists() else None
if "indexing_done" not in st.session_state:
    st.session_state.indexing_done = False


# =====================================================================
# Document Ingestion Engine with Self-Healing Vector Index
# =====================================================================
def ensure_indexed(force_rebuild: bool = False):
    """Loads PDF, splits chunks, and sets up vector store & hybrid retriever."""
    if not st.session_state.current_pdf or not Path(st.session_state.current_pdf).exists():
        st.error("No active PDF found. Please upload a document.")
        return False

    if not force_rebuild and st.session_state.indexing_done and st.session_state.vectorstore is not None and st.session_state.chunks is not None:
        return True

    with st.spinner("Extracting pages, generating semantic chunks, and loading vector index..."):
        try:
            # 1. Load document
            st.session_state.docs = load_pdf(st.session_state.current_pdf)
            
            # 2. Chunk document
            st.session_state.chunks = split_documents(st.session_state.docs)

            # 3. Vector Store with Segment Validation
            safe_coll = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(st.session_state.current_pdf).stem)[:60]
            if not safe_coll:
                safe_coll = "pdf_rag_collection"

            need_build = force_rebuild
            if not force_rebuild:
                try:
                    vstore = get_or_create_vectorstore(collection_name=safe_coll)
                    count = vstore._collection.count()
                    if count >= len(st.session_state.chunks):
                        st.session_state.vectorstore = vstore
                    else:
                        need_build = True
                except Exception as check_err:
                    print(f"[VectorStore] Existing index check failed: {check_err}. Rebuilding...")
                    need_build = True

            if need_build:
                st.session_state.vectorstore = reset_or_rebuild_vectorstore(
                    st.session_state.chunks,
                    collection_name=safe_coll,
                    batch_size=5,
                    batch_delay_seconds=2.0
                )

            st.session_state.indexing_done = True
            return True
        except Exception as e:
            st.error(f"Error during document indexing: {e}")
            return False


# Auto-index default document on first startup
if not st.session_state.indexing_done:
    ensure_indexed()


# =====================================================================
# Sidebar: Controls & Document Management
# =====================================================================
with st.sidebar:
    # Dedicated RAG System Branding Header (replaces brain logo)
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 12px 14px; background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; margin-bottom: 20px;">
        <div style="background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%); width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35); flex-shrink: 0;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 4C4 2.89543 4.89543 2 6 2H14L20 8V20C20 21.1046 19.1046 22 18 22H6C4.89543 22 4 21.1046 4 20V4Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <polyline points="14 2 14 8 20 8" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="11" cy="14" r="3" stroke="#38BDF8" stroke-width="2"/>
                <path d="M13.2 16.2L16 19" stroke="#38BDF8" stroke-width="2" stroke-linecap="round"/>
            </svg>
        </div>
        <div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.01em; line-height: 1.2;">RAG System</div>
            <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 3px;">Hybrid Knowledge</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📄 Document Repository")
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload any PDF report to extract and query on the fly"
    )
    
    if uploaded_file is not None:
        save_path = config.DATA_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.session_state.current_pdf != str(save_path):
            st.session_state.current_pdf = str(save_path)
            st.session_state.indexing_done = False
            st.session_state.vectorstore = None
            st.session_state.chunks = None
            st.session_state.docs = None
            st.success(f"Uploaded: {uploaded_file.name}")
            ensure_indexed()
            st.rerun()
    
    active_doc_name = Path(st.session_state.current_pdf).name if st.session_state.current_pdf else "No document loaded"
    st.info(f"**Active Document:**\n`{active_doc_name}`")
    
    st.divider()
    
    st.subheader("⚡ Retrieval Architecture")
    search_mode = st.radio(
        "Search Engine",
        options=["Hybrid Search (BM25 + ChromaDB)", "Dense Vector Only"],
        index=0,
        help="Hybrid Search combines lexical keyword matching (BM25) with semantic embeddings (ChromaDB)"
    )
    is_hybrid = ("Hybrid" in search_mode)
    
    if is_hybrid:
        bm25_weight = st.slider("BM25 Keyword Weight", 0.1, 0.9, 0.4, 0.05, help="Higher = stricter exact keyword matching")
        vector_weight = round(1.0 - bm25_weight, 2)
        st.caption(f"Vector Semantic Weight: **{vector_weight}**")
    else:
        bm25_weight, vector_weight = 0.0, 1.0

    retriever_k = st.slider("Top-K Chunks Retrieved", min_value=1, max_value=8, value=config.RETRIEVER_K)

    if st.button("🔄 Rebuild Vector Index", use_container_width=True, help="Force clean rebuild of ChromaDB index"):
        ensure_indexed(force_rebuild=True)
        st.success("Vector index successfully rebuilt!")
        st.rerun()

    st.divider()

    st.subheader("🎙️ Voice & Audio Settings")
    auto_speak = st.checkbox("Auto-play Answer Audio (TTS)", value=True)

    st.divider()
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =====================================================================
# Main Header & Metrics
# =====================================================================
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">Voice Document RAG Intelligence</h1>
            <p class="hero-subtitle">Bidirectional Multimodal Assistant: Speech-to-Text &rarr; Hybrid RAG &rarr; Text-to-Speech</p>
        </div>
        <div style="margin-top: 8px;">
            <span class="status-badge status-badge-green">● Gemini 3.5 Flash Lite</span>
            <span class="status-badge">● Hybrid Search (RRF)</span>
            <span class="status-badge">● Native STT/TTS</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Overview Metrics Row
total_pages = len(st.session_state.docs) if st.session_state.docs else 0
total_chunks = len(st.session_state.chunks) if st.session_state.chunks else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{total_pages}</div><div class="metric-lbl">Total Pages</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{total_chunks}</div><div class="metric-lbl">Indexed Chunks</div></div>', unsafe_allow_html=True)
with c3:
    mode_label = "Hybrid RRF" if is_hybrid else "Dense Vector"
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.2rem;">{mode_label}</div><div class="metric-lbl">Retrieval Mode</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.2rem;">Top-{retriever_k}</div><div class="metric-lbl">Context Depth</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =====================================================================
# Chat Interface: Render Full-Width Conversation History
# =====================================================================
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            
            # Full-width Audio Player for vocalized final answer
            if msg["role"] == "assistant" and msg.get("audio_file") and Path(msg["audio_file"]).exists():
                st.audio(msg["audio_file"])

            # Full-width Source Citations Accordion
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander(f"📚 View {len(msg['citations'])} Verified Source Chunks", expanded=False):
                    for c_idx, cit in enumerate(msg["citations"], 1):
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-header">Chunk #{c_idx} • Page {cit.get('page', 'Unknown')}</div>
                            <div>{cit.get('text', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)


# =====================================================================
# Query Execution Logic with Segment Auto-Repair
# =====================================================================
def process_query(query_text: str):
    """Executes the complete RAG pipeline with full-width response delivery."""
    if not query_text.strip():
        return

    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": query_text})
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(query_text)

    # 2. Ensure document is indexed
    if not ensure_indexed():
        return

    # 3. Generate assistant response
    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching document context and generating answer..."):
                try:
                    # Setup Retriever (Hybrid RRF or Vector)
                    if is_hybrid and st.session_state.chunks:
                        retriever = create_hybrid_retriever(
                            st.session_state.chunks,
                            st.session_state.vectorstore,
                            k=retriever_k,
                            bm25_weight=bm25_weight,
                            vector_weight=vector_weight
                        )
                    else:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": retriever_k})

                    # Retrieve chunks with automatic HNSW disk repair
                    try:
                        retrieved_docs = retriever.invoke(query_text)
                    except Exception as ret_err:
                        err_str = str(ret_err)
                        if "hnsw" in err_str.lower() or "nothing found on disk" in err_str.lower() or "invalidargumenterror" in err_str.lower():
                            safe_coll = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(st.session_state.current_pdf).stem)[:60] or "pdf_rag_collection"
                            st.session_state.vectorstore = reset_or_rebuild_vectorstore(
                                st.session_state.chunks,
                                collection_name=safe_coll,
                                batch_size=5,
                                batch_delay_seconds=2.0
                            )
                            if is_hybrid and st.session_state.chunks:
                                retriever = create_hybrid_retriever(
                                    st.session_state.chunks,
                                    st.session_state.vectorstore,
                                    k=retriever_k,
                                    bm25_weight=bm25_weight,
                                    vector_weight=vector_weight
                                )
                            else:
                                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": retriever_k})
                            retrieved_docs = retriever.invoke(query_text)
                        else:
                            raise ret_err
                    
                    # Format context for prompt
                    formatted_context = "\n---\n".join(d.page_content for d in retrieved_docs)
                    
                    # Run LLM chain
                    prompt_input = {"context": formatted_context, "question": query_text}
                    prompt_template = ChatPromptTemplate.from_template(config.DEFAULT_PROMPT_TEMPLATE)
                    llm = get_llm()
                    chain = prompt_template | llm | StrOutputParser()
                    
                    answer = chain.invoke(prompt_input).strip()
                    st.markdown(answer)

                    # Synthesize TTS Audio for Final Answer Only
                    audio_path = str(TEMP_DIR / f"response_{int(time.time())}.mp3")
                    saved_audio = synthesize_audio_file(answer, audio_path)
                    
                    if saved_audio and Path(saved_audio).exists():
                        st.audio(saved_audio, autoplay=auto_speak)

                    # Build citations
                    citations = []
                    for doc in retrieved_docs:
                        citations.append({
                            "page": doc.metadata.get("page", "N/A"),
                            "source": Path(doc.metadata.get("source", "")).name,
                            "text": doc.page_content.strip()
                        })

                    if citations:
                        with st.expander(f"📚 View {len(citations)} Verified Source Chunks", expanded=True):
                            for c_idx, cit in enumerate(citations, 1):
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">Chunk #{c_idx} • Page {cit['page']}</div>
                                    <div>{cit['text']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Store in session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "audio_file": saved_audio if saved_audio else None,
                        "citations": citations
                    })

                except Exception as err:
                    st.error(f"Error generating answer: {err}")


# =====================================================================
# Dual Input Area: Question Input (LEFT) + Speak Button (RIGHT)
# =====================================================================
input_col_text, input_col_speak = st.columns([6, 1], vertical_alignment="bottom")

query_to_submit = None

with input_col_text:
    typed_input = st.chat_input("Ask any question about the document (or click 🎙️ Speak on the right)...")
    if typed_input:
        query_to_submit = typed_input

with input_col_speak:
    btn_container = st.empty()
    # Check if Speak button is pressed
    if btn_container.button("🎙️ Speak", use_container_width=True, help="Record question from your microphone", key="speak_btn"):
        # Replace the button label directly with "Listening..."
        btn_container.button("🔴 Listening...", disabled=True, use_container_width=True, key="listening_btn")
        # Transcribe directly without any banner or spinner
        spoken = transcribe_from_microphone(timeout=6, phrase_time_limit=10)
        # Restore button back to normal state
        btn_container.button("🎙️ Speak", use_container_width=True, help="Record question from your microphone", key="speak_btn_reset")
        if spoken:
            query_to_submit = spoken
        else:
            st.toast("No speech detected. Please speak clearly or type your question.")

# Process query in the full-width main container
if query_to_submit:
    process_query(query_to_submit)
