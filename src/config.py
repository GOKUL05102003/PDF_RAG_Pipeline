import os
from pathlib import Path
from dotenv import load_dotenv

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# API Configuration (Single API Key for all Google/LangChain tools)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Document Configuration
DEFAULT_PDF_PATH = DATA_DIR / "iPhone_16_Pro_and_iPhone_16_Pro_Max_PER_Sept2024.pdf"

# RAG & Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_rag_store")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))

# Voice Configuration
TTS_RATE = int(os.getenv("TTS_RATE", "175"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))

# Default RAG Prompt Template matching the original notebook
DEFAULT_PROMPT_TEMPLATE = """You are a precise document assistant. Use only the provided document context below to answer the user's question accurately. If you do not know the answer or if it is not explicitly mentioned in the context, say that you cannot find it.

Context:
{context}

Question: {question}"""

