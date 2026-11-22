import os
import time
from pathlib import Path
from typing import List, Optional
import chromadb
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src import config

def get_embedding_function(api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """Initializes Google Gemini Embeddings using langchain-google-genai."""
    key = api_key or config.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Google Gemini API Key is missing. Please set GEMINI_API_KEY in .env.")

    model_name = os.getenv("EMBEDDING_MODEL", config.EMBEDDING_MODEL)
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=key
    )

def delete_collection(
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None
) -> bool:
    """Safely deletes a collection from Chroma persistent storage."""
    c_name = collection_name or config.COLLECTION_NAME
    p_dir = persist_directory or str(config.CHROMA_DIR)
    try:
        client = chromadb.PersistentClient(path=p_dir)
        client.delete_collection(name=c_name)
        print(f"[VectorStore] Successfully deleted collection '{c_name}'.")
        return True
    except Exception as e:
        print(f"[VectorStore] Notice: Collection '{c_name}' could not be deleted or does not exist: {e}")
        return False

def build_vectorstore(
    chunks: List[Document],
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    batch_size: int = 5,
    batch_delay_seconds: float = 2.0,
    api_key: Optional[str] = None
) -> Chroma:
    """
    Creates or updates a Chroma vector store by embedding document chunks
    in throttled batches to respect API rate limits.
    """
    embeddings = get_embedding_function(api_key=api_key)
    c_name = collection_name or config.COLLECTION_NAME
    p_dir = persist_directory or str(config.CHROMA_DIR)

    print(f"\n[VectorStore] Initializing Chroma collection '{c_name}'...")
    vectorstore = Chroma(
        collection_name=c_name,
        embedding_function=embeddings,
        persist_directory=p_dir
    )

    total_chunks = len(chunks)
    print(f"[VectorStore] Embedding and inserting {total_chunks} chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        current_progress = min(i + batch_size, total_chunks)
        print(f"  - Successfully indexed chunks {i} to {current_progress} / {total_chunks}")

        if current_progress < total_chunks and batch_delay_seconds > 0:
            time.sleep(batch_delay_seconds)

    print(f"[VectorStore] All {total_chunks} chunks successfully embedded and stored in Chroma!\n")
    return vectorstore

def reset_or_rebuild_vectorstore(
    chunks: List[Document],
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    batch_size: int = 5,
    batch_delay_seconds: float = 2.0,
    api_key: Optional[str] = None
) -> Chroma:
    """Deletes existing collection if corrupted and rebuilds fresh."""
    c_name = collection_name or config.COLLECTION_NAME
    delete_collection(collection_name=c_name, persist_directory=persist_directory)
    return build_vectorstore(
        chunks=chunks,
        collection_name=c_name,
        persist_directory=persist_directory,
        batch_size=batch_size,
        batch_delay_seconds=batch_delay_seconds,
        api_key=api_key
    )

def get_or_create_vectorstore(
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    api_key: Optional[str] = None
) -> Chroma:
    """Loads an existing Chroma vector store without re-indexing."""
    embeddings = get_embedding_function(api_key=api_key)
    c_name = collection_name or config.COLLECTION_NAME
    p_dir = persist_directory or str(config.CHROMA_DIR)

    return Chroma(
        collection_name=c_name,
        embedding_function=embeddings,
        persist_directory=p_dir
    )
