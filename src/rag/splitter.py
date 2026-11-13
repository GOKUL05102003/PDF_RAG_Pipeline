from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config

def split_documents(
    docs: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None
) -> List[Document]:
    """
    Breaks loaded documents into chunks while preserving paragraph boundaries
    and semantic continuity, matching Step 5 of the reference notebook.
    """
    c_size = chunk_size or config.CHUNK_SIZE
    c_overlap = chunk_overlap or config.CHUNK_OVERLAP
    seps = separators or ["\n\n", "\n", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=seps
    )

    chunks = text_splitter.split_documents(docs)
    print(f"[Splitter] Split {len(docs)} document page(s) into {len(chunks)} discrete chunks "
          f"(chunk_size={c_size}, chunk_overlap={c_overlap}).")
    return chunks

