"""
Hybrid Retriever combining BM25 Keyword Search with Chroma Dense Vector Search
using Reciprocal Rank Fusion (RRF).

Significantly improves retrieval for exact terms, standards, codes, and numerical values
while maintaining strong semantic generalization.
"""

from typing import List, Dict, Any, Optional
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

class RRFHybridRetriever(BaseRetriever):
    """
    Native Reciprocal Rank Fusion (RRF) Hybrid Retriever.
    RRF score = sum(weight / (60 + rank))
    """
    bm25_retriever: Any = Field(...)
    vector_retriever: Any = Field(...)
    bm25_weight: float = Field(default=0.4)
    vector_weight: float = Field(default=0.6)
    k: int = Field(default=3)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # 1. Fetch candidates from both retrievers
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 2. Score with BM25 rankings
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content.strip()
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + (self.bm25_weight / (60 + rank))

        # 3. Score with Vector rankings
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content.strip()
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + (self.vector_weight / (60 + rank))

        # 4. Sort documents by combined RRF score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:self.k]]

def create_hybrid_retriever(
    chunks: List[Document],
    vectorstore: Chroma,
    k: int = 3,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6
) -> BaseRetriever:
    """
    Creates an RRFHybridRetriever combining BM25 keyword matching and Chroma dense semantic embeddings.
    """
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})

    if not chunks:
        return vector_retriever

    try:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = k * 2

        return RRFHybridRetriever(
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            k=k
        )
    except Exception as e:
        print(f"[HybridRetriever Notice] BM25 fallback: {e}")
        return vectorstore.as_retriever(search_kwargs={"k": k})
