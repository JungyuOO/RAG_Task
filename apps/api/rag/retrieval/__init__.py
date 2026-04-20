"""Retrieval-layer components for RAG."""

from apps.api.rag.retrieval.document_retriever import DocumentRetriever
from apps.api.rag.retrieval.embedding_clients import BGEOllamaEmbedder, BGETEIEmbedder
from apps.api.rag.retrieval.pgvector_bridge import PgvectorRetrievalBridge
from apps.api.rag.retrieval.pgvector_store import PgvectorIndex, PgvectorIndexRepository

__all__ = [
    "BGEOllamaEmbedder",
    "BGETEIEmbedder",
    "DocumentRetriever",
    "PgvectorIndex",
    "PgvectorIndexRepository",
    "PgvectorRetrievalBridge",
]
