"""
Hybrid Retrieval Layer
- Vector Search (semantic similarity via ChromaDB)
- Keyword Search (BM25 for exact legal terminology)
- Metadata filtering support
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    document: Document
    score: float
    source: str  # "vector" or "bm25" or "hybrid"


class HybridRetriever:
    """
    Hybrid retrieval combining:
    1. Vector search (semantic similarity)
    2. BM25 search (keyword matching for legal terms)
    
    Results are merged using Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "legal_documents",
        openai_api_key: Optional[str] = None
    ):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        self.vector_store: Optional[Chroma] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self.tokenized_docs: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        return text.lower().split()
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents in both vector store and BM25
        """
        self.documents = documents
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        # Create BM25 index
        self.tokenized_docs = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        self.bm25_index = BM25Okapi(self.tokenized_docs)
        
        print(f"✓ Indexed {len(documents)} documents in vector store and BM25")
    
    def load_existing_index(self) -> None:
        """Load existing vector store from disk"""
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Retrieve all documents for BM25
        results = self.vector_store.get(include=["documents", "metadatas"])
        self.documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        
        # Rebuild BM25 index
        self.tokenized_docs = [
            self._tokenize(doc.page_content) for doc in self.documents
        ]
        self.bm25_index = BM25Okapi(self.tokenized_docs)
    
    def vector_search(
        self,
        query: str,
        k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Perform semantic vector search
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call index_documents first.")
        
        # Search with optional metadata filter
        if filter_dict:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k, filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k
            )
        
        return [
            RetrievalResult(document=doc, score=score, source="vector")
            for doc, score in results
        ]
    
    def bm25_search(
        self,
        query: str,
        k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Perform BM25 keyword search
        """
        if not self.bm25_index:
            raise ValueError("BM25 index not initialized. Call index_documents first.")
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            
            # Apply metadata filter if provided
            if filter_dict:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_dict.items()
                )
                if not match:
                    continue
            
            results.append(RetrievalResult(
                document=doc,
                score=float(scores[idx]),
                source="bm25"
            ))
        
        return results[:k]
    
    def reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k: int = 60,
        weights: Tuple[float, float] = (0.5, 0.5)
    ) -> List[RetrievalResult]:
        """
        Merge results using Reciprocal Rank Fusion (RRF)
        
        RRF score = sum(1 / (k + rank)) for each result list
        """
        # Create document ID to result mapping
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result.document.metadata.get("chunk_id", str(hash(result.document.page_content)))
            rrf_score = weights[0] * (1 / (k + rank + 1))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = result.document
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result.document.metadata.get("chunk_id", str(hash(result.document.page_content)))
            rrf_score = weights[1] * (1 / (k + rank + 1))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = result.document
        
        # Sort by combined RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            RetrievalResult(
                document=doc_map[doc_id],
                score=score,
                source="hybrid"
            )
            for doc_id, score in sorted_docs
        ]
    
    def hybrid_search(
        self,
        query: str,
        k: int = 20,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search combining vector and BM25
        """
        # Get results from both methods
        vector_results = self.vector_search(query, k=k, filter_dict=filter_dict)
        bm25_results = self.bm25_search(query, k=k, filter_dict=filter_dict)
        
        # Merge using RRF
        hybrid_results = self.reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            weights=(vector_weight, bm25_weight)
        )
        
        return hybrid_results[:k]


# Factory function
def create_hybrid_retriever(
    embedding_model: str = "text-embedding-3-small",
    persist_directory: str = "./data/chroma_db",
    openai_api_key: Optional[str] = None
) -> HybridRetriever:
    """Factory function to create hybrid retriever"""
    return HybridRetriever(
        embedding_model=embedding_model,
        persist_directory=persist_directory,
        openai_api_key=openai_api_key
    )
