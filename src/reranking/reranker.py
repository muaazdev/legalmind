"""
Re-ranking Layer
Uses Cross-Encoder models to refine retrieval results
- Cohere Rerank (primary)
- Sentence Transformers Cross-Encoder (fallback)
"""
from typing import List, Optional
from abc import ABC, abstractmethod
import cohere

from langchain.schema import Document


class BaseReranker(ABC):
    """Abstract base class for rerankers"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """Rerank documents based on query relevance"""
        pass


class CohereReranker(BaseReranker):
    """
    Cohere Rerank API for high-quality reranking
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0"
    ):
        self.client = cohere.Client(api_key)
        self.model = model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Rerank documents using Cohere's cross-encoder model
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return
        
        Returns:
            Reranked list of documents (top_k)
        """
        if not documents:
            return []
        
        # Extract text content for reranking
        doc_texts = [doc.page_content for doc in documents]
        
        # Call Cohere Rerank API
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=doc_texts,
            top_n=top_k,
            return_documents=False
        )
        
        # Reorder documents based on rerank results
        reranked_docs = []
        for result in response.results:
            doc = documents[result.index]
            # Add rerank score to metadata
            doc.metadata["rerank_score"] = result.relevance_score
            reranked_docs.append(doc)
        
        return reranked_docs


class CrossEncoderReranker(BaseReranker):
    """
    Fallback reranker using sentence-transformers Cross-Encoder
    (No API key required)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Rerank using local Cross-Encoder model
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Add scores to metadata and return top_k
        reranked_docs = []
        for doc, score in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            reranked_docs.append(doc)
        
        return reranked_docs


class RerankerPipeline:
    """
    Unified reranking pipeline with fallback support
    """
    
    def __init__(
        self,
        cohere_api_key: Optional[str] = None,
        cohere_model: str = "rerank-english-v3.0",
        use_fallback: bool = True
    ):
        self.primary_reranker: Optional[BaseReranker] = None
        self.fallback_reranker: Optional[BaseReranker] = None
        
        # Initialize Cohere if API key provided
        if cohere_api_key:
            self.primary_reranker = CohereReranker(
                api_key=cohere_api_key,
                model=cohere_model
            )
        
        # Initialize fallback if requested
        if use_fallback and not self.primary_reranker:
            self.fallback_reranker = CrossEncoderReranker()
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Rerank documents using primary or fallback reranker
        """
        reranker = self.primary_reranker or self.fallback_reranker
        
        if not reranker:
            # No reranker available, return original order
            return documents[:top_k]
        
        try:
            return reranker.rerank(query, documents, top_k)
        except Exception as e:
            print(f"Primary reranker failed: {e}")
            if self.fallback_reranker and reranker != self.fallback_reranker:
                print("Falling back to Cross-Encoder reranker...")
                return self.fallback_reranker.rerank(query, documents, top_k)
            return documents[:top_k]


# Factory function
def create_reranker(
    cohere_api_key: Optional[str] = None,
    model: str = "rerank-english-v3.0"
) -> RerankerPipeline:
    """Factory function to create reranker pipeline"""
    return RerankerPipeline(
        cohere_api_key=cohere_api_key,
        cohere_model=model,
        use_fallback=True
    )
