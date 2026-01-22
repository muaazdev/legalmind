"""
LegalMind - Main Orchestrator
Ties together all RAG components:
- Ingestion Pipeline
- Hybrid Retrieval
- Reranking
- Generation with Citations
- Evaluation Agents
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os

from langchain_core.documents import Document

# Import components
from src.ingestion.pipeline import DocumentIngestionPipeline, create_ingestion_pipeline
from src.retrieval.hybrid_retriever import HybridRetriever, create_hybrid_retriever
from src.reranking.reranker import RerankerPipeline, create_reranker
from src.generation.generator import LegalGenerator, create_generator, GenerationResult

# Import evaluation agents
from src.agents.adversarial_lawyer import AdversarialLawyerAgent, create_adversarial_lawyer
from src.agents.compliance_auditor import ComplianceAuditorAgent, create_compliance_auditor, AuditResult
from src.agents.shepardizer import ShepardizerAgent, create_shepardizer, ShepardizeResult


@dataclass
class LegalMindResponse:
    """Complete response from LegalMind system"""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    confidence: str
    retrieved_chunks: int
    reranked_chunks: int
    sources: List[Dict[str, Any]]
    
    # Evaluation results (if run)
    faithfulness_score: Optional[float] = None
    context_precision: Optional[float] = None
    citation_accuracy: Optional[float] = None


class LegalMind:
    """
    LegalMind Knowledge Assistant
    A modular RAG system for legal document Q&A with citation requirements
    """
    
    def __init__(
        self,
        openai_api_key: str,
        cohere_api_key: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        retrieval_k: int = 20,
        rerank_k: int = 5,
        persist_directory: str = "./data/chroma_db"
    ):
        self.openai_api_key = openai_api_key
        self.cohere_api_key = cohere_api_key
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        
        # Initialize components
        self.ingestion = create_ingestion_pipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.retriever = create_hybrid_retriever(
            persist_directory=persist_directory,
            openai_api_key=openai_api_key
        )
        
        self.reranker = create_reranker(
            cohere_api_key=cohere_api_key
        )
        
        self.generator = create_generator(
            openai_api_key=openai_api_key
        )
        
        # Initialize evaluation agents
        self.adversarial_lawyer = create_adversarial_lawyer(
            openai_api_key=openai_api_key
        )
        
        self.compliance_auditor = create_compliance_auditor(
            openai_api_key=openai_api_key
        )
        
        self.shepardizer = create_shepardizer(
            openai_api_key=openai_api_key
        )
        
        self._documents_indexed = False
    
    def ingest_documents(
        self,
        directory_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None
    ) -> int:
        """
        Ingest documents into the system
        
        Args:
            directory_path: Path to directory containing documents
            file_paths: List of specific file paths to ingest
        
        Returns:
            Number of chunks indexed
        """
        all_chunks = []
        
        if directory_path:
            chunks = self.ingestion.process_directory(directory_path)
            all_chunks.extend(chunks)
        
        if file_paths:
            for path in file_paths:
                chunks = self.ingestion.process_file(path)
                all_chunks.extend(chunks)
        
        if all_chunks:
            self.retriever.index_documents(all_chunks)
            self._documents_indexed = True
            print(f"✓ Indexed {len(all_chunks)} document chunks")
        
        return len(all_chunks)
    
    def query(
        self,
        question: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        run_evaluation: bool = False
    ) -> LegalMindResponse:
        """
        Query the legal knowledge base
        
        Args:
            question: The legal question to answer
            filter_metadata: Optional metadata filters (e.g., {"doc_type": "contract"})
            run_evaluation: Whether to run evaluation agents on the response
        
        Returns:
            LegalMindResponse with answer and metadata
        """
        if not self._documents_indexed:
            raise ValueError("No documents indexed. Call ingest_documents first.")
        
        # Step 1: Hybrid retrieval
        retrieval_results = self.retriever.hybrid_search(
            query=question,
            k=self.retrieval_k,
            filter_dict=filter_metadata
        )
        
        retrieved_docs = [r.document for r in retrieval_results]
        
        # Step 2: Reranking
        reranked_docs = self.reranker.rerank(
            query=question,
            documents=retrieved_docs,
            top_k=self.rerank_k
        )
        
        # Step 3: Generation with citations
        generation_result = self.generator.generate(
            query=question,
            documents=reranked_docs
        )
        
        # Build source list
        sources = [
            {
                "doc_id": doc.metadata.get("doc_id"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "source_file": doc.metadata.get("source_file"),
                "rerank_score": doc.metadata.get("rerank_score")
            }
            for doc in reranked_docs
        ]
        
        # Build response
        response = LegalMindResponse(
            query=question,
            answer=generation_result.answer,
            citations=generation_result.citations,
            confidence=generation_result.confidence,
            retrieved_chunks=len(retrieved_docs),
            reranked_chunks=len(reranked_docs),
            sources=sources
        )
        
        # Run evaluation if requested
        if run_evaluation:
            # Compliance audit (faithfulness)
            audit_result = self.compliance_auditor.audit_response(
                response=generation_result.answer,
                context_documents=reranked_docs
            )
            response.faithfulness_score = audit_result.faithfulness_score
            
            # Shepardize (citation validation)
            shepard_result = self.shepardizer.shepardize(
                response=generation_result.answer,
                context_documents=reranked_docs
            )
            response.context_precision = shepard_result.context_precision
            response.citation_accuracy = shepard_result.citation_accuracy
        
        return response
    
    def generate_golden_dataset(
        self,
        output_path: str = "./data/golden_dataset.json",
        target_size: int = 50
    ) -> Dict[str, Any]:
        """
        Generate synthetic test dataset using Adversarial Lawyer agent
        
        Args:
            output_path: Where to save the dataset
            target_size: Target number of test cases
        
        Returns:
            The generated Golden Dataset
        """
        if not self._documents_indexed:
            raise ValueError("No documents indexed. Call ingest_documents first.")
        
        # Get all documents from retriever
        documents = self.retriever.documents
        
        # Generate dataset
        dataset = self.adversarial_lawyer.generate_golden_dataset(
            documents=documents,
            target_size=target_size
        )
        
        # Save to file
        self.adversarial_lawyer.save_golden_dataset(dataset, output_path)
        
        return dataset
    
    def evaluate_rag_triad(
        self,
        question: str,
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a query using the RAG Triad metrics:
        1. Faithfulness (Groundedness)
        2. Answer Relevance
        3. Context Precision
        
        Args:
            question: The question to evaluate
            expected_answer: Optional ground truth answer for comparison
        
        Returns:
            Dictionary with RAG Triad metrics
        """
        # Get response
        response = self.query(question, run_evaluation=True)
        
        return {
            "question": question,
            "answer": response.answer,
            "metrics": {
                "faithfulness": response.faithfulness_score,
                "context_precision": response.context_precision,
                "citation_accuracy": response.citation_accuracy,
                "confidence": response.confidence
            },
            "sources_used": len(response.sources),
            "citations_provided": len(response.citations)
        }
    
    def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run batch evaluation on test cases
        
        Args:
            test_cases: List of {"question": str, "expected_answer": str}
        
        Returns:
            Aggregate evaluation results
        """
        results = []
        
        for tc in test_cases:
            result = self.evaluate_rag_triad(
                question=tc["question"],
                expected_answer=tc.get("expected_answer")
            )
            results.append(result)
        
        # Aggregate metrics
        faithfulness_scores = [r["metrics"]["faithfulness"] for r in results if r["metrics"]["faithfulness"] is not None]
        precision_scores = [r["metrics"]["context_precision"] for r in results if r["metrics"]["context_precision"] is not None]
        accuracy_scores = [r["metrics"]["citation_accuracy"] for r in results if r["metrics"]["citation_accuracy"] is not None]
        
        return {
            "total_tests": len(test_cases),
            "aggregate_metrics": {
                "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
                "avg_context_precision": sum(precision_scores) / len(precision_scores) if precision_scores else None,
                "avg_citation_accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else None
            },
            "individual_results": results
        }


# Factory function
def create_legalmind(
    openai_api_key: str,
    cohere_api_key: Optional[str] = None
) -> LegalMind:
    """Factory function to create LegalMind instance"""
    return LegalMind(
        openai_api_key=openai_api_key,
        cohere_api_key=cohere_api_key
    )


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize LegalMind
    legalmind = create_legalmind(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    # Ingest documents
    legalmind.ingest_documents(directory_path="./data/sample_docs")
    
    # Query
    response = legalmind.query(
        "What are the liability limitations in our contracts?",
        run_evaluation=True
    )
    
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Faithfulness: {response.faithfulness_score}")
    print(f"Citations: {response.citations}")
