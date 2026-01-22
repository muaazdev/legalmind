"""
LegalMind Evaluation Tests
Uses RAG Triad metrics with pytest integration
- Faithfulness (Groundedness)
- Answer Relevance  
- Context Precision

Run with: pytest tests/test_rag_evaluation.py -v
"""
import pytest
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LegalMind components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.legalmind import LegalMind, create_legalmind
from src.agents.compliance_auditor import create_compliance_auditor
from src.agents.shepardizer import create_shepardizer
from src.agents.adversarial_lawyer import create_adversarial_lawyer

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Evaluation thresholds (fail test if below these)
FAITHFULNESS_THRESHOLD = 0.9
CONTEXT_PRECISION_THRESHOLD = 0.8
CITATION_ACCURACY_THRESHOLD = 0.8


class TestRAGEvaluation:
    """
    Test suite for RAG system evaluation
    Uses the RAG Triad metrics
    """
    
    @pytest.fixture(scope="class")
    def legalmind(self):
        """Initialize LegalMind instance for testing"""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not set")
        
        lm = create_legalmind(
            openai_api_key=OPENAI_API_KEY,
            cohere_api_key=COHERE_API_KEY
        )
        
        # Check if sample documents exist
        sample_dir = "./data/sample_docs"
        if os.path.exists(sample_dir) and os.listdir(sample_dir):
            lm.ingest_documents(directory_path=sample_dir)
        else:
            pytest.skip("No sample documents found in ./data/sample_docs")
        
        return lm
    
    @pytest.fixture(scope="class")
    def golden_dataset(self):
        """Load or generate Golden Dataset"""
        dataset_path = "./data/golden_dataset.json"
        
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def test_faithfulness_threshold(self, legalmind):
        """
        Test that faithfulness score meets threshold
        Faithfulness measures if answers are grounded in retrieved context
        """
        test_question = "What are the key terms in the contract?"
        
        response = legalmind.query(test_question, run_evaluation=True)
        
        assert response.faithfulness_score is not None, "Faithfulness score not calculated"
        assert response.faithfulness_score >= FAITHFULNESS_THRESHOLD, \
            f"Faithfulness score {response.faithfulness_score} below threshold {FAITHFULNESS_THRESHOLD}"
    
    def test_context_precision_threshold(self, legalmind):
        """
        Test that context precision meets threshold
        Context Precision measures if relevant info is ranked at top
        """
        test_question = "What are the liability limitations?"
        
        response = legalmind.query(test_question, run_evaluation=True)
        
        assert response.context_precision is not None, "Context precision not calculated"
        assert response.context_precision >= CONTEXT_PRECISION_THRESHOLD, \
            f"Context precision {response.context_precision} below threshold {CONTEXT_PRECISION_THRESHOLD}"
    
    def test_citation_accuracy_threshold(self, legalmind):
        """
        Test that citation accuracy meets threshold
        Ensures all citations are valid and point to real sources
        """
        test_question = "What obligations does each party have?"
        
        response = legalmind.query(test_question, run_evaluation=True)
        
        assert response.citation_accuracy is not None, "Citation accuracy not calculated"
        assert response.citation_accuracy >= CITATION_ACCURACY_THRESHOLD, \
            f"Citation accuracy {response.citation_accuracy} below threshold {CITATION_ACCURACY_THRESHOLD}"


class TestComplianceAuditor:
    """
    Tests for the Compliance Auditor (hallucination detection)
    """
    
    @pytest.fixture(scope="class")
    def compliance_auditor(self):
        """Initialize Compliance Auditor agent"""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not set")
        return create_compliance_auditor(openai_api_key=OPENAI_API_KEY)
    
    def test_detects_hallucination(self, compliance_auditor):
        """
        Test that auditor correctly identifies hallucinated content
        """
        from langchain.schema import Document
        
        # Context only mentions $1M liability
        context = [
            Document(
                page_content="The maximum liability under this agreement is $1,000,000.",
                metadata={"doc_id": "test", "chunk_index": 0}
            )
        ]
        
        # Response incorrectly claims $5M liability
        hallucinated_response = "The contract specifies a maximum liability of $5,000,000."
        
        result = compliance_auditor.audit_response(
            response=hallucinated_response,
            context_documents=context
        )
        
        assert not result.passed or result.faithfulness_score < 0.9, \
            "Should detect the hallucinated liability amount"
    
    def test_validates_faithful_response(self, compliance_auditor):
        """
        Test that auditor correctly validates faithful content
        """
        from langchain.schema import Document
        
        context = [
            Document(
                page_content="The agreement shall be governed by the laws of California.",
                metadata={"doc_id": "test", "chunk_index": 0}
            )
        ]
        
        faithful_response = "This agreement is governed by California law."
        
        result = compliance_auditor.audit_response(
            response=faithful_response,
            context_documents=context
        )
        
        assert result.faithfulness_score >= 0.8, \
            f"Faithful response should have high faithfulness score, got {result.faithfulness_score}"


class TestShepardizer:
    """
    Tests for the Shepardizer (citation validator)
    """
    
    @pytest.fixture(scope="class")
    def shepardizer(self):
        """Initialize Shepardizer agent"""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not set")
        return create_shepardizer(openai_api_key=OPENAI_API_KEY)
    
    def test_validates_correct_citation(self, shepardizer):
        """
        Test that Shepardizer validates correct citations
        """
        from langchain.schema import Document
        
        context = [
            Document(
                page_content="The indemnification clause requires Party B to hold Party A harmless.",
                metadata={"doc_id": "contract_001", "chunk_index": 0}
            )
        ]
        
        response_with_citation = "Party B must indemnify Party A [Source: contract_001, Chunk: 0]."
        
        result = shepardizer.shepardize(
            response=response_with_citation,
            context_documents=context
        )
        
        assert result.total_citations > 0, "Should detect the citation"
        assert result.valid_citations > 0, "Citation should be valid"
    
    def test_detects_broken_citation(self, shepardizer):
        """
        Test that Shepardizer detects citations to non-existent documents
        """
        from langchain.schema import Document
        
        context = [
            Document(
                page_content="Some legal text here.",
                metadata={"doc_id": "real_doc", "chunk_index": 0}
            )
        ]
        
        # Citation references non-existent document
        response_with_broken_citation = "The clause states XYZ [Source: fake_doc, Chunk: 99]."
        
        result = shepardizer.shepardize(
            response=response_with_broken_citation,
            context_documents=context
        )
        
        broken = [v for v in result.validations if not v.exists]
        assert len(broken) > 0, "Should detect the broken citation"


class TestAdversarialLawyer:
    """
    Tests for the Adversarial Lawyer synthetic test generator
    """
    
    @pytest.fixture(scope="class")
    def adversarial_lawyer(self):
        """Initialize Adversarial Lawyer agent"""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not set")
        return create_adversarial_lawyer(openai_api_key=OPENAI_API_KEY)
    
    def test_generates_test_cases(self, adversarial_lawyer):
        """
        Test that Adversarial Lawyer generates test cases from documents
        """
        from langchain.schema import Document
        
        sample_docs = [
            Document(
                page_content="The liability of Party A shall not exceed $1,000,000 in aggregate.",
                metadata={"doc_id": "contract_001", "chunk_index": 0}
            ),
            Document(
                page_content="Force majeure events include natural disasters, war, and government actions.",
                metadata={"doc_id": "contract_001", "chunk_index": 1}
            )
        ]
        
        test_cases = adversarial_lawyer.generate_test_cases(
            documents=sample_docs,
            num_questions=3
        )
        
        assert len(test_cases) > 0, "Should generate at least one test case"
        assert all(tc.question for tc in test_cases), "All test cases should have questions"


class TestEndToEndPipeline:
    """
    End-to-end integration tests
    """
    
    def test_full_pipeline_with_sample_docs(self):
        """
        Test the complete pipeline from ingestion to evaluation
        """
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not set")
        
        from langchain.schema import Document
        
        # Create LegalMind instance
        legalmind = create_legalmind(
            openai_api_key=OPENAI_API_KEY,
            cohere_api_key=COHERE_API_KEY
        )
        
        # Create sample documents
        sample_docs = [
            Document(
                page_content="""
                SERVICES AGREEMENT
                
                1. SCOPE OF SERVICES
                The Contractor agrees to provide software development services.
                
                2. COMPENSATION
                The Client shall pay the Contractor $150 per hour for services rendered.
                Payment is due within 30 days of invoice receipt.
                
                3. LIABILITY
                The Contractor's total liability shall not exceed the total fees paid.
                """,
                metadata={
                    "doc_id": "services_001",
                    "chunk_index": 0,
                    "source_file": "services_agreement.pdf",
                    "doc_type": "contract"
                }
            )
        ]
        
        # Index documents
        legalmind.retriever.index_documents(sample_docs)
        legalmind._documents_indexed = True
        
        # Query
        response = legalmind.query(
            "What is the hourly rate?",
            run_evaluation=True
        )
        
        # Verify response
        assert response.answer is not None, "Should return an answer"
        assert "$150" in response.answer or "150" in response.answer, \
            "Answer should mention the hourly rate"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
