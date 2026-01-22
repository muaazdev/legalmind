"""
Compliance Auditor Agent
Performs fact-checking and hallucination detection using LLM-as-a-judge
- Extracts individual claims from RAG responses
- Cross-references claims against retrieved context
- Calculates Faithfulness (Groundedness) score
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class Claim:
    """A single claim extracted from a response"""
    text: str
    is_supported: bool
    supporting_evidence: Optional[str]
    confidence: float  # 0.0 to 1.0


@dataclass
class AuditResult:
    """Result of compliance audit"""
    faithfulness_score: float  # 0.0 to 1.0
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    claims: List[Claim]
    hallucinations: List[str]
    passed: bool
    threshold: float


CLAIM_EXTRACTION_PROMPT = """You are a legal fact-checker. Extract all factual claims from the following legal response.

A claim is any statement that asserts something as true or factual.

RESPONSE TO ANALYZE:
{response}

OUTPUT FORMAT (JSON):
{{
    "claims": [
        "First factual claim",
        "Second factual claim",
        ...
    ]
}}

Extract ONLY factual claims, not opinions or qualifiers. Return ONLY valid JSON."""


CLAIM_VERIFICATION_PROMPT = """You are a legal compliance auditor. Your task is to verify if a claim is supported by the provided legal context.

CLAIM TO VERIFY:
{claim}

LEGAL CONTEXT (Source Documents):
{context}

VERIFICATION RULES:
1. The claim must be DIRECTLY supported by text in the context
2. Reasonable inference is allowed, but speculation is not
3. If the context doesn't mention the topic at all, the claim is NOT supported
4. Partial support counts as NOT supported - claims must be fully verifiable

OUTPUT FORMAT (JSON):
{{
    "is_supported": true/false,
    "supporting_evidence": "Quote from context that supports the claim, or null if unsupported",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}

Return ONLY valid JSON."""


class ComplianceAuditorAgent:
    """
    Agent that performs faithfulness checking on RAG responses
    Uses LLM-as-a-judge pattern to detect hallucinations
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        faithfulness_threshold: float = 0.9
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=0.0  # Deterministic for consistent evaluation
        )
        
        self.faithfulness_threshold = faithfulness_threshold
        
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("human", CLAIM_EXTRACTION_PROMPT)
        ])
        
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("human", CLAIM_VERIFICATION_PROMPT)
        ])
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        return json.loads(response.strip())
    
    def extract_claims(self, response: str) -> List[str]:
        """
        Extract factual claims from a response
        
        Args:
            response: The RAG system's response
        
        Returns:
            List of factual claims
        """
        messages = self.extraction_prompt.format_messages(response=response)
        
        try:
            result = self.llm.invoke(messages)
            data = self._parse_json_response(result.content)
            return data.get("claims", [])
        
        except Exception as e:
            print(f"Error extracting claims: {e}")
            # Fallback: split by sentences
            sentences = re.split(r'[.!?]+', response)
            return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def verify_claim(
        self,
        claim: str,
        context_documents: List[Document]
    ) -> Claim:
        """
        Verify a single claim against context documents
        
        Args:
            claim: The claim to verify
            context_documents: Source documents to check against
        
        Returns:
            Claim object with verification result
        """
        # Format context
        context = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('doc_id', 'unknown')}]\n{doc.page_content}"
            for doc in context_documents
        ])
        
        messages = self.verification_prompt.format_messages(
            claim=claim,
            context=context
        )
        
        try:
            result = self.llm.invoke(messages)
            data = self._parse_json_response(result.content)
            
            return Claim(
                text=claim,
                is_supported=data.get("is_supported", False),
                supporting_evidence=data.get("supporting_evidence"),
                confidence=data.get("confidence", 0.0)
            )
        
        except Exception as e:
            print(f"Error verifying claim: {e}")
            return Claim(
                text=claim,
                is_supported=False,
                supporting_evidence=None,
                confidence=0.0
            )
    
    def audit_response(
        self,
        response: str,
        context_documents: List[Document]
    ) -> AuditResult:
        """
        Perform full compliance audit on a RAG response
        
        Args:
            response: The RAG system's response
            context_documents: Retrieved source documents
        
        Returns:
            AuditResult with faithfulness score and details
        """
        # Extract claims
        claims_text = self.extract_claims(response)
        
        if not claims_text:
            return AuditResult(
                faithfulness_score=1.0,  # No claims to verify
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                claims=[],
                hallucinations=[],
                passed=True,
                threshold=self.faithfulness_threshold
            )
        
        # Verify each claim
        verified_claims = []
        hallucinations = []
        
        for claim_text in claims_text:
            claim = self.verify_claim(claim_text, context_documents)
            verified_claims.append(claim)
            
            if not claim.is_supported:
                hallucinations.append(claim_text)
        
        # Calculate faithfulness score
        supported_count = sum(1 for c in verified_claims if c.is_supported)
        total_count = len(verified_claims)
        faithfulness_score = supported_count / total_count if total_count > 0 else 1.0
        
        # Determine if audit passed
        passed = faithfulness_score >= self.faithfulness_threshold
        
        return AuditResult(
            faithfulness_score=faithfulness_score,
            total_claims=total_count,
            supported_claims=supported_count,
            unsupported_claims=total_count - supported_count,
            claims=verified_claims,
            hallucinations=hallucinations,
            passed=passed,
            threshold=self.faithfulness_threshold
        )
    
    def batch_audit(
        self,
        responses: List[Tuple[str, List[Document]]]
    ) -> List[AuditResult]:
        """
        Audit multiple responses
        
        Args:
            responses: List of (response, context_documents) tuples
        
        Returns:
            List of AuditResults
        """
        results = []
        for response, context in responses:
            result = self.audit_response(response, context)
            results.append(result)
        return results
    
    def get_summary_report(self, results: List[AuditResult]) -> Dict[str, Any]:
        """Generate summary report from multiple audit results"""
        if not results:
            return {"error": "No results to summarize"}
        
        avg_faithfulness = sum(r.faithfulness_score for r in results) / len(results)
        pass_rate = sum(1 for r in results if r.passed) / len(results)
        total_hallucinations = sum(len(r.hallucinations) for r in results)
        
        return {
            "total_audits": len(results),
            "average_faithfulness": round(avg_faithfulness, 3),
            "pass_rate": round(pass_rate, 3),
            "total_hallucinations_detected": total_hallucinations,
            "threshold": self.faithfulness_threshold,
            "all_passed": all(r.passed for r in results)
        }


# Factory function
def create_compliance_auditor(
    openai_api_key: Optional[str] = None,
    faithfulness_threshold: float = 0.9
) -> ComplianceAuditorAgent:
    """Factory function to create Compliance Auditor agent"""
    return ComplianceAuditorAgent(
        openai_api_key=openai_api_key,
        faithfulness_threshold=faithfulness_threshold
    )
