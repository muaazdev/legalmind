"""
Shepardizer Agent
Named after "Shepardizing" - the legal practice of verifying citations
- Validates that every citation in the response exists in source documents
- Checks citation accuracy and relevance
- Ensures auditability and explainability standards
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class CitationValidation:
    """Validation result for a single citation"""
    citation_text: str  # The citation as it appears in response
    doc_id: str
    chunk_index: Optional[int]
    exists: bool  # Does the document exist?
    is_relevant: bool  # Is the citation relevant to the claim?
    is_accurate: bool  # Does the source actually say what's claimed?
    error_message: Optional[str]


@dataclass
class ShepardizeResult:
    """Result of citation validation"""
    context_precision: float  # How relevant is the top context?
    citation_accuracy: float  # What % of citations are valid?
    total_citations: int
    valid_citations: int
    invalid_citations: int
    missing_citations: int  # Claims without citations
    validations: List[CitationValidation]
    passed: bool
    issues: List[str]


RELEVANCE_CHECK_PROMPT = """You are a legal citation validator. Determine if the cited source document is relevant to the claim being made.

CLAIM WITH CITATION:
{claim}

CITED SOURCE CONTENT:
{source_content}

Is the source content relevant and accurately supporting the claim?

OUTPUT FORMAT (JSON):
{{
    "is_relevant": true/false,
    "is_accurate": true/false,
    "reasoning": "Brief explanation"
}}

Return ONLY valid JSON."""


class ShepardizerAgent:
    """
    Agent that validates citations and source references
    Ensures legal compliance through proper attribution
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini"
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=0.0
        )
        
        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("human", RELEVANCE_CHECK_PROMPT)
        ])
    
    def _extract_citations(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract citations from response text
        Expected format: [Source: DOC_ID, Chunk: X]
        """
        citations = []
        
        # Pattern for [Source: X, Chunk: Y]
        pattern = r'\[Source:\s*([^\],]+),\s*Chunk:\s*(\d+)\]'
        matches = re.findall(pattern, response)
        
        for doc_id, chunk_idx in matches:
            citations.append({
                "citation_text": f"[Source: {doc_id.strip()}, Chunk: {chunk_idx}]",
                "doc_id": doc_id.strip(),
                "chunk_index": int(chunk_idx)
            })
        
        # Also handle simpler patterns [Source: X]
        simple_pattern = r'\[Source:\s*([^\]]+)\]'
        simple_matches = re.findall(simple_pattern, response)
        
        for match in simple_matches:
            if "Chunk" not in match:  # Avoid duplicates
                citations.append({
                    "citation_text": f"[Source: {match.strip()}]",
                    "doc_id": match.strip(),
                    "chunk_index": None
                })
        
        return citations
    
    def _find_document(
        self,
        doc_id: str,
        chunk_index: Optional[int],
        documents: List[Document]
    ) -> Optional[Document]:
        """Find document by ID and optional chunk index"""
        for doc in documents:
            if doc.metadata.get("doc_id") == doc_id:
                if chunk_index is None:
                    return doc
                if doc.metadata.get("chunk_index") == chunk_index:
                    return doc
        return None
    
    def _get_claim_for_citation(self, response: str, citation_text: str) -> str:
        """Extract the claim associated with a citation"""
        # Find the sentence containing the citation
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        for sentence in sentences:
            if citation_text in sentence:
                return sentence
        
        # Fallback: return surrounding context
        idx = response.find(citation_text)
        if idx != -1:
            start = max(0, idx - 200)
            end = min(len(response), idx + len(citation_text) + 100)
            return response[start:end]
        
        return citation_text
    
    def _check_relevance(
        self,
        claim: str,
        source_content: str
    ) -> Tuple[bool, bool]:
        """Check if source is relevant and accurate for the claim"""
        messages = self.relevance_prompt.format_messages(
            claim=claim,
            source_content=source_content
        )
        
        try:
            result = self.llm.invoke(messages)
            
            # Parse response
            content = result.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            import json
            data = json.loads(content.strip())
            
            return data.get("is_relevant", False), data.get("is_accurate", False)
        
        except Exception as e:
            print(f"Error checking relevance: {e}")
            return False, False
    
    def _calculate_context_precision(
        self,
        response: str,
        documents: List[Document]
    ) -> float:
        """
        Calculate Context Precision - is the most relevant info ranked at top?
        Uses position-weighted relevance scoring
        """
        if not documents:
            return 0.0
        
        # Check each document's relevance (higher weight for earlier documents)
        relevance_scores = []
        
        for i, doc in enumerate(documents[:5]):  # Top 5 documents
            # Simple heuristic: check if document content appears in response
            doc_terms = set(doc.page_content.lower().split())
            response_terms = set(response.lower().split())
            overlap = len(doc_terms & response_terms) / len(doc_terms) if doc_terms else 0
            
            # Position weight: earlier documents should be more relevant
            position_weight = 1.0 / (i + 1)
            
            relevance_scores.append(overlap * position_weight)
        
        # Normalize
        max_possible = sum(1.0 / (i + 1) for i in range(len(documents[:5])))
        
        return sum(relevance_scores) / max_possible if max_possible > 0 else 0.0
    
    def shepardize(
        self,
        response: str,
        context_documents: List[Document]
    ) -> ShepardizeResult:
        """
        Validate all citations in a response
        
        Args:
            response: The RAG system's response
            context_documents: Source documents that were provided to the LLM
        
        Returns:
            ShepardizeResult with validation details
        """
        # Extract citations from response
        citations = self._extract_citations(response)
        
        validations = []
        issues = []
        valid_count = 0
        
        for citation in citations:
            # Find the source document
            doc = self._find_document(
                citation["doc_id"],
                citation["chunk_index"],
                context_documents
            )
            
            if not doc:
                # Citation references non-existent document
                validations.append(CitationValidation(
                    citation_text=citation["citation_text"],
                    doc_id=citation["doc_id"],
                    chunk_index=citation["chunk_index"],
                    exists=False,
                    is_relevant=False,
                    is_accurate=False,
                    error_message="Document not found in provided context"
                ))
                issues.append(f"Broken citation: {citation['citation_text']} - document not found")
                continue
            
            # Get the claim associated with this citation
            claim = self._get_claim_for_citation(response, citation["citation_text"])
            
            # Check relevance and accuracy
            is_relevant, is_accurate = self._check_relevance(
                claim,
                doc.page_content
            )
            
            validation = CitationValidation(
                citation_text=citation["citation_text"],
                doc_id=citation["doc_id"],
                chunk_index=citation["chunk_index"],
                exists=True,
                is_relevant=is_relevant,
                is_accurate=is_accurate,
                error_message=None if (is_relevant and is_accurate) else "Citation may be irrelevant or inaccurate"
            )
            
            validations.append(validation)
            
            if validation.exists and validation.is_relevant and validation.is_accurate:
                valid_count += 1
            else:
                if not is_relevant:
                    issues.append(f"Irrelevant citation: {citation['citation_text']}")
                if not is_accurate:
                    issues.append(f"Potentially inaccurate citation: {citation['citation_text']}")
        
        # Calculate metrics
        total = len(citations)
        citation_accuracy = valid_count / total if total > 0 else 1.0
        
        # Calculate context precision
        context_precision = self._calculate_context_precision(response, context_documents)
        
        # Check for claims without citations (if response has substantial content)
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 30]
        sentences_with_citations = [s for s in sentences if '[Source:' in s]
        missing_citations = len(sentences) - len(sentences_with_citations)
        
        if missing_citations > len(sentences) // 2:
            issues.append(f"{missing_citations} statements appear to lack citations")
        
        # Determine pass/fail
        passed = citation_accuracy >= 0.8 and context_precision >= 0.5 and len([v for v in validations if not v.exists]) == 0
        
        return ShepardizeResult(
            context_precision=round(context_precision, 3),
            citation_accuracy=round(citation_accuracy, 3),
            total_citations=total,
            valid_citations=valid_count,
            invalid_citations=total - valid_count,
            missing_citations=missing_citations,
            validations=validations,
            passed=passed,
            issues=issues
        )
    
    def batch_shepardize(
        self,
        responses: List[Tuple[str, List[Document]]]
    ) -> List[ShepardizeResult]:
        """Validate citations for multiple responses"""
        return [self.shepardize(r, d) for r, d in responses]
    
    def get_summary_report(self, results: List[ShepardizeResult]) -> Dict[str, Any]:
        """Generate summary report from multiple validations"""
        if not results:
            return {"error": "No results to summarize"}
        
        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_accuracy = sum(r.citation_accuracy for r in results) / len(results)
        pass_rate = sum(1 for r in results if r.passed) / len(results)
        total_issues = sum(len(r.issues) for r in results)
        
        return {
            "total_validations": len(results),
            "average_context_precision": round(avg_precision, 3),
            "average_citation_accuracy": round(avg_accuracy, 3),
            "pass_rate": round(pass_rate, 3),
            "total_issues_found": total_issues,
            "all_passed": all(r.passed for r in results)
        }


# Factory function
def create_shepardizer(
    openai_api_key: Optional[str] = None
) -> ShepardizerAgent:
    """Factory function to create Shepardizer agent"""
    return ShepardizerAgent(openai_api_key=openai_api_key)
