"""
Generation Layer
- LLM-based response generation with mandatory citations
- System prompt enforcing source attribution
- "I don't know" fallback for insufficient context
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


@dataclass
class GenerationResult:
    """Container for generation results with citations"""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: str  # "high", "medium", "low", "insufficient"
    raw_response: str


# System prompt enforcing citations and honesty
LEGAL_SYSTEM_PROMPT = """You are LegalMind, an AI legal research assistant for a law firm. 
Your role is to answer questions based ONLY on the provided legal documents.

CRITICAL RULES:
1. ONLY use information from the provided context documents. Do not use external knowledge.
2. EVERY claim must include a citation in the format [Source: DOC_ID, Chunk: X].
3. If the provided context does not contain sufficient information to answer the question, 
   you MUST respond with: "I don't have sufficient information in the provided documents to answer this question."
4. Never fabricate or hallucinate legal information - this could cause serious harm.
5. If documents contain conflicting information, acknowledge the conflict and cite both sources.

RESPONSE FORMAT:
- Provide a clear, well-structured answer
- Include inline citations for every factual claim
- End with a "Sources" section listing all referenced documents

CONFIDENCE LEVELS:
- HIGH: Multiple documents support the answer with clear, direct statements
- MEDIUM: One document supports the answer, or inference is required
- LOW: Partial information available, significant inference required
- INSUFFICIENT: Cannot answer from provided context"""


LEGAL_HUMAN_PROMPT = """Based on the following legal documents, please answer the question.

DOCUMENTS:
{context}

QUESTION: {question}

Remember: Cite every claim with [Source: DOC_ID, Chunk: X]. Say "I don't know" if context is insufficient."""


class LegalGenerator:
    """
    LLM-based generator with mandatory citation enforcement
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        temperature: float = 0.1  # Low temperature for factual accuracy
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=temperature
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(LEGAL_SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(LEGAL_HUMAN_PROMPT)
        ])
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string with clear identifiers"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            doc_id = doc.metadata.get("doc_id", f"doc_{i}")
            chunk_id = doc.metadata.get("chunk_index", i)
            source_file = doc.metadata.get("source_file", "Unknown")
            doc_type = doc.metadata.get("doc_type", "document")
            
            context_parts.append(
                f"[Document ID: {doc_id}, Chunk: {chunk_id}]\n"
                f"Source: {source_file}\n"
                f"Type: {doc_type}\n"
                f"Content:\n{doc.page_content}\n"
                f"{'='*50}"
            )
        
        return "\n\n".join(context_parts)
    
    def _extract_citations(self, response: str, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract citations from response and validate against source documents"""
        citations = []
        seen = set()
        
        # Find all citation patterns [Source: X, Chunk: Y]
        import re
        pattern = r'\[Source:\s*([^\],]+),\s*Chunk:\s*(\d+)\]'
        matches = re.findall(pattern, response)
        
        for doc_id, chunk_idx in matches:
            doc_id = doc_id.strip()
            chunk_idx = int(chunk_idx)
            
            # Avoid duplicates
            key = f"{doc_id}_{chunk_idx}"
            if key in seen:
                continue
            seen.add(key)
            
            # Find matching document
            matching_doc = None
            for doc in documents:
                if doc.metadata.get("doc_id") == doc_id and doc.metadata.get("chunk_index") == chunk_idx:
                    matching_doc = doc
                    break
            
            citations.append({
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "source_file": matching_doc.metadata.get("source_file") if matching_doc else "Unknown",
                "verified": matching_doc is not None
            })
        
        return citations
    
    def _assess_confidence(self, response: str, citations: List[Dict], documents: List[Document]) -> str:
        """Assess confidence level of the response"""
        # Check for "I don't know" patterns
        insufficient_patterns = [
            "i don't have sufficient information",
            "cannot answer",
            "not enough information",
            "no information available"
        ]
        
        if any(pattern in response.lower() for pattern in insufficient_patterns):
            return "insufficient"
        
        # Check citation coverage
        verified_citations = sum(1 for c in citations if c["verified"])
        
        if verified_citations >= 3:
            return "high"
        elif verified_citations >= 1:
            return "medium"
        elif len(citations) > 0:
            return "low"
        else:
            return "low"
    
    def generate(
        self,
        query: str,
        documents: List[Document]
    ) -> GenerationResult:
        """
        Generate response with citations
        
        Args:
            query: User's legal question
            documents: Retrieved and reranked documents
        
        Returns:
            GenerationResult with answer, citations, and confidence
        """
        # Format context
        context = self._format_context(documents)
        
        # Create prompt
        messages = self.prompt.format_messages(
            context=context,
            question=query
        )
        
        # Generate response
        response = self.llm.invoke(messages)
        raw_response = response.content
        
        # Extract and validate citations
        citations = self._extract_citations(raw_response, documents)
        
        # Assess confidence
        confidence = self._assess_confidence(raw_response, citations, documents)
        
        return GenerationResult(
            answer=raw_response,
            citations=citations,
            confidence=confidence,
            raw_response=raw_response
        )


# Factory function
def create_generator(
    model_name: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None
) -> LegalGenerator:
    """Factory function to create legal generator"""
    return LegalGenerator(
        model_name=model_name,
        openai_api_key=openai_api_key
    )
