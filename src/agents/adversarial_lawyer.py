"""
Adversarial Lawyer Agent
Generates synthetic test data (Golden Dataset) by analyzing legal documents
- Creates complex, multi-hop legal questions
- Generates ground-truth pairs (Question, Reference Context, Expected Answer)
- Used for benchmarking RAG system performance
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import random

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


@dataclass
class TestCase:
    """A single test case in the Golden Dataset"""
    question: str
    reference_contexts: List[str]  # Ground truth context chunks
    expected_answer: str
    difficulty: str  # "simple", "multi-hop", "complex"
    doc_ids: List[str]  # Source document IDs
    metadata: Dict[str, Any]


ADVERSARIAL_SYSTEM_PROMPT = """You are the "Adversarial Lawyer" - an expert at creating challenging legal questions 
that test the limits of a RAG (Retrieval-Augmented Generation) system.

Your task is to analyze legal documents and generate test cases that will:
1. Test single-document retrieval (simple questions)
2. Test multi-document reasoning (questions requiring information from multiple sources)
3. Test edge cases (questions where the answer is NOT in the documents)

For each document chunk provided, generate diverse questions that a real lawyer might ask.

OUTPUT FORMAT (JSON):
{
    "test_cases": [
        {
            "question": "The legal question",
            "expected_answer": "The correct answer based on the document",
            "difficulty": "simple|multi-hop|complex",
            "reasoning": "Why this tests the RAG system well"
        }
    ]
}

QUESTION TYPES TO GENERATE:
1. SIMPLE: Direct questions answerable from a single chunk
2. MULTI-HOP: Questions requiring synthesis from multiple chunks
3. COMPLEX: Questions about interactions between clauses, exceptions, or edge cases
4. NEGATIVE: Questions where the answer is NOT in the provided context (to test "I don't know" behavior)"""


ADVERSARIAL_HUMAN_PROMPT = """Analyze these legal document chunks and generate {num_questions} challenging test questions.

DOCUMENT CHUNKS:
{documents}

Generate a mix of simple, multi-hop, complex, and negative (unanswerable) questions.
Return ONLY valid JSON."""


class AdversarialLawyerAgent:
    """
    Agent that generates synthetic test data from legal documents
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini"
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=0.7  # Some creativity for diverse questions
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ADVERSARIAL_SYSTEM_PROMPT),
            ("human", ADVERSARIAL_HUMAN_PROMPT)
        ])
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for the prompt"""
        formatted = []
        for i, doc in enumerate(documents):
            doc_id = doc.metadata.get("doc_id", f"doc_{i}")
            chunk_idx = doc.metadata.get("chunk_index", i)
            formatted.append(
                f"[DOC_ID: {doc_id}, CHUNK: {chunk_idx}]\n{doc.page_content}\n---"
            )
        return "\n\n".join(formatted)
    
    def _parse_response(self, response: str, documents: List[Document]) -> List[TestCase]:
        """Parse LLM response into TestCase objects"""
        test_cases = []
        
        try:
            # Clean response and parse JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response)
            
            for tc in data.get("test_cases", []):
                # Get relevant document contexts
                doc_ids = []
                contexts = []
                
                # For now, use all provided documents as potential context
                for doc in documents:
                    doc_ids.append(doc.metadata.get("doc_id", "unknown"))
                    contexts.append(doc.page_content)
                
                test_cases.append(TestCase(
                    question=tc["question"],
                    reference_contexts=contexts,
                    expected_answer=tc["expected_answer"],
                    difficulty=tc.get("difficulty", "simple"),
                    doc_ids=list(set(doc_ids)),
                    metadata={"reasoning": tc.get("reasoning", "")}
                ))
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse response as JSON: {e}")
            print(f"Response was: {response[:500]}...")
        
        return test_cases
    
    def generate_test_cases(
        self,
        documents: List[Document],
        num_questions: int = 10,
        batch_size: int = 5
    ) -> List[TestCase]:
        """
        Generate synthetic test cases from documents
        
        Args:
            documents: List of document chunks to analyze
            num_questions: Total number of questions to generate
            batch_size: Number of documents to process at a time
        
        Returns:
            List of TestCase objects forming the Golden Dataset
        """
        all_test_cases = []
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            if not batch:
                continue
            
            # Format documents
            formatted_docs = self._format_documents(batch)
            
            # Calculate questions per batch
            questions_per_batch = max(1, num_questions // (len(documents) // batch_size + 1))
            
            # Generate questions
            messages = self.prompt.format_messages(
                documents=formatted_docs,
                num_questions=questions_per_batch
            )
            
            try:
                response = self.llm.invoke(messages)
                test_cases = self._parse_response(response.content, batch)
                all_test_cases.extend(test_cases)
                print(f"✓ Generated {len(test_cases)} test cases from batch {i // batch_size + 1}")
            
            except Exception as e:
                print(f"✗ Error generating test cases: {e}")
        
        return all_test_cases
    
    def generate_golden_dataset(
        self,
        documents: List[Document],
        target_size: int = 50
    ) -> Dict[str, Any]:
        """
        Generate a complete Golden Dataset for benchmarking
        
        Args:
            documents: All document chunks
            target_size: Target number of test cases
        
        Returns:
            Dictionary containing the Golden Dataset
        """
        # Sample documents if too many
        if len(documents) > 50:
            sampled_docs = random.sample(documents, 50)
        else:
            sampled_docs = documents
        
        # Generate test cases
        test_cases = self.generate_test_cases(
            sampled_docs,
            num_questions=target_size
        )
        
        # Organize by difficulty
        golden_dataset = {
            "metadata": {
                "total_cases": len(test_cases),
                "source_documents": len(set(tc.doc_ids[0] for tc in test_cases if tc.doc_ids)),
                "difficulty_distribution": {
                    "simple": sum(1 for tc in test_cases if tc.difficulty == "simple"),
                    "multi-hop": sum(1 for tc in test_cases if tc.difficulty == "multi-hop"),
                    "complex": sum(1 for tc in test_cases if tc.difficulty == "complex"),
                    "negative": sum(1 for tc in test_cases if tc.difficulty == "negative")
                }
            },
            "test_cases": [
                {
                    "question": tc.question,
                    "reference_contexts": tc.reference_contexts,
                    "expected_answer": tc.expected_answer,
                    "difficulty": tc.difficulty,
                    "doc_ids": tc.doc_ids,
                    "metadata": tc.metadata
                }
                for tc in test_cases
            ]
        }
        
        return golden_dataset
    
    def save_golden_dataset(self, dataset: Dict[str, Any], filepath: str) -> None:
        """Save Golden Dataset to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"✓ Saved Golden Dataset to {filepath}")
    
    def load_golden_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load Golden Dataset from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)


# Factory function
def create_adversarial_lawyer(
    openai_api_key: Optional[str] = None
) -> AdversarialLawyerAgent:
    """Factory function to create Adversarial Lawyer agent"""
    return AdversarialLawyerAgent(openai_api_key=openai_api_key)
