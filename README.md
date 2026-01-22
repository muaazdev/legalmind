# LegalMind - AI Legal Knowledge Assistant

A modular RAG (Retrieval-Augmented Generation) system for legal document Q&A with mandatory citation requirements and automated evaluation.

## 🎯 Features

- **Modular RAG Architecture**: Swappable components for ingestion, retrieval, reranking, and generation
- **Hybrid Search**: Combines vector search (semantic) with BM25 (keyword) for better legal terminology handling
- **Cross-Encoder Reranking**: Uses Cohere Rerank to refine results
- **Mandatory Citations**: Every response includes source attribution
- **Three Evaluation Agents**:
  - 🎭 **Adversarial Lawyer**: Generates synthetic test data
  - 🔍 **Compliance Auditor**: Detects hallucinations
  - 📚 **Shepardizer**: Validates citations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LegalMind System                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Ingestion  │───▶│  Retrieval  │───▶│  Reranking  │     │
│  │  Pipeline   │    │   (Hybrid)  │    │  (Cohere)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   ChromaDB  │    │    BM25     │    │ Generation  │     │
│  │  (Vectors)  │    │   Index     │    │ + Citations │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                   Evaluation Agents                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Adversarial │    │ Compliance  │    │ Shepardizer │     │
│  │   Lawyer    │    │   Auditor   │    │  (Citation) │     │
│  │ (Test Gen)  │    │(Faithfulness│    │ (Validator) │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/muaazdev/legalmind.git
cd legalmind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
```

### 3. Basic Usage

```python
from src.legalmind import create_legalmind

# Initialize
legalmind = create_legalmind(
    openai_api_key="your_key",
    cohere_api_key="your_cohere_key"  # Optional
)

# Ingest documents
legalmind.ingest_documents(directory_path="./data/sample_docs")

# Query with evaluation
response = legalmind.query(
    "What are the liability limitations in our contracts?",
    run_evaluation=True
)

print(f"Answer: {response.answer}")
print(f"Citations: {response.citations}")
print(f"Faithfulness Score: {response.faithfulness_score}")
```

## 📊 RAG Triad Evaluation

LegalMind implements the **RAG Triad** metrics:

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Faithfulness** | Is the answer grounded in context? | ≥ 0.9 |
| **Answer Relevance** | Does it address the question? | ≥ 0.8 |
| **Context Precision** | Is relevant info ranked at top? | ≥ 0.8 |

### Running Evaluations

```bash
# Run all tests
pytest tests/test_rag_evaluation.py -v

# Run specific test
pytest tests/test_rag_evaluation.py::TestComplianceAuditor -v
```

## 🤖 Evaluation Agents

### 1. Adversarial Lawyer (Synthetic Test Generator)

Generates challenging test questions from your documents:

```python
from src.agents.adversarial_lawyer import create_adversarial_lawyer

lawyer = create_adversarial_lawyer(openai_api_key="your_key")
dataset = legalmind.generate_golden_dataset(target_size=50)
```

### 2. Compliance Auditor (Hallucination Detector)

Checks if responses are grounded in source documents:

```python
from src.agents.compliance_auditor import create_compliance_auditor

auditor = create_compliance_auditor(openai_api_key="your_key")
result = auditor.audit_response(response.answer, retrieved_docs)

if not result.passed:
    print(f"Hallucinations detected: {result.hallucinations}")
```

### 3. Shepardizer (Citation Validator)

Validates all citations in responses:

```python
from src.agents.shepardizer import create_shepardizer

shepardizer = create_shepardizer(openai_api_key="your_key")
result = shepardizer.shepardize(response.answer, retrieved_docs)

print(f"Citation Accuracy: {result.citation_accuracy}")
print(f"Broken Citations: {result.invalid_citations}")
```

## 🔄 CI/CD Integration

LegalMind includes GitHub Actions workflow for automated evaluation:

```yaml
# .github/workflows/rag-evaluation.yml
# Runs on every PR:
# - Linting
# - Unit tests
# - RAG Triad evaluation
# - Golden Dataset evaluation (on main)
```

### Setting up CI/CD

1. Add secrets to your GitHub repository:
   - `OPENAI_API_KEY`
   - `COHERE_API_KEY`

2. Push to trigger evaluation:
```bash
git push origin feature/my-changes
```

## 📁 Project Structure

```
legalmind/
├── src/
│   ├── ingestion/
│   │   └── pipeline.py        # Document parsing & chunking
│   ├── retrieval/
│   │   └── hybrid_retriever.py # Vector + BM25 search
│   ├── reranking/
│   │   └── reranker.py        # Cohere cross-encoder
│   ├── generation/
│   │   └── generator.py       # LLM with citations
│   ├── agents/
│   │   ├── adversarial_lawyer.py
│   │   ├── compliance_auditor.py
│   │   └── shepardizer.py
│   ├── config.py
│   └── legalmind.py           # Main orchestrator
├── tests/
│   └── test_rag_evaluation.py
├── data/
│   └── sample_docs/
├── .github/
│   └── workflows/
│       └── rag-evaluation.yml
├── requirements.txt
└── README.md
```

## 🛠️ Configuration Options

```python
legalmind = LegalMind(
    openai_api_key="...",
    cohere_api_key="...",        # Optional, falls back to local model
    chunk_size=512,              # Token chunk size
    chunk_overlap=51,            # ~10% overlap
    retrieval_k=20,              # Initial retrieval count
    rerank_k=5,                  # Final chunks after reranking
    persist_directory="./data/chroma_db"
)
```

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

Muaaz Ahmad Saeed - muaazahmed93@gmail.com

Project Link: https://github.com/muaazdev/legalmind
