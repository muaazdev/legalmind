"""
LegalMind Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Model Settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RERANK_MODEL = "rerank-english-v3.0"

# Chunking Settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 51  # ~10% overlap

# Retrieval Settings
VECTOR_SEARCH_K = 20  # Initial retrieval count
BM25_SEARCH_K = 20
RERANK_TOP_K = 5  # Final chunks after reranking

# Vector Store
CHROMA_PERSIST_DIR = "./data/chroma_db"

# Redis Cache (optional)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 3600  # 1 hour

# Evaluation Thresholds
FAITHFULNESS_THRESHOLD = 0.9
ANSWER_RELEVANCE_THRESHOLD = 0.8
CONTEXT_PRECISION_THRESHOLD = 0.8
