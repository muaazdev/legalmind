"""
LegalMind FastAPI Application
RESTful API for the RAG system with health checks and metrics
"""
import os
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from src.legalmind import LegalMind, create_legalmind, LegalMindResponse
from dotenv import load_dotenv

load_dotenv()

# ============== Pydantic Models ==============

class QueryRequest(BaseModel):
    """Request model for querying LegalMind"""
    question: str = Field(..., min_length=5, description="Legal question to ask")
    run_evaluation: bool = Field(default=False, description="Run evaluation agents")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class QueryResponse(BaseModel):
    """Response model for query results"""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    confidence: str
    retrieved_chunks: int
    reranked_chunks: int
    sources: List[Dict[str, Any]]
    faithfulness_score: Optional[float] = None
    context_precision: Optional[float] = None
    citation_accuracy: Optional[float] = None
    latency_ms: float


class IngestRequest(BaseModel):
    """Request model for document ingestion"""
    directory_path: Optional[str] = None
    file_paths: Optional[List[str]] = None


class IngestResponse(BaseModel):
    """Response model for ingestion results"""
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    documents_indexed: bool
    version: str


class EvaluationRequest(BaseModel):
    """Request for RAG triad evaluation"""
    question: str
    expected_answer: Optional[str] = None


class MetricsResponse(BaseModel):
    """Metrics for monitoring"""
    total_queries: int
    avg_latency_ms: float
    avg_faithfulness: Optional[float]
    documents_indexed: int


# ============== Global State ==============

class AppState:
    def __init__(self):
        self.legalmind: Optional[LegalMind] = None
        self.total_queries: int = 0
        self.total_latency: float = 0.0
        self.faithfulness_scores: List[float] = []
        self.initialized: bool = False


app_state = AppState()


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LegalMind on startup"""
    print("🚀 Starting LegalMind API...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    
    if not openai_key:
        print("⚠️ WARNING: OPENAI_API_KEY not set. API will not function properly.")
    else:
        app_state.legalmind = create_legalmind(
            openai_api_key=openai_key,
            cohere_api_key=cohere_key
        )
        
        # Auto-ingest sample docs if they exist
        sample_docs_path = "./data/sample_docs"
        if os.path.exists(sample_docs_path):
            try:
                chunks = app_state.legalmind.ingest_documents(directory_path=sample_docs_path)
                print(f"✅ Auto-ingested {chunks} chunks from sample_docs")
                app_state.initialized = True
            except Exception as e:
                print(f"⚠️ Could not auto-ingest: {e}")
        
        print("✅ LegalMind initialized successfully")
    
    yield
    
    print("👋 Shutting down LegalMind API...")


# ============== FastAPI App ==============

app = FastAPI(
    title="LegalMind API",
    description="AI Legal Knowledge Assistant with RAG, Citations, and Evaluation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Endpoints ==============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple frontend"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LegalMind - AI Legal Assistant</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .loading { animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        </style>
    </head>
    <body class="bg-gray-900 text-white min-h-screen">
        <div class="container mx-auto px-4 py-8 max-w-4xl">
            <!-- Header -->
            <div class="text-center mb-8">
                <h1 class="text-4xl font-bold text-blue-400 mb-2">⚖️ LegalMind</h1>
                <p class="text-gray-400">AI Legal Knowledge Assistant with RAG & Citations</p>
                <div class="mt-4 flex justify-center gap-4 text-sm">
                    <span class="px-3 py-1 bg-green-900 text-green-300 rounded-full">Hybrid Retrieval</span>
                    <span class="px-3 py-1 bg-blue-900 text-blue-300 rounded-full">Cross-Encoder Reranking</span>
                    <span class="px-3 py-1 bg-purple-900 text-purple-300 rounded-full">RAG Evaluation</span>
                </div>
            </div>
            
            <!-- Query Input -->
            <div class="bg-gray-800 rounded-lg p-6 mb-6">
                <label class="block text-sm font-medium text-gray-300 mb-2">Ask a Legal Question</label>
                <textarea 
                    id="question" 
                    rows="3" 
                    class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                    placeholder="What are the liability limitations in our contracts?"
                ></textarea>
                
                <div class="flex items-center gap-4 mt-4">
                    <label class="flex items-center gap-2 text-sm text-gray-300">
                        <input type="checkbox" id="runEval" class="rounded bg-gray-700 border-gray-600">
                        Run Evaluation (Faithfulness, Citations)
                    </label>
                    <button 
                        onclick="submitQuery()" 
                        id="submitBtn"
                        class="ml-auto px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition"
                    >
                        Ask LegalMind
                    </button>
                </div>
            </div>
            
            <!-- Results -->
            <div id="results" class="hidden">
                <!-- Answer -->
                <div class="bg-gray-800 rounded-lg p-6 mb-4">
                    <h2 class="text-lg font-semibold text-blue-400 mb-3">Answer</h2>
                    <div id="answer" class="text-gray-200 whitespace-pre-wrap"></div>
                </div>
                
                <!-- Metrics -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div class="bg-gray-800 rounded-lg p-4 text-center">
                        <div class="text-2xl font-bold text-green-400" id="confidence">-</div>
                        <div class="text-xs text-gray-400">Confidence</div>
                    </div>
                    <div class="bg-gray-800 rounded-lg p-4 text-center">
                        <div class="text-2xl font-bold text-blue-400" id="faithfulness">-</div>
                        <div class="text-xs text-gray-400">Faithfulness</div>
                    </div>
                    <div class="bg-gray-800 rounded-lg p-4 text-center">
                        <div class="text-2xl font-bold text-purple-400" id="citations">-</div>
                        <div class="text-xs text-gray-400">Citations</div>
                    </div>
                    <div class="bg-gray-800 rounded-lg p-4 text-center">
                        <div class="text-2xl font-bold text-yellow-400" id="latency">-</div>
                        <div class="text-xs text-gray-400">Latency (ms)</div>
                    </div>
                </div>
                
                <!-- Sources -->
                <div class="bg-gray-800 rounded-lg p-6">
                    <h2 class="text-lg font-semibold text-blue-400 mb-3">Sources Used</h2>
                    <div id="sources" class="space-y-2"></div>
                </div>
            </div>
            
            <!-- Loading -->
            <div id="loading" class="hidden text-center py-12">
                <div class="loading text-4xl mb-4">⚖️</div>
                <p class="text-gray-400">Analyzing legal documents...</p>
            </div>
            
            <!-- Architecture Info -->
            <div class="mt-8 text-center text-gray-500 text-sm">
                <p>Built with: LangChain | ChromaDB | BM25 | Cohere Reranker | OpenAI</p>
                <p class="mt-1">Deployed on: Kubernetes (GKE) with HPA & Prometheus</p>
            </div>
        </div>
        
        <script>
            async function submitQuery() {
                const question = document.getElementById('question').value;
                const runEval = document.getElementById('runEval').checked;
                const btn = document.getElementById('submitBtn');
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                
                if (!question.trim()) {
                    alert('Please enter a question');
                    return;
                }
                
                btn.disabled = true;
                btn.textContent = 'Processing...';
                loading.classList.remove('hidden');
                results.classList.add('hidden');
                
                try {
                    const response = await fetch('/api/v1/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question, run_evaluation: runEval })
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.detail || 'Query failed');
                    }
                    
                    // Display results
                    document.getElementById('answer').textContent = data.answer;
                    document.getElementById('confidence').textContent = data.confidence.toUpperCase();
                    document.getElementById('faithfulness').textContent = 
                        data.faithfulness_score ? data.faithfulness_score.toFixed(2) : 'N/A';
                    document.getElementById('citations').textContent = data.citations.length;
                    document.getElementById('latency').textContent = Math.round(data.latency_ms);
                    
                    // Sources
                    const sourcesDiv = document.getElementById('sources');
                    sourcesDiv.innerHTML = data.sources.map(s => `
                        <div class="bg-gray-700 rounded px-3 py-2 text-sm">
                            <span class="text-blue-300">${s.source_file || 'Unknown'}</span>
                            <span class="text-gray-400 ml-2">Chunk ${s.chunk_index}</span>
                            ${s.rerank_score ? `<span class="text-green-400 ml-2">Score: ${s.rerank_score.toFixed(3)}</span>` : ''}
                        </div>
                    `).join('');
                    
                    results.classList.remove('hidden');
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    btn.disabled = false;
                    btn.textContent = 'Ask LegalMind';
                    loading.classList.add('hidden');
                }
            }
            
            // Allow Enter key to submit
            document.getElementById('question').addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                    submitQuery();
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes probes"""
    return HealthResponse(
        status="healthy" if app_state.legalmind else "degraded",
        documents_indexed=app_state.initialized,
        version="1.0.0"
    )


@app.get("/ready")
async def readiness_check():
    """Readiness probe - only ready if documents are indexed"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="Documents not yet indexed")
    return {"status": "ready"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    avg_latency = app_state.total_latency / max(app_state.total_queries, 1)
    avg_faithfulness = (
        sum(app_state.faithfulness_scores) / len(app_state.faithfulness_scores)
        if app_state.faithfulness_scores else None
    )
    
    return MetricsResponse(
        total_queries=app_state.total_queries,
        avg_latency_ms=avg_latency,
        avg_faithfulness=avg_faithfulness,
        documents_indexed=len(app_state.legalmind.retriever.documents) if app_state.legalmind else 0
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_legalmind(request: QueryRequest):
    """
    Query LegalMind with a legal question
    
    - Performs hybrid retrieval (vector + BM25)
    - Reranks with cross-encoder
    - Generates answer with citations
    - Optionally runs evaluation agents
    """
    if not app_state.legalmind:
        raise HTTPException(status_code=503, detail="LegalMind not initialized")
    
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="No documents indexed")
    
    start_time = time.time()
    
    try:
        response = app_state.legalmind.query(
            question=request.question,
            filter_metadata=request.filter_metadata,
            run_evaluation=request.run_evaluation
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        app_state.total_queries += 1
        app_state.total_latency += latency_ms
        if response.faithfulness_score:
            app_state.faithfulness_scores.append(response.faithfulness_score)
        
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            citations=response.citations,
            confidence=response.confidence,
            retrieved_chunks=response.retrieved_chunks,
            reranked_chunks=response.reranked_chunks,
            sources=response.sources,
            faithfulness_score=response.faithfulness_score,
            context_precision=response.context_precision,
            citation_accuracy=response.citation_accuracy,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents into the knowledge base"""
    if not app_state.legalmind:
        raise HTTPException(status_code=503, detail="LegalMind not initialized")
    
    if not request.directory_path and not request.file_paths:
        raise HTTPException(status_code=400, detail="Provide directory_path or file_paths")
    
    try:
        chunks = app_state.legalmind.ingest_documents(
            directory_path=request.directory_path,
            file_paths=request.file_paths
        )
        app_state.initialized = True
        
        return IngestResponse(
            chunks_indexed=chunks,
            message=f"Successfully indexed {chunks} document chunks"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/evaluate")
async def evaluate_rag_triad(request: EvaluationRequest):
    """Run RAG Triad evaluation on a question"""
    if not app_state.legalmind or not app_state.initialized:
        raise HTTPException(status_code=503, detail="LegalMind not ready")
    
    try:
        result = app_state.legalmind.evaluate_rag_triad(
            question=request.question,
            expected_answer=request.expected_answer
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Main ==============

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
