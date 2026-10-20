"""
Saarthi AI — FastAPI Chat Backend
==================================
Exposes a REST API that connects the SRM University frontend chat widget
to the RAG pipeline (document retrieval + Groq LLM).

Endpoints:
  GET  /api/health       — Health check
  POST /api/chat         — Accept {query, session_id}, return {answer, sources}

CORS is opened for:
  - http://localhost:5173  (Vite dev server)
  - http://127.0.0.1:5173
  - http://localhost:4173  (Vite preview)

Run with:
  python api_server.py
  or
  uvicorn api_server:app --reload --port 8000
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

# ---------------------------------------------------------------------------
# Dependency check — give a clear error before FastAPI import
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Run:  pip install fastapi uvicorn[standard] pydantic")
    sys.exit(1)

try:
    from groq import Groq
except ImportError:
    print("\n[ERROR] groq package not found. Run: pip install groq")
    sys.exit(1)

# RAG pipeline modules (same directory)
sys.path.insert(0, str(PROJECT_ROOT))
from document_processor import process_documents
from retriever import load_embedding_model, build_vector_index, retrieve_relevant_chunks

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("saarthi-api")

# ---------------------------------------------------------------------------
# RAG State (loaded once at startup)
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
GROQ_MODEL = "llama-3.3-70b-versatile"

rag_state: dict = {
    "embedding_model": None,
    "index_embeddings": None,
    "chunks": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding model and build vector index at startup."""
    log.info("=" * 60)
    log.info("  Saarthi AI — RAG Chat API starting up")
    log.info("=" * 60)

    log.info("Loading embedding model (all-MiniLM-L6-v2)...")
    rag_state["embedding_model"] = load_embedding_model()
    log.info("Embedding model loaded.")

    log.info(f"Processing documents from: {DOCUMENTS_DIR}")
    chunks = process_documents(DOCUMENTS_DIR)

    if chunks:
        emb, indexed = build_vector_index(chunks, rag_state["embedding_model"])
        rag_state["index_embeddings"] = emb
        rag_state["chunks"] = indexed
        log.info(f"Vector index built: {emb.shape[0]} chunks x {emb.shape[1]} dims")
    else:
        import numpy as np
        rag_state["index_embeddings"] = np.empty((0, 384), dtype="float32")
        rag_state["chunks"] = []
        log.warning(
            "No documents found in 'documents/' — RAG will use Groq general knowledge."
        )

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        log.info(f"Groq LLM: active (model: {GROQ_MODEL})")
    else:
        log.warning("GROQ_API_KEY not set — answers will use fallback text.")

    log.info("API ready at http://localhost:8000")
    log.info("=" * 60)

    yield  # Server runs here

    log.info("Saarthi API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Saarthi AI Chat API",
    description="RAG-powered campus assistant for SRM University Delhi-NCR",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default-session"


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []


# ---------------------------------------------------------------------------
# Helper: Groq LLM answer generation
# ---------------------------------------------------------------------------
def generate_answer_with_groq(query: str, retrieved_chunks: list[dict]) -> str | None:
    """Call Groq with retrieved context. Returns None if API key missing."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        client = Groq(api_key=api_key)

        if retrieved_chunks:
            context_blocks = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_blocks.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
            context_text = "\n\n".join(context_blocks)

            system_prompt = (
                "You are Saarthi, a friendly and helpful campus AI assistant for "
                "SRM University Delhi-NCR, Sonepat. "
                "Answer the user's question using the provided context passages. "
                "If the context doesn't contain sufficient information, use your general "
                "knowledge about the university to give a helpful answer. "
                "Keep responses clear, concise, and conversational (2-4 sentences). "
                "Do not fabricate specific numbers, fees, or dates not in the context."
            )
            user_prompt = (
                f"Context passages:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                f"Helpful Answer:"
            )
        else:
            # No RAG context — use general knowledge
            system_prompt = (
                "You are Saarthi, a friendly campus AI assistant for "
                "SRM University Delhi-NCR, Sonepat (Haryana). "
                "Answer questions about admissions, courses, campus life, fees, "
                "placements, and university information helpfully and concisely. "
                "Keep responses to 2-4 sentences. For specific numbers or dates, "
                "advise the user to verify with the admissions office."
            )
            user_prompt = f"Question: {query}\n\nHelpful Answer:"

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        log.error(f"Groq API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    chunks_count = len(rag_state["chunks"])
    groq_ready = bool(os.getenv("GROQ_API_KEY", "").strip())
    return {
        "status": "ok",
        "service": "Saarthi AI Chat API",
        "rag_chunks": chunks_count,
        "groq_llm": groq_ready,
        "model": GROQ_MODEL,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    log.info(f"[{req.session_id}] Query: {query!r}")

    # Stage 5: Retrieve relevant chunks
    retrieved = []
    if rag_state["chunks"] and rag_state["index_embeddings"] is not None:
        retrieved = retrieve_relevant_chunks(
            query=query,
            index_embeddings=rag_state["index_embeddings"],
            chunks=rag_state["chunks"],
            model=rag_state["embedding_model"],
            top_k=3,
        )
        log.info(f"Retrieved {len(retrieved)} chunk(s)")

    # Stage 6: Generate grounded answer
    answer = generate_answer_with_groq(query, retrieved)

    if answer is None:
        # Fallback if Groq is unavailable
        if retrieved:
            best = retrieved[0]
            answer = (
                f"Based on our documents: {best['text'][:300]}... "
                "For complete information, please contact the admissions office at "
                "+91-8816033301."
            )
        else:
            answer = (
                "Welcome to SRM University Delhi-NCR, Sonepat! "
                "For admissions and enquiries, please contact +91-8816033301 "
                "or visit the campus at 39, Rajiv Gandhi Education City, Sonepat-131029."
            )

    # Build sources list
    sources = list({chunk["source"] for chunk in retrieved}) if retrieved else []

    log.info(f"[{req.session_id}] Answer generated ({len(answer)} chars)")

    return ChatResponse(answer=answer, sources=sources)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
