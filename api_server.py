"""
Saarthi AI — FastAPI Chat Backend
==================================
Exposes a REST API that connects the SRM University frontend chat widget
to the RAG pipeline (Supabase pgvector retrieval + Groq LLM).

Endpoints:
  GET  /api/health       — Health check
  POST /api/chat         — Accept {query, session_id}, return {answer, sources}
"""
import os
import sys
import re
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    sys.exit(1)

try:
    from groq import Groq
except ImportError:
    print("\n[ERROR] groq package not found. Run: pip install groq")
    sys.exit(1)

sys.path.insert(0, str(PROJECT_ROOT))
from document_processor import process_documents, ingest_chunks_to_supabase
from retriever import (
    load_embedding_model,
    retrieve_relevant_chunks_supabase,
    build_vector_index,
    retrieve_relevant_chunks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("saarthi-api")

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

rag_state: dict = {
    "embedding_model": None,
    "db_url": None,
    "supabase_connected": False,
    "local_index_embeddings": None,
    "local_chunks": [],
}

def check_or_ingest_supabase(db_url: str, model):
    """Ensures Supabase database contains document chunks."""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_chunks;")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        if count == 0:
            log.info("Supabase database empty. Processing and ingesting local documents...")
            chunks = process_documents(DOCUMENTS_DIR)
            if chunks:
                texts = [c["text"] for c in chunks]
                embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                ingest_chunks_to_supabase(chunks, embeddings, db_url)
                log.info(f"Ingested {len(chunks)} chunks into Supabase pgvector.")
            else:
                log.warning("No documents found in 'documents/' to ingest.")
        else:
            log.info(f"Supabase ready: found {count} existing chunks in database.")
        return True
    except Exception as e:
        log.error(f"Failed to connect or ingest into Supabase: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("  Saarthi AI — RAG Chat API starting up")
    log.info("=" * 60)
    log.info("Loading embedding model (all-MiniLM-L6-v2)...")
    rag_state["embedding_model"] = load_embedding_model()
    log.info("Embedding model loaded.")

    db_url = os.getenv("DATABASE_URL", "").strip()
    rag_state["db_url"] = db_url

    if db_url and "YOUR_PASSWORD" not in db_url and "YOUR_SUPABASE_PASSWORD" not in db_url:
        rag_state["supabase_connected"] = check_or_ingest_supabase(db_url, rag_state["embedding_model"])
    else:
        log.warning("DATABASE_URL not configured — initializing local in-memory RAG index.")
        rag_state["supabase_connected"] = False

    # Initialize local in-memory index as fallback
    chunks = process_documents(DOCUMENTS_DIR)
    if chunks:
        emb, indexed = build_vector_index(chunks, rag_state["embedding_model"])
        rag_state["local_index_embeddings"] = emb
        rag_state["local_chunks"] = indexed
        log.info(f"Local vector index ready: {emb.shape[0]} chunks x {emb.shape[1]} dims")

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        log.info(f"Groq LLM: active (model: {GROQ_MODEL})")
    else:
        log.warning("GROQ_API_KEY not set — answers will use fallback text.")

    log.info("API ready at http://localhost:8000")
    log.info("=" * 60)
    yield
    log.info("Saarthi API shutting down.")

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
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default-session"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

def generate_answer_with_groq(query: str, retrieved_chunks: list[dict]) -> str | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        client = Groq(api_key=api_key)
        if retrieved_chunks:
            context_blocks = [
                f"[{i}] Source: {chunk['source']}\n{chunk['text']}"
                for i, chunk in enumerate(retrieved_chunks, 1)
            ]
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
            user_prompt = f"Context passages:\n{context_text}\n\nQuestion: {query}\n\nHelpful Answer:"
        else:
            system_prompt = (
                "You are Saarthi, a friendly campus AI assistant for "
                "SRM University Delhi-NCR, Sonepat (Haryana). "
                "Answer questions about admissions, courses, campus life, fees, "
                "placements, and university information helpfully and concisely. "
                "Keep responses to 2-4 sentences."
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

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "Saarthi AI Chat API",
        "supabase_connected": rag_state["supabase_connected"],
        "groq_llm": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "model": GROQ_MODEL,
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    log.info(f"[{req.session_id}] Query: {query!r}")
    retrieved = []

    if rag_state["supabase_connected"] and rag_state["db_url"]:
        retrieved = retrieve_relevant_chunks_supabase(
            query=query,
            db_url=rag_state["db_url"],
            model=rag_state["embedding_model"],
            top_k=5,
        )
        log.info(f"Retrieved {len(retrieved)} chunk(s) from Supabase pgvector")
    elif rag_state["local_chunks"] and rag_state["local_index_embeddings"] is not None:
        retrieved = retrieve_relevant_chunks(
            query=query,
            index_embeddings=rag_state["local_index_embeddings"],
            chunks=rag_state["local_chunks"],
            model=rag_state["embedding_model"],
            top_k=4,
        )
        log.info(f"Retrieved {len(retrieved)} chunk(s) from local in-memory index")

    answer = generate_answer_with_groq(query, retrieved)
    if answer is None:
        if retrieved:
            best = retrieved[0]
            answer = (
                f"Based on our documents: {best['text'][:300]}... "
                "For complete information, please contact the admissions office at +91-8816033301."
            )
        else:
            answer = (
                "Welcome to SRM University Delhi-NCR, Sonepat! "
                "For admissions and enquiries, please contact +91-8816033301."
            )

    url_pattern = re.compile(r"https?://[^\s)\]\"' >]+")
    extracted_urls = []
    for chunk in retrieved:
        for url in url_pattern.findall(chunk["text"]):
            clean_url = url.rstrip(".,;)")
            if clean_url not in extracted_urls:
                extracted_urls.append(clean_url)

    sources = extracted_urls[:4] if extracted_urls else ([chunk["source"] for chunk in retrieved] if retrieved else [])
    log.info(f"[{req.session_id}] Answer generated ({len(answer)} chars)")

    return ChatResponse(answer=answer, sources=sources)


# ===========================================================================
# BACKEND REAL-TIME VOICE PIPELINE (COMMENTED OUT / ON HOLD FOR STAGE 5)
# ===========================================================================
# Note: Voice pipeline backend integration is commented out below per project
# roadmap while focusing on the 4 RAG stages with the knowledge pack.
# It will be completed and re-activated in the final stage.
# ===========================================================================
# async def run_voice_response(...):
#     ...
# @app.websocket("/api/voice")
# async def voice_websocket(websocket: WebSocket):
#     ...

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
