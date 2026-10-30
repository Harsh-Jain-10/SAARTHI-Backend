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
import json
import base64
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import httpx

try:
    import websockets
except ImportError:
    websockets = None

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
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

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
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
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
# Real-Time Voice WebSocket (/api/voice)
# ---------------------------------------------------------------------------
async def run_voice_response(query_text: str, websocket: WebSocket, deepgram_key: str, groq_key: str):
    """
    RAG-grounded Groq generation followed by Deepgram Aura TTS audio streaming.
    Can be cancelled instantly by barge-in interruption.
    """
    try:
        await websocket.send_json({"type": "status", "status": "thinking"})

        # Stage 1: RAG context retrieval
        retrieved = []
        if rag_state["chunks"] and rag_state["index_embeddings"] is not None:
            retrieved = retrieve_relevant_chunks(
                query=query_text,
                index_embeddings=rag_state["index_embeddings"],
                chunks=rag_state["chunks"],
                model=rag_state["embedding_model"],
                top_k=2,
            )
            log.info(f"[VoiceWS] Retrieved {len(retrieved)} chunk(s) for query: {query_text!r}")

        voice_prompt = (
            "You are Saarthi, a friendly campus voice assistant for SRM University Delhi-NCR. "
            "Respond to the user in 1 or 2 concise, conversational spoken sentences. "
            "Do not use markdown, emojis, asterisks, bullet points, or URLs."
        )
        if retrieved:
            context_text = f"Context:\n{retrieved[0]['text']}"
        else:
            context_text = "No specific college documents loaded."

        # Stage 2: Groq streaming inference
        bot_answer = ""
        if groq_key:
            try:
                groq_client = Groq(api_key=groq_key)
                completion = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": voice_prompt},
                        {"role": "user", "content": f"{context_text}\n\nUser Question: {query_text}"},
                    ],
                    max_tokens=120,
                    temperature=0.4,
                    stream=True,
                )

                collected = []
                for chunk in completion:
                    if asyncio.current_task().cancelled():
                        log.info("[VoiceWS] Groq generation cancelled by barge-in.")
                        return
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        collected.append(delta)
                        await websocket.send_json({"type": "answer_chunk", "delta": delta})

                bot_answer = "".join(collected).strip()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"[VoiceWS] Groq streaming error: {e}")
                bot_answer = "Welcome to SRM University Delhi-NCR. Please contact our admissions desk at +91-8816033301 for details."
        else:
            bot_answer = "Welcome to SRM University Delhi-NCR. Groq API key is not configured."

        if not bot_answer:
            bot_answer = "Welcome to SRM University Delhi-NCR. How else may I assist you?"

        await websocket.send_json({"type": "answer", "text": bot_answer})

        # Stage 3: Deepgram Aura TTS audio streaming
        if deepgram_key and bot_answer:
            await websocket.send_json({"type": "status", "status": "speaking"})
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                async with http_client.stream(
                    "POST",
                    "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mp3",
                    headers={
                        "Authorization": f"Token {deepgram_key}",
                        "Content-Type": "application/json",
                    },
                    json={"text": bot_answer},
                ) as tts_res:
                    if tts_res.status_code == 200:
                        async for chunk in tts_res.aiter_bytes(chunk_size=4096):
                            if asyncio.current_task().cancelled():
                                log.info("[VoiceWS] TTS audio stream cancelled by barge-in.")
                                return
                            chunk_b64 = base64.b64encode(chunk).decode("utf-8")
                            await websocket.send_json({
                                "type": "audio",
                                "data": chunk_b64,
                                "mime": "audio/mp3",
                                "streaming": True,
                            })
                        await websocket.send_json({"type": "audio_end"})
                    else:
                        log.error(f"[VoiceWS] Deepgram Aura TTS returned status {tts_res.status_code}")

        await websocket.send_json({"type": "status", "status": "idle"})

    except asyncio.CancelledError:
        log.info("[VoiceWS] Voice response task successfully interrupted.")
        raise
    except Exception as e:
        log.error(f"[VoiceWS] Voice response error: {e}")
        try:
            await websocket.send_json({"type": "status", "status": "idle"})
        except Exception:
            pass


@app.websocket("/api/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional voice chat with browser.
    Pipeline:
      Browser Mic -> WebSocket -> Deepgram STT -> Groq LLM -> Deepgram Aura TTS -> WebSocket -> Browser Speaker
    Features:
      - Real-time streaming audio ingestion (binary chunks or base64 JSON).
      - Live interim transcription updates.
      - Groq LLM token streaming with RAG context.
      - Deepgram Aura MP3 audio streaming.
      - Sub-250ms barge-in interruption.
      - Clean resource disposal on disconnection.
    """
    await websocket.accept()
    log.info("[VoiceWS] Browser voice client connected.")

    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    current_response_task: asyncio.Task | None = None
    dg_ws = None
    dg_listener_task: asyncio.Task | None = None
    audio_buffer = bytearray()

    async def cancel_active_response():
        nonlocal current_response_task
        if current_response_task and not current_response_task.done():
            log.info("[VoiceWS] Barge-in! Cancelling active response task.")
            current_response_task.cancel()
            try:
                await current_response_task
            except (asyncio.CancelledError, Exception):
                pass
            current_response_task = None
            try:
                await websocket.send_json({"type": "interrupted"})
                await websocket.send_json({"type": "status", "status": "listening"})
            except Exception:
                pass

    # Deepgram Live WebSocket Listener
    async def listen_deepgram(dg_socket):
        nonlocal current_response_task
        try:
            async for raw_msg in dg_socket:
                dg_msg = json.loads(raw_msg)
                msg_type = dg_msg.get("type")

                if msg_type == "Results":
                    is_final = dg_msg.get("is_final", False)
                    speech_final = dg_msg.get("speech_final", False)
                    channel = dg_msg.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "").strip()
                        if transcript:
                            # Barge-in: if user starts speaking while bot is thinking or speaking
                            if current_response_task and not current_response_task.done():
                                await cancel_active_response()

                            # Send live transcript to browser
                            await websocket.send_json({
                                "type": "transcript",
                                "text": transcript,
                                "is_final": speech_final or is_final,
                            })

                            # On finalized speech utterance, trigger RAG + Groq + TTS response
                            if speech_final:
                                log.info(f"[VoiceWS] Finalized utterance: {transcript!r}")
                                if current_response_task and not current_response_task.done():
                                    current_response_task.cancel()
                                current_response_task = asyncio.create_task(
                                    run_voice_response(transcript, websocket, deepgram_key, groq_key)
                                )
                elif msg_type == "UtteranceEnd":
                    log.info("[VoiceWS] Deepgram UtteranceEnd event received.")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning(f"[VoiceWS] Deepgram listener ended: {e}")

    try:
        await websocket.send_json({
            "type": "status",
            "status": "ready",
            "message": "Connected to Saarthi Voice Engine",
        })

        # Connect to Deepgram live streaming WebSocket if available
        if websockets is not None and deepgram_key:
            try:
                dg_url = (
                    "wss://api.deepgram.com/v1/listen?"
                    "model=nova-2&smart_format=true&interim_results=true"
                    "&endpointing=300&utterance_end_ms=1000"
                )
                dg_ws = await websockets.connect(
                    dg_url,
                    additional_headers={"Authorization": f"Token {deepgram_key}"}
                )
                dg_listener_task = asyncio.create_task(listen_deepgram(dg_ws))
                log.info("[VoiceWS] Connected to Deepgram Live Streaming STT.")
            except Exception as e:
                log.warning(f"[VoiceWS] Could not connect to Deepgram live websocket, fallback to chunk buffering: {e}")
                dg_ws = None

        # Main receive loop for browser messages
        while True:
            recv = await websocket.receive()

            # Handle binary audio frame from browser
            if recv.get("bytes"):
                audio_bytes = recv["bytes"]
                audio_buffer.extend(audio_bytes)
                if dg_ws is not None:
                    try:
                        await dg_ws.send(audio_bytes)
                    except Exception as e:
                        log.warning(f"[VoiceWS] Error forwarding binary chunk to Deepgram: {e}")
                continue

            # Handle text/JSON messages
            if recv.get("text"):
                try:
                    data = json.loads(recv["text"])
                except Exception:
                    continue

                msg_type = data.get("type")

                # Client-initiated barge-in / interrupt
                if msg_type == "interrupt":
                    await cancel_active_response()
                    continue

                # Browser sends start stream signal
                if msg_type == "start_stream":
                    await cancel_active_response()
                    audio_buffer.clear()
                    await websocket.send_json({"type": "status", "status": "listening"})
                    continue

                # Browser sends base64 audio chunk
                if msg_type == "audio_chunk":
                    chunk_b64 = data.get("data", "")
                    if chunk_b64:
                        raw_chunk = base64.b64decode(chunk_b64)
                        audio_buffer.extend(raw_chunk)
                        if dg_ws is not None:
                            try:
                                await dg_ws.send(raw_chunk)
                            except Exception as e:
                                log.warning(f"[VoiceWS] Error forwarding b64 chunk to Deepgram: {e}")
                    continue

                # Browser signals end of speech stream
                if msg_type in ("stop_stream", "stop_audio"):
                    # If live STT is active, tell Deepgram to finalize
                    if dg_ws is not None:
                        try:
                            # Deepgram finalize message
                            await dg_ws.send(json.dumps({"type": "CloseStream"}))
                        except Exception:
                            pass
                    elif audio_buffer and deepgram_key:
                        # Fallback: transcribe buffered audio via REST
                        await websocket.send_json({"type": "status", "status": "transcribing"})
                        user_text = ""
                        try:
                            async with httpx.AsyncClient(timeout=20.0) as http_client:
                                res = await http_client.post(
                                    "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
                                    headers={
                                        "Authorization": f"Token {deepgram_key}",
                                        "Content-Type": data.get("mime", "audio/webm"),
                                    },
                                    content=bytes(audio_buffer),
                                )
                                if res.status_code == 200:
                                    ch = res.json().get("results", {}).get("channels", [])
                                    if ch and ch[0].get("alternatives"):
                                        user_text = ch[0]["alternatives"][0].get("transcript", "").strip()
                        except Exception as e:
                            log.error(f"[VoiceWS] Fallback REST STT error: {e}")

                        audio_buffer.clear()
                        if user_text:
                            await websocket.send_json({"type": "transcript", "text": user_text, "is_final": True})
                            if current_response_task and not current_response_task.done():
                                current_response_task.cancel()
                            current_response_task = asyncio.create_task(
                                run_voice_response(user_text, websocket, deepgram_key, groq_key)
                            )
                        else:
                            await websocket.send_json({"type": "status", "status": "idle"})
                    continue

                # Browser sends complete recorded audio (backward compatibility)
                if msg_type in ("audio_complete", "transcribe_and_answer"):
                    audio_b64 = data.get("audio", "")
                    if not audio_b64:
                        continue
                    raw_audio = base64.b64decode(audio_b64)
                    await websocket.send_json({"type": "status", "status": "transcribing"})

                    user_transcript = ""
                    if deepgram_key:
                        try:
                            async with httpx.AsyncClient(timeout=20.0) as http_client:
                                dg_res = await http_client.post(
                                    "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
                                    headers={
                                        "Authorization": f"Token {deepgram_key}",
                                        "Content-Type": data.get("mime", "audio/webm"),
                                    },
                                    content=raw_audio,
                                )
                                if dg_res.status_code == 200:
                                    channels = dg_res.json().get("results", {}).get("channels", [])
                                    if channels and channels[0].get("alternatives"):
                                        user_transcript = channels[0]["alternatives"][0].get("transcript", "").strip()
                        except Exception as e:
                            log.error(f"[VoiceWS] Deepgram transcription error: {e}")

                    if user_transcript:
                        await websocket.send_json({"type": "transcript", "text": user_transcript, "is_final": True})
                        if current_response_task and not current_response_task.done():
                            current_response_task.cancel()
                        current_response_task = asyncio.create_task(
                            run_voice_response(user_transcript, websocket, deepgram_key, groq_key)
                        )
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Could not recognize audio. Please try speaking again.",
                        })
                        await websocket.send_json({"type": "status", "status": "idle"})
                    continue

                # Direct text query via WebSocket
                if msg_type == "text_query":
                    query_text = data.get("text", "").strip()
                    if query_text:
                        await cancel_active_response()
                        current_response_task = asyncio.create_task(
                            run_voice_response(query_text, websocket, deepgram_key, groq_key)
                        )
                    continue

    except WebSocketDisconnect:
        log.info("[VoiceWS] Browser voice client disconnected.")
    except Exception as e:
        log.error(f"[VoiceWS] Unexpected WebSocket error: {e}")
    finally:
        # Resource cleanup
        if current_response_task and not current_response_task.done():
            current_response_task.cancel()
        if dg_listener_task and not dg_listener_task.done():
            dg_listener_task.cancel()
        if dg_ws is not None:
            try:
                await dg_ws.close()
            except Exception:
                pass
        log.info("[VoiceWS] Cleaned up voice session resources.")


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
