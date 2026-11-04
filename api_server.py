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
import json
import base64
import asyncio
import re
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import httpx

try:
    import websockets
except ImportError:
    websockets = None

from dotenv import load_dotenv
import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

def retrieve_knowledge(query: str, top_k: int = 4) -> list[dict]:
    """Retrieve relevant chunks from Supabase pgvector or local in-memory fallback."""
    if rag_state.get("supabase_connected") and rag_state.get("db_url"):
        try:
            return retrieve_relevant_chunks_supabase(
                query=query,
                db_url=rag_state["db_url"],
                model=rag_state["embedding_model"],
                top_k=top_k,
            )
        except Exception as e:
            log.warning(f"Supabase retrieval error, falling back to local index: {e}")

    if rag_state.get("local_chunks") and rag_state.get("local_index_embeddings") is not None:
        return retrieve_relevant_chunks(
            query=query,
            index_embeddings=rag_state["local_index_embeddings"],
            chunks=rag_state["local_chunks"],
            model=rag_state["embedding_model"],
            top_k=top_k,
        )
    return []

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    log.info(f"[{req.session_id}] Query: {query!r}")
    retrieved = retrieve_knowledge(query, top_k=5)
    log.info(f"Retrieved {len(retrieved)} chunk(s)")

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
# BACKEND REAL-TIME VOICE PIPELINE
# ===========================================================================

async def run_voice_response(
    query_text: str,
    websocket: WebSocket,
    deepgram_key: str,
    groq_key: str,
    on_speaking_start=None,
):
    """
    RAG-grounded Groq answer generation followed by Deepgram Aura TTS audio streaming.
    Can be cancelled instantly by barge-in interruption.
    """
    try:
        await websocket.send_json({"type": "status", "status": "thinking"})

        # Stage 1: RAG context retrieval
        retrieved = retrieve_knowledge(query_text, top_k=2)
        log.info(f"[VoiceWS] Retrieved {len(retrieved)} chunk(s) for query: {query_text!r}")

        voice_prompt = (
            "You are Saarthi, a friendly, ultra-fast campus voice assistant for SRM University Delhi-NCR, Sonepat. "
            "Respond directly to the user's question in 1 or 2 concise, natural spoken sentences (maximum 35 words). "
            "Strict rules for voice output:\n"
            "1. Keep answers conversational, clear, and direct.\n"
            "2. Never use bullet points, numbered lists, markdown symbols (*, _, #, `), emojis, or URLs.\n"
            "3. Spell out phone numbers or acronyms simply so they sound completely natural when spoken aloud."
        )

        if retrieved:
            context_text = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(retrieved))
        else:
            context_text = "No specific college documents matched."

        bot_answer = ""
        if groq_key:
            try:
                groq_client = Groq(api_key=groq_key)
                completion = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": voice_prompt},
                        {"role": "user", "content": f"Context:\n{context_text}\n\nUser Question: {query_text}"},
                    ],
                    max_tokens=90,
                    temperature=0.3,
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
                if retrieved:
                    bot_answer = f"Based on SRM University records: {retrieved[0]['text'][:140]}."
                else:
                    bot_answer = "Welcome to SRM University Delhi-NCR. Please contact admissions at 8816033301."
        else:
            bot_answer = "Welcome to SRM University Delhi-NCR. Groq API key is not configured."

        if not bot_answer:
            bot_answer = "Welcome to SRM University Delhi-NCR. How else may I assist you?"

        # Clean any accidental markdown or symbols for speech
        clean_speech = re.sub(r"[*_#`~[\]()]", " ", bot_answer)
        clean_speech = re.sub(r"https?://\S+", "", clean_speech)
        clean_speech = re.sub(r"\s+", " ", clean_speech).strip()

        await websocket.send_json({"type": "answer", "text": bot_answer})

        # Stage 3: Deepgram Aura TTS audio streaming
        if deepgram_key and clean_speech:
            if on_speaking_start:
                on_speaking_start()
            await websocket.send_json({"type": "status", "status": "speaking"})
            try:
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    async with http_client.stream(
                        "POST",
                        "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mp3",
                        headers={
                            "Authorization": f"Token {deepgram_key}",
                            "Content-Type": "application/json",
                        },
                        json={"text": clean_speech},
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
                            log.error(f"[VoiceWS] Deepgram Aura TTS status {tts_res.status_code}")
                            await websocket.send_json({"type": "audio_end"})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"[VoiceWS] TTS streaming exception: {e}")
                await websocket.send_json({"type": "audio_end"})
        else:
            # Fallback to client-side speech synthesis
            await websocket.send_json({"type": "audio_end"})

        await websocket.send_json({"type": "status", "status": "idle"})

    except asyncio.CancelledError:
        log.info("[VoiceWS] Voice response task interrupted.")
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
      Browser Mic -> WebSocket -> Deepgram STT (Nova-2) -> Groq LLM -> Deepgram Aura TTS -> WebSocket -> Browser Speaker
    """
    await websocket.accept()
    log.info("[VoiceWS] Browser voice client connected.")

    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    current_response_task: asyncio.Task | None = None
    dg_ws = None
    dg_listener_task: asyncio.Task | None = None
    audio_buffer = bytearray()
    is_bot_speaking = False
    latest_transcript = ""
    incoming_mime = "audio/webm"

    def on_speaking_start():
        nonlocal is_bot_speaking
        is_bot_speaking = True

    async def cancel_active_response():
        nonlocal current_response_task, is_bot_speaking
        is_bot_speaking = False
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

    async def stop_deepgram_session():
        nonlocal dg_ws, dg_listener_task
        if dg_listener_task and not dg_listener_task.done():
            dg_listener_task.cancel()
            try:
                await dg_listener_task
            except (asyncio.CancelledError, Exception):
                pass
            dg_listener_task = None
        if dg_ws is not None:
            try:
                await dg_ws.send(json.dumps({"type": "CloseStream"}))
                await dg_ws.close()
            except Exception:
                pass
            dg_ws = None

    async def start_deepgram_session():
        nonlocal dg_ws, dg_listener_task
        await stop_deepgram_session()
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
                log.warning(f"[VoiceWS] Could not connect to Deepgram live websocket: {e}")
                dg_ws = None

    async def trigger_answer(text: str):
        nonlocal current_response_task, is_bot_speaking, latest_transcript
        if not text or not text.strip():
            return
        log.info(f"[VoiceWS] Triggering answer for speech: {text!r}")
        if current_response_task and not current_response_task.done():
            current_response_task.cancel()
        is_bot_speaking = False
        latest_transcript = ""
        current_response_task = asyncio.create_task(
            run_voice_response(text.strip(), websocket, deepgram_key, groq_key, on_speaking_start=on_speaking_start)
        )

    async def listen_deepgram(dg_socket):
        nonlocal latest_transcript, is_bot_speaking
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
                            latest_transcript = transcript
                            await websocket.send_json({
                                "type": "transcript",
                                "text": transcript,
                                "is_final": speech_final or is_final,
                            })

                            # Barge-in: only cancel if bot is actively speaking out loud
                            if is_bot_speaking:
                                log.info(f"[VoiceWS] User spoke while bot speaking: {transcript!r}. Interrupting.")
                                await cancel_active_response()

                            if speech_final:
                                log.info(f"[VoiceWS] Finalized speech utterance: {transcript!r}")
                                await trigger_answer(transcript)

                elif msg_type == "UtteranceEnd":
                    log.info(f"[VoiceWS] Deepgram UtteranceEnd event. latest_transcript: {latest_transcript!r}")
                    if latest_transcript:
                        await trigger_answer(latest_transcript)

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

        while True:
            recv = await websocket.receive()

            # Handle binary audio frame from browser (e.g. MediaRecorder WebM/Opus)
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

                if msg_type == "client_speaking_start":
                    is_bot_speaking = True
                    continue

                if msg_type == "client_speaking_end":
                    is_bot_speaking = False
                    continue

                if msg_type == "interrupt":
                    await cancel_active_response()
                    continue

                if msg_type == "start_stream":
                    await cancel_active_response()
                    audio_buffer.clear()
                    latest_transcript = ""
                    incoming_mime = data.get("mime", "audio/webm")
                    await start_deepgram_session()
                    await websocket.send_json({"type": "status", "status": "listening"})
                    continue

                if msg_type == "audio_chunk":
                    chunk_b64 = data.get("data", "")
                    if chunk_b64:
                        raw_chunk = base64.b64decode(chunk_b64)
                        audio_buffer.extend(raw_chunk)
                        if dg_ws is not None:
                            try:
                                await dg_ws.send(raw_chunk)
                            except Exception as e:
                                log.warning(f"[VoiceWS] Error forwarding b64 chunk: {e}")
                    continue

                if msg_type in ("stop_stream", "stop_audio"):
                    # Stop listening; finalize speech
                    if latest_transcript:
                        log.info(f"[VoiceWS] stop_stream received with pending transcript: {latest_transcript!r}")
                        await stop_deepgram_session()
                        await trigger_answer(latest_transcript)
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
                                        "Content-Type": incoming_mime or "audio/webm",
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
                        await stop_deepgram_session()
                        if user_text:
                            await websocket.send_json({"type": "transcript", "text": user_text, "is_final": True})
                            await trigger_answer(user_text)
                        else:
                            await websocket.send_json({"type": "status", "status": "idle"})
                    else:
                        await stop_deepgram_session()
                        await websocket.send_json({"type": "status", "status": "idle"})
                    continue

                if msg_type == "text_query":
                    query_text = data.get("text", "").strip()
                    if query_text:
                        await cancel_active_response()
                        await trigger_answer(query_text)
                    continue

    except WebSocketDisconnect:
        log.info("[VoiceWS] Browser voice client disconnected.")
    except Exception as e:
        log.error(f"[VoiceWS] Unexpected WebSocket error: {e}")
    finally:
        if current_response_task and not current_response_task.done():
            current_response_task.cancel()
        await stop_deepgram_session()
        log.info("[VoiceWS] Cleaned up voice session resources.")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
