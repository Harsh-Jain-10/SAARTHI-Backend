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
import time
import hashlib
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
    from groq import AsyncGroq
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
    retrieve_relevant_chunks_lexical,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("saarthi-api")

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Jf701ovNUg1ZFQugpfXU").strip()
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5").strip()

rag_state: dict = {
    "embedding_model": None,
    "db_url": None,
    "supabase_connected": False,
    "local_index_embeddings": None,
    "local_chunks": [],
}
chat_sessions: dict[str, list[dict]] = {}

def check_or_ingest_supabase(db_url: str, model, chunks: list[dict]) -> bool:
    """Sync Supabase chunks whenever local source text changes."""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT source, chunk_index, md5(content) FROM document_chunks;")
        existing = {(row[0], int(row[1]), row[2]) for row in cursor.fetchall()}
        cursor.close()
        conn.close()

        expected = {
            (chunk["source"], int(chunk["chunk_number"]), hashlib.md5(chunk["text"].encode("utf-8")).hexdigest())
            for chunk in chunks
        }
        if existing != expected:
            log.info("Knowledge base changed (%d local chunks); refreshing Supabase index.", len(chunks))
            embeddings = model.encode(
                [chunk["text"] for chunk in chunks],
                convert_to_numpy=True,
                show_progress_bar=False,
            ) if chunks else []
            ingest_chunks_to_supabase(chunks, embeddings, db_url)
        else:
            log.info("Supabase vector index is current (%d chunks).", len(existing))
        return True
    except Exception as e:
        log.error(f"Failed to connect or ingest into Supabase: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("  Saarthi AI — RAG Chat API starting up")
    log.info("=" * 60)
    chunks = process_documents(DOCUMENTS_DIR)
    rag_state["local_chunks"] = chunks
    log.info("Loaded %d deduplicated knowledge chunks.", len(chunks))
    log.info("Loading embedding model (all-MiniLM-L6-v2)...")
    try:
        rag_state["embedding_model"] = load_embedding_model()
        log.info("Embedding model loaded.")
    except Exception as e:
        # Some managed Windows installations block scikit-learn's native DLLs.
        # Keep the API and document-grounded lexical fallback available.
        rag_state["embedding_model"] = None
        log.warning("Embedding model unavailable; using local lexical RAG fallback: %s", e)

    db_url = os.getenv("DATABASE_URL", "").strip()
    rag_state["db_url"] = db_url

    if rag_state["embedding_model"] and db_url and "YOUR_PASSWORD" not in db_url and "YOUR_SUPABASE_PASSWORD" not in db_url:
        rag_state["supabase_connected"] = check_or_ingest_supabase(
            db_url, rag_state["embedding_model"], chunks
        )
    else:
        log.warning("Supabase vector search unavailable — using local document retrieval.")
        rag_state["supabase_connected"] = False

    # Initialize local in-memory index as fallback
    if chunks:
        if rag_state["embedding_model"]:
            emb, indexed = build_vector_index(chunks, rag_state["embedding_model"])
            rag_state["local_index_embeddings"] = emb
            rag_state["local_chunks"] = indexed
            log.info(f"Local vector index ready: {emb.shape[0]} chunks x {emb.shape[1]} dims")
        else:
            log.info("Loaded %d document chunks for lexical fallback retrieval.", len(chunks))

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

async def generate_answer_with_groq(
    query: str,
    retrieved_chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> str | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        client = AsyncGroq(api_key=api_key)
        if retrieved_chunks:
            context_blocks = [
                re.sub(r"https?://\S+", "", chunk["text"])
                for chunk in retrieved_chunks
            ]
            context_text = "\n\n".join(context_blocks)
            system_prompt = (
                "You are Saarthi, a friendly and helpful campus AI assistant for "
                "SRM University Delhi-NCR, Sonepat. "
                "Answer using the provided university records. Lead with the direct answer, "
                "then add a brief practical detail or next step if it helps the student. "
                "Sound warm and natural without generic greetings or repeated offers. "
                "Do not mention source files, citations, or website URLs in the reply. "
                "If the records do not contain an exact fact, avoid the stock phrase 'I don't have this information'. "
                "Give safe context, ask one precise clarifying question, or suggest the relevant campus team; never guess. "
                "Keep it to 2-3 concise sentences. "
                "Never fabricate university details, numbers, fees, or dates. "
                "Always use the Indian number system for currency and large numbers: "
                "say 'lakh' instead of 'hundred thousand' and 'crore' instead of 'ten million'. "
                "For example, say '3 lakh rupees' not 'three hundred thousand rupees', "
                "and '1.5 crore' not '15 million'."
            )
            user_prompt = f"Student-support records:\n{context_text}\n\nCurrent question: {query}"
        else:
            system_prompt = (
                "You are Saarthi, a friendly campus AI assistant for "
                "SRM University Delhi-NCR, Sonepat (Haryana). "
                "Be warm and practical. For undocumented exact facts, avoid the stock phrase "
                "'I don't have this information'; give safe context, ask one useful question, "
                "or suggest the relevant campus team without guessing. Avoid generic greetings. "
                "Always use the Indian number system: say 'lakh' not 'hundred thousand', "
                "'crore' not 'ten million'."
            )
            user_prompt = f"Question: {query}\n\nHelpful Answer:"

        async with client:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend((conversation_history or [])[-6:])
            messages.append({"role": "user", "content": user_prompt})
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=220,
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
        "rag_mode": (
            "supabase-vector" if rag_state["supabase_connected"]
            else "local-vector" if rag_state.get("embedding_model")
            else "local-bm25"
        ),
        "knowledge_chunks": len(rag_state.get("local_chunks", [])),
        "groq_llm": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "model": GROQ_MODEL,
    }

def retrieve_knowledge(query: str, top_k: int = 4) -> list[dict]:
    """Retrieve relevant chunks from Supabase pgvector or local in-memory fallback."""
    model = rag_state.get("embedding_model")
    if model and rag_state.get("supabase_connected") and rag_state.get("db_url"):
        try:
            return retrieve_relevant_chunks_supabase(
                query=query,
                db_url=rag_state["db_url"],
                model=rag_state["embedding_model"],
                top_k=top_k,
            )
        except Exception as e:
            log.warning(f"Supabase retrieval error, falling back to local index: {e}")

    if model and rag_state.get("local_chunks") and rag_state.get("local_index_embeddings") is not None:
        return retrieve_relevant_chunks(
            query=query,
            index_embeddings=rag_state["local_index_embeddings"],
            chunks=rag_state["local_chunks"],
            model=rag_state["embedding_model"],
            top_k=top_k,
        )
    if rag_state.get("local_chunks"):
        return retrieve_relevant_chunks_lexical(query, rag_state["local_chunks"], top_k=top_k)
    return []

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    log.info(f"[{req.session_id}] Query: {query!r}")
    history = chat_sessions.setdefault(req.session_id, [])
    previous_user_question = next(
        (item["content"] for item in reversed(history) if item.get("role") == "user"),
        "",
    )
    is_follow_up = bool(re.search(
        r"\b(it|that|those|they|them|there|same)\b|\b(what|how) about\b|^\s*(and|also)\b",
        query.lower(),
    ))
    retrieval_query = (
        f"{previous_user_question} {query}".strip()
        if previous_user_question and is_follow_up
        else query
    )
    retrieved = retrieve_knowledge(retrieval_query, top_k=5)
    log.info(f"Retrieved {len(retrieved)} chunk(s)")

    answer = await generate_answer_with_groq(query, retrieved, history)
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

    log.info(f"[{req.session_id}] Answer generated ({len(answer)} chars)")
    history.extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ])
    del history[:-8]
    if len(chat_sessions) > 500:
        oldest_session = next(iter(chat_sessions))
        if oldest_session != req.session_id:
            chat_sessions.pop(oldest_session, None)

    # Keep provenance internal; student-facing answers should stay conversational.
    return ChatResponse(answer=answer, sources=[])


# ===========================================================================
# BACKEND REAL-TIME VOICE PIPELINE
# ===========================================================================

async def run_voice_response(
    query_text: str,
    websocket: WebSocket,
    deepgram_key: str,
    groq_key: str,
    on_speaking_start=None,
    conversation_history: list[dict] | None = None,
):
    """Stream a grounded Groq answer into sentence-sized TTS requests."""
    turn_started_at = time.perf_counter()
    retrieval_ms = first_token_ms = first_audio_ms = None
    tts_worker_task: asyncio.Task | None = None
    tts_http_client: httpx.AsyncClient | None = None
    try:
        await websocket.send_json({"type": "status", "status": "thinking"})
        recent_history = (conversation_history or [])[-6:]
        previous_user_question = next(
            (item["content"] for item in reversed(recent_history) if item.get("role") == "user"),
            "",
        )
        is_follow_up = bool(re.search(
            r"\b(it|that|those|they|them|there|same)\b|\b(what|how) about\b|^\s*(and|also)\b",
            query_text.lower(),
        ))
        retrieval_query = (
            f"{previous_user_question} {query_text}".strip()
            if previous_user_question and is_follow_up
            else query_text
        )
        retrieved = retrieve_knowledge(retrieval_query, top_k=3)
        retrieval_ms = round((time.perf_counter() - turn_started_at) * 1000)
        log.info("[VoiceWS] Retrieved %d chunk(s) for query: %r", len(retrieved), query_text)

        voice_prompt = (
            "You are Saarthi, a capable and considerate student-support assistant for SRM University Delhi-NCR, Sonepat. "
            "Sound like a helpful campus support person: warm, calm, and natural, without sounding scripted or overly excited. "
            "Answer the question first using the college records. Add one practical detail or next step when it helps. "
            "Acknowledge a student's concern briefly when they express one; do not add generic empathy to routine questions. "
            "Ask a short follow-up only when needed to clarify the request or offer a directly relevant next step. "
            "Do not repeat greetings or end every answer with an offer. Never claim you completed an action you cannot perform. "
            "Do not mention source files, citations, or website URLs in the spoken reply. "
            "Use 2 or 3 short spoken sentences, usually under 50 words. Never use bullets, markdown, emojis, or URLs. "
            "For example, for a fee question: 'B.Tech CSE tuition is 3 lakh rupees per year for 2027–28. "
            "Hostel and other charges are separate; I can check those too if useful.' "
            "If a precise fact is not documented, do not use the stock phrase 'I don't have this information'. "
            "Give safe context, ask one useful clarifying question, or suggest the relevant campus team; never guess. "
            "Speak phone numbers and acronyms naturally. Use the Indian number system for currency and large numbers."
        )
        context_text = "\n\n".join(
            re.sub(r"https?://\S+", "", chunk["text"])
            for chunk in retrieved
        ) or "No specific college documents matched."
        collected: list[str] = []
        speech_buffer = ""
        tts_started = False
        any_audio_sent = False
        session_uses_premade_voice = False
        tts_http_client = httpx.AsyncClient(timeout=30.0)

        async def synthesize_segment(text: str) -> bool:
            nonlocal tts_started, any_audio_sent, session_uses_premade_voice
            speech = re.sub(r"[*_#`~\[\]()]+", " ", text)
            speech = re.sub(r"https?://\S+", "", speech)
            speech = re.sub(r"\s+", " ", speech).strip()
            if not speech:
                return False
            if not tts_started:
                tts_started = True
                if on_speaking_start:
                    on_speaking_start()
                await websocket.send_json({"type": "status", "status": "speaking"})

            async def send_audio(response, provider: str) -> bool:
                nonlocal first_audio_ms, session_uses_premade_voice
                if response.status_code != 200:
                    body = await response.aread()
                    if provider == "elevenlabs" and response.status_code == 402:
                        session_uses_premade_voice = True
                    log.warning("[VoiceWS] TTS returned %s: %s", response.status_code, body.decode("utf-8", errors="ignore")[:250])
                    return False
                async for audio_chunk in response.aiter_bytes(chunk_size=4096):
                    if asyncio.current_task().cancelled():
                        return False
                    if first_audio_ms is None:
                        first_audio_ms = round((time.perf_counter() - turn_started_at) * 1000)
                    await websocket.send_json({
                        "type": "audio",
                        "data": base64.b64encode(audio_chunk).decode("ascii"),
                        "mime": "audio/mp3",
                        "streaming": True,
                    })
                return True

            sent = False
            elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
            configured_voice_id = os.getenv("ELEVENLABS_VOICE_ID", ELEVENLABS_VOICE_ID).strip()
            voice_id = "cjVigY5qzO86Huf0OWal" if session_uses_premade_voice else configured_voice_id
            model_id = os.getenv("ELEVENLABS_MODEL_ID", ELEVENLABS_MODEL_ID).strip()
            client = tts_http_client
            if client is not None:
                if elevenlabs_key:
                    try:
                        async with client.stream(
                            "POST", f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
                            headers={"xi-api-key": elevenlabs_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                            json={"text": speech, "model_id": model_id, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
                        ) as response:
                            sent = await send_audio(response, "elevenlabs")
                    except Exception as exc:
                        log.warning("[VoiceWS] ElevenLabs segment failed: %s", exc)
                    # Preserve the existing premade-voice fallback for accounts
                    # that cannot access their configured library voice.
                    if not sent and voice_id != "cjVigY5qzO86Huf0OWal":
                        session_uses_premade_voice = True
                        try:
                            async with client.stream(
                                "POST", "https://api.elevenlabs.io/v1/text-to-speech/cjVigY5qzO86Huf0OWal/stream",
                                headers={"xi-api-key": elevenlabs_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                                json={"text": speech, "model_id": model_id, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
                            ) as response:
                                sent = await send_audio(response, "elevenlabs")
                        except Exception as exc:
                            log.warning("[VoiceWS] ElevenLabs premade fallback failed: %s", exc)
                if not sent and deepgram_key:
                    try:
                        async with client.stream(
                            "POST", "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mp3",
                            headers={"Authorization": f"Token {deepgram_key}", "Content-Type": "application/json"},
                            json={"text": speech},
                        ) as response:
                            sent = await send_audio(response, "deepgram")
                    except Exception as exc:
                        log.warning("[VoiceWS] Deepgram segment failed: %s", exc)
            if sent:
                any_audio_sent = True
                await websocket.send_json({"type": "audio_segment_end"})
            return sent

        tts_queue: asyncio.Queue = asyncio.Queue()

        async def synthesize_queued_segments():
            while True:
                segment = await tts_queue.get()
                try:
                    if segment is None:
                        return
                    await synthesize_segment(segment)
                finally:
                    tts_queue.task_done()

        tts_worker_task = asyncio.create_task(synthesize_queued_segments())

        if groq_key:
            try:
                messages = [{"role": "system", "content": voice_prompt}]
                messages.extend(recent_history)
                messages.append({
                    "role": "user",
                    "content": f"College records:\n{context_text}\n\nCurrent student question: {query_text}",
                })
                async with AsyncGroq(api_key=groq_key) as client:
                    completion = await client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=messages,
                        max_tokens=125,
                        temperature=0.3,
                        stream=True,
                    )
                    async for chunk in completion:
                        if asyncio.current_task().cancelled():
                            log.info("[VoiceWS] Groq generation cancelled by barge-in.")
                            return
                        delta = chunk.choices[0].delta.content or ""
                        if not delta:
                            continue
                        if first_token_ms is None:
                            first_token_ms = round((time.perf_counter() - turn_started_at) * 1000)
                        collected.append(delta)
                        speech_buffer += delta
                        await websocket.send_json({"type": "answer_chunk", "delta": delta})
                        # Queue complete sentences so TTS and Groq generation overlap.
                        while True:
                            boundary = re.search(r"[.!?](?:[\"'”’)]*)\s+", speech_buffer)
                            if not boundary:
                                break
                            sentence = speech_buffer[:boundary.end()].strip()
                            speech_buffer = speech_buffer[boundary.end():]
                            await tts_queue.put(sentence)
                bot_answer = "".join(collected).strip()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("[VoiceWS] Groq streaming error: %s", exc)
                bot_answer = "".join(collected).strip() or (
                    "I can help with programmes, fees, admissions, scholarships, hostel, placements, and campus services. "
                    "Which one would you like to check?"
                )
        else:
            bot_answer = "The student-support service is not connected right now. Please try again in a moment."

        if not bot_answer:
            bot_answer = "I can help with programmes, fees, admissions, scholarships, hostel, placements, and campus services. What would you like to know?"
        # Flush the final clause, then let the TTS worker finish all queued segments.
        if speech_buffer.strip():
            await tts_queue.put(speech_buffer.strip())
        if not collected and not any_audio_sent:
            await tts_queue.put(bot_answer)
        await tts_queue.join()
        await tts_queue.put(None)
        await tts_worker_task
        tts_worker_task = None
        if tts_http_client is not None:
            await tts_http_client.aclose()
            tts_http_client = None

        if conversation_history is not None:
            conversation_history.extend([
                {"role": "user", "content": query_text},
                {"role": "assistant", "content": bot_answer},
            ])
            del conversation_history[:-8]

        await websocket.send_json({"type": "answer", "text": bot_answer})
        await websocket.send_json({"type": "audio_end"})
        await websocket.send_json({"type": "status", "status": "listening"})
        log.info(
            "[VoiceWS][timing] retrieval_ms=%s first_token_ms=%s first_audio_ms=%s total_ms=%d",
            retrieval_ms, first_token_ms, first_audio_ms,
            round((time.perf_counter() - turn_started_at) * 1000),
        )
    except asyncio.CancelledError:
        if tts_worker_task and not tts_worker_task.done():
            tts_worker_task.cancel()
            await asyncio.gather(tts_worker_task, return_exceptions=True)
        if tts_http_client is not None:
            await tts_http_client.aclose()
        log.info("[VoiceWS] Voice response task interrupted.")
        raise
    except Exception as exc:
        if tts_worker_task and not tts_worker_task.done():
            tts_worker_task.cancel()
            await asyncio.gather(tts_worker_task, return_exceptions=True)
        if tts_http_client is not None:
            await tts_http_client.aclose()
        log.error("[VoiceWS] Voice response error: %s", exc)
        try:
            await websocket.send_json({"type": "status", "status": "listening"})
        except Exception:
            pass


@app.websocket("/api/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional voice chat with browser.
    Pipeline:
      Browser Mic -> WebSocket -> Deepgram STT (Nova-2) -> Groq LLM -> ElevenLabs TTS (Flash v2.5) -> WebSocket -> Browser Speaker
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
    conversation_history: list[dict] = []

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

    last_handled_transcript = ""
    last_trigger_time = 0.0

    async def trigger_answer(text: str):
        nonlocal current_response_task, is_bot_speaking, latest_transcript, last_handled_transcript, last_trigger_time
        clean = text.strip()
        if not clean:
            return
        now = asyncio.get_event_loop().time()
        # Debounce: if the same transcript was triggered within 2.5s, ignore duplicate
        if clean.lower() == last_handled_transcript.lower() and (now - last_trigger_time < 2.5):
            log.info(f"[VoiceWS] Ignoring duplicate speech trigger: {clean!r}")
            return

        last_handled_transcript = clean
        last_trigger_time = now
        latest_transcript = ""

        log.info(f"[VoiceWS] Triggering answer for speech: {clean!r}")
        if current_response_task and not current_response_task.done():
            current_response_task.cancel()
        is_bot_speaking = False
        await websocket.send_json({"type": "turn_started"})
        current_response_task = asyncio.create_task(
            run_voice_response(
                clean,
                websocket,
                deepgram_key,
                groq_key,
                on_speaking_start=on_speaking_start,
                conversation_history=conversation_history,
            )
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
                                "turn_ended": speech_final,
                            })

                            # Barge-in: only cancel if bot is actively speaking out loud
                            if is_bot_speaking:
                                log.info(f"[VoiceWS] User spoke while bot speaking: {transcript!r}. Interrupting.")
                                await cancel_active_response()

                            if speech_final:
                                log.info(f"[VoiceWS] Finalized speech utterance: {transcript!r}")
                                latest_transcript = ""
                                await trigger_answer(transcript)

                elif msg_type == "UtteranceEnd":
                    log.info(f"[VoiceWS] Deepgram UtteranceEnd event. latest_transcript: {latest_transcript!r}")
                    if latest_transcript:
                        pending = latest_transcript
                        latest_transcript = ""
                        await trigger_answer(pending)

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
            if recv.get("type") == "websocket.disconnect":
                break

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

                if msg_type in ("client_speaking_start",):
                    is_bot_speaking = True
                    continue

                if msg_type in ("client_speaking_end",):
                    is_bot_speaking = False
                    continue

                if msg_type in ("interrupt", "user_speech_start"):
                    # Instant client-side VAD barge-in
                    log.info(f"[VoiceWS] Immediate barge-in received ({msg_type}).")
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
                            await websocket.send_json({"type": "transcript", "text": user_text, "is_final": True, "turn_ended": True})
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
        # Avoid Windows reload subprocess startup failures when launched directly.
        reload=False,
        log_level="info",
    )
