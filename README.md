# Saarthi AI — Backend & Real-Time Voice Pipeline

**Project Title:** Saarthi AI — Real-Time Voice & RAG Campus Assistant  
**Institution:** SRM University, Delhi-NCR, Sonipat  
**Department:** Computer Science & Engineering (Data Science & AI)  

---

## 1. Overview & Architecture

This repository hosts two core backend subsystems for Saarthi AI:
1. **Real-Time Voice Pipeline (`src/voice_agent.py`):** An ultra-low latency, full-duplex conversational voice bot powered by **Pipecat AI**, **Deepgram Nova-2 STT**, **Groq Llama 3.1 8B**, **Deepgram Aura TTS**, and local **Silero VAD** with sub-250ms barge-in interruption.
2. **Terminal RAG Prototype (`main.py`):** An honest, grounded knowledge retrieval pipeline combining `sentence-transformers` vector search with college documents.

### Real-Time Voice Pipeline Flow:
```
[User Mic] ──> [Silero VAD (200ms onset)] ──> [Deepgram Nova-2 STT]
                                                     │
                                                     ▼
[User Speaker] <── [Deepgram Aura TTS] <── [Groq Llama 3.1 8B]
```

### Voice Pipeline Highlights:
* **Framework:** Pipecat AI (`pipecat-ai`)
* **Speech-to-Text (STT):** Deepgram Nova-2 (`DeepgramSTTService`, model: `nova-2`) via real-time WebSocket streaming.
* **LLM Engine:** Groq Cloud (`GroqLLMService`, model: `llama-3.1-8b-instant`) with system prompt enforced to strictly 1–2 conversational sentences.
* **Text-to-Speech (TTS):** Deepgram Aura (`DeepgramTTSService`, model: `aura-asteria-en`) streaming natural spoken responses.
* **Voice Activity Detection (VAD):** Local Silero VAD (`SileroVADAnalyzer`) running via ONNX for sub-250ms speech segmentation and barge-in detection.
* **Audio Transport:** Local machine audio drivers via PyAudio (`LocalAudioTransport`).
* **Structured Lifecycle Events:**
  * `[LISTENING]` — Waiting for user voice input
  * `[USER TRANSCRIPT]` — Real-time speech transcription
  * `[BOT THINKING]` — User turn ended, formulating LLM response
  * `[BOT SPEAKING]` — Audio synthesized and streaming to speakers
  * `[BARGE-IN DETECTED]` — User interrupted active bot speech; 

---

## 2. Directory Structure

```text
Saarthi Backend/
├── src/
│   ├── __init__.py
│   └── voice_agent.py        # Pipecat real-time voice pipeline with sub-250ms barge-in
├── verify_voice.py           # Pre-flight check for microphone/speaker & API connectivity
├── main.py                   # Terminal RAG knowledge retrieval prototype
├── document_processor.py      # Text extraction, normalization, and chunking
├── retriever.py               # Vector embeddings (all-MiniLM-L6-v2) & cosine search
├── documents/                 # Directory for campus .txt documents
├── requirements.txt           # Pinned backend dependencies
├── .env.example               # Template environment configuration
├── .env                       # Local secrets (excluded from Git)
├── .gitignore                 # Protects secrets, venvs, and cache
└── README.md                  # Documentation
```

---

## 3. Setup & Installation

### Step 1: Install Dependencies
Activate your virtual environment and install packages:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

*(Or run directly via python executable without activating)*:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env` (or edit existing `.env`):
```text
DEEPGRAM_API_KEY=your_deepgram_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```
* **Deepgram API Key:** Free tier at [https://console.deepgram.com/](https://console.deepgram.com/)
* **Groq API Key:** Free tier at [https://console.groq.com/keys](https://console.groq.com/keys)

---

## 4. Pre-Flight Verification (`verify_voice.py`)

Run the pre-flight verification script to test your system microphone, speaker output, and API connectivity:

```powershell
python verify_voice.py
# or
.\.venv\Scripts\python.exe verify_voice.py
```

The script performs:
1. Python version & package check
2. Microphone stream test (records 0.5s audio buffer)
3. Speaker driver test (plays a gentle 440 Hz test tone)
4. `.env` credentials presence check
5. Live Groq API latency & model verification
6. Live Deepgram API authentication check

---

## 5. Running the Voice Agent (`src/voice_agent.py`)

Launch the voice bot directly from terminal:

```powershell
python src/voice_agent.py
# or
.\.venv\Scripts\python.exe src/voice_agent.py
```

* Speak into your microphone; the assistant will respond through your speakers in 1–2 sentences.
* **Barge-in:** Interrupt the assistant at any time while it is speaking; playback will immediately halt and Saarthi will listen to your new speech.
* **Stop:** Press `Ctrl + C` in the terminal to cleanly shut down.

---

## 6. Running the RAG Pipeline Prototype

To run the offline document Q&A prototype:
```powershell
python main.py
```
Place college `.txt` files in `documents/` to index them automatically on startup.
