# Saarthi AI — Backend RAG Pipeline Prototype

**Project Title:** Saarthi AI — A RAG-Based AI Voice Assistant  
**Institution:** SRM University, Delhi-NCR, Sonipat  
**Department:** Computer Science & Engineering (Data Science & AI)  
**Milestone:** Review 2 — Terminal RAG Pipeline Prototype  

---

## 1. Objective & Architecture

This repository contains the terminal-based Python RAG backend for **Saarthi AI**. It demonstrates every stage of Retrieval-Augmented Generation using clean, readable, modular Python code without heavy framework overhead.

### Key Principles:
1. **Data Independence:** The knowledge base is initially empty (`documents/` contains no hardcoded or fabricated data). It is ready to accept official `.txt` documents collected from SRM University.
2. **Real Embeddings:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to produce real 384-dimensional dense vectors.
3. **In-Memory Vector Search:** Stores vectors in NumPy arrays and retrieves top-matching chunks via mathematical cosine similarity.
4. **Honest Generation:** Uses the Groq API (`llama-3.3-70b-versatile`) only if a valid `GROQ_API_KEY` is present in `.env`. Otherwise, it displays retrieved evidence without hallucinating.

---

## 2. Workspace Layout

All files are organized directly inside `Saarthi Backend/`:

```text
Saarthi Backend/
├── main.py                 # Terminal application entrypoint & interactive query loop
├── document_processor.py   # Document loading, text cleaning & overlapping chunking
├── retriever.py            # Sentence-transformers embedding & cosine similarity retrieval
├── documents/              # Directory for official college .txt documents (initially empty)
├── requirements.txt        # Minimal Python dependencies
├── .gitignore              # Ignores .venv/ and .env
├── .venv/                  # Virtual environment with installed dependencies
└── README.md               # System documentation
```

---

## 3. Pipeline Stages (In Order)

```
[Documents] ──> [Extraction] ──> [Cleaning] ──> [Chunking] ──> [Embeddings]
     │                                                              │
     ▼                                                              ▼
[LLM Answer] <── [Similarity Retrieval] <── [Query Embedding] <── [In-Memory Index]
```

1. **Documents (`documents/`):** Real plain-text files stored locally.
2. **Extraction (`document_processor.py`):** Reads `.txt` files with UTF-8 encoding and tracks filenames as source metadata.
3. **Cleaning (`document_processor.py`):** Normalizes line breaks (`\r\n` to `\n`), trims whitespace, and collapses excessive blank lines.
4. **Chunking (`document_processor.py`):** Slices text into overlapping passages (default: 500 characters, 50 characters overlap) with sequential IDs (`filename-chk-001`).
   * *Why chunking?* Embedding models and LLMs perform best on focused, semantically cohesive passages.
   * *Why overlap?* Prevents ideas at chunk boundaries from being cut off.
5. **Embeddings (`retriever.py`):** Generates 384-dimensional dense vector embeddings using `all-MiniLM-L6-v2`.
6. **In-Memory Index (`retriever.py`):** Stores chunk embeddings in a NumPy matrix alongside chunk text and source filenames.
7. **Query Embedding (`retriever.py`):** Encodes the user's terminal question into the same 384-dimensional vector space.
8. **Similarity Retrieval (`retriever.py`):** Computes cosine similarity between query and chunk vectors, returning the top 3 passages.
9. **LLM Answer (`main.py`):** Formats retrieved passages into a strict factual prompt for Groq (`llama-3.3-70b-versatile`). If no key is configured, displays the retrieved evidence.

---

## 4. How to Run (Windows PowerShell)

In `Saarthi Backend`:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

> **PowerShell Script Policy Alternative:**  
> If PowerShell blocks `Activate.ps1`, run directly with the virtual environment's executable:
> ```powershell
> .\.venv\Scripts\python.exe main.py
> ```

---

## 5. Adding Real College Documents

1. Place your `.txt` files (e.g. `curriculum.txt`, `hostel_rules.txt`, `academic_calendar.txt`) inside `documents/`.
2. Run `python main.py`.
3. The program automatically reads, cleans, chunks, and indexes all documents on startup.

---

## 6. Configuring Groq LLM (Optional)

Create a `.env` file in the root directory:
```text
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(The `.env` file is excluded from Git via `.gitignore`).*
