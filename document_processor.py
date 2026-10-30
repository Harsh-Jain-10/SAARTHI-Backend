"""
Document Processor for Saarthi AI
=================================
Handles:
1. Loading plain-text (.txt) files from the documents directory.
2. Cleaning extracted text (normalizing spaces and line breaks).
3. Partitioning text into overlapping chunks with source metadata.
4. Ingesting generated chunks and embeddings into Supabase pgvector.
"""
import os
from pathlib import Path
import psycopg2

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

def load_documents(documents_dir: str | Path) -> list[dict]:
    docs_path = Path(documents_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        return []
    doc_files = sorted(list(docs_path.glob("*.txt")) + list(docs_path.glob("*.md")))
    if not doc_files:
        return []
    documents = []
    for file_path in doc_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        if content.strip():
            documents.append({"source": file_path.name, "text": content})
    return documents

def clean_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        clean_line = " ".join(line.split())
        lines.append(clean_line)
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

def chunk_text(
    text: str,
    source_filename: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"Overlap ({chunk_overlap}) must be strictly smaller than chunk size ({chunk_size})."
        )
    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    chunk_number = 1
    while start < len(text):
        end = start + chunk_size
        chunk_body = text[start:end].strip()
        if chunk_body:
            chunks.append(
                {
                    "id": f"{source_filename}-chk-{chunk_number:03d}",
                    "chunk_number": chunk_number,
                    "source": source_filename,
                    "text": chunk_body,
                }
            )
            chunk_number += 1
        start += step
    return chunks

def process_documents(
    documents_dir: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    raw_docs = load_documents(documents_dir)
    if not raw_docs:
        return []
    all_chunks = []
    for doc in raw_docs:
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned, doc["source"], chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks

def ingest_chunks_to_supabase(chunks: list[dict], embeddings, db_url: str):
    """
    Ingests document chunks and their embeddings into Supabase pgvector.
    Resets existing table data to avoid duplicate entries.
    """
    if not chunks or len(chunks) == 0:
        return

    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    # Clear previous documents for a clean re-ingestion
    cursor.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")

    for chunk, emb in zip(chunks, embeddings):
        embedding_str = "[" + ",".join(map(str, emb.tolist())) + "]"
        try:
            cursor.execute(
                """
                INSERT INTO document_chunks (content, source, chunk_index, embedding)
                VALUES (%s, %s, %s, %s::extensions.vector)
                """,
                (chunk["text"], chunk["source"], chunk["chunk_number"], embedding_str)
            )
        except Exception:
            conn.rollback()
            cursor.execute(
                """
                INSERT INTO document_chunks (content, source, chunk_index, embedding)
                VALUES (%s, %s, %s, %s::vector)
                """,
                (chunk["text"], chunk["source"], chunk["chunk_number"], embedding_str)
            )

    conn.commit()
    cursor.close()
    conn.close()
