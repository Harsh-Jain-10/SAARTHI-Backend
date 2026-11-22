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
import re
import hashlib

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 80

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
    # Keep FAQ blocks and bullet facts intact instead of cutting through a fee line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    units: list[str] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if (
            line.lower().startswith("question:")
            and line_index + 1 < len(lines)
            and lines[line_index + 1].lower().startswith("answer:")
        ):
            units.append(f"{line}\n{lines[line_index + 1]}")
            line_index += 2
        else:
            units.append(line)
            line_index += 1
    safe_units: list[str] = []
    for unit in units:
        if len(unit) <= chunk_size:
            safe_units.append(unit)
            continue
        remainder = unit
        while len(remainder) > chunk_size:
            boundary = remainder.rfind(" ", 0, chunk_size + 1)
            if boundary <= 0:
                boundary = chunk_size
            safe_units.append(remainder[:boundary].strip())
            next_start = max(0, boundary - chunk_overlap)
            remainder = remainder[next_start:].strip()
        if remainder:
            safe_units.append(remainder)

    chunks: list[dict] = []
    current: list[str] = []
    current_length = 0
    for unit in safe_units:
        is_faq = unit.lower().startswith("question:")
        # Give every FAQ its own retrieval unit. Combining several questions into
        # one chunk makes lexical/vector search return neighboring answers.
        if is_faq and current:
            chunk_number = len(chunks) + 1
            body = "\n".join(current)
            chunks.append({
                "id": f"{source_filename}-chk-{chunk_number:03d}",
                "chunk_number": chunk_number,
                "source": source_filename,
                "text": body,
            })
            current = []
            current_length = 0
        added_length = len(unit) + (1 if current else 0)
        if current and current_length + added_length > chunk_size:
            chunk_number = len(chunks) + 1
            body = "\n".join(current)
            chunks.append({
                "id": f"{source_filename}-chk-{chunk_number:03d}",
                "chunk_number": chunk_number,
                "source": source_filename,
                "text": body,
            })
            current = []
            current_length = 0
        current.append(unit)
        current_length += len(unit) + (1 if len(current) > 1 else 0)
        if is_faq:
            chunk_number = len(chunks) + 1
            chunks.append({
                "id": f"{source_filename}-chk-{chunk_number:03d}",
                "chunk_number": chunk_number,
                "source": source_filename,
                "text": "\n".join(current),
            })
            current = []
            current_length = 0

    if current:
        chunk_number = len(chunks) + 1
        chunks.append({
            "id": f"{source_filename}-chk-{chunk_number:03d}",
            "chunk_number": chunk_number,
            "source": source_filename,
            "text": "\n".join(current),
        })
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
    seen_lines: set[str] = set()
    for doc in raw_docs:
        unique_lines = []
        for line in doc["text"].splitlines():
            normalized = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
            if not normalized or normalized in seen_lines:
                continue
            seen_lines.add(normalized)
            unique_lines.append(line)
        cleaned = clean_text("\n".join(unique_lines))
        chunks = chunk_text(cleaned, doc["source"], chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    # Stable IDs let repeated app starts and Supabase re-ingestion use the same records.
    for chunk in all_chunks:
        digest = hashlib.sha1(chunk["text"].encode("utf-8")).hexdigest()[:10]
        chunk["content_hash"] = digest
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
