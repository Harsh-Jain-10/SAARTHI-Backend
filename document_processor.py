"""
Document Processor for Saarthi AI
=================================
Handles:
1. Loading plain-text (.txt) files from the documents directory.
2. Cleaning extracted text (normalizing spaces and line breaks).
3. Partitioning text into overlapping chunks with source metadata.
"""

from pathlib import Path

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def load_documents(documents_dir: str | Path) -> list[dict]:
    """
    Stage 1: Document Loading & Text Extraction
    Reads all .txt files from the target directory and preserves the filename as source.
    """
    docs_path = Path(documents_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        return []

    txt_files = sorted(docs_path.glob("*.txt"))
    if not txt_files:
        return []

    documents = []
    for file_path in txt_files:
        # Read file with UTF-8, fallback to Latin-1 if special characters exist
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
    """
    Stage 2 (Part A): Text Cleaning
    Normalizes excessive whitespace and standardized line breaks while
    preserving meaningful sentence punctuation and paragraph breaks.
    """
    if not raw_text:
        return ""

    # Standardize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Trim and collapse extra spaces per line
    lines = []
    for line in text.split("\n"):
        clean_line = " ".join(line.split())
        lines.append(clean_line)

    text = "\n".join(lines)

    # Collapse 3 or more consecutive newlines down to double newline (paragraphs)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


def chunk_text(
    text: str,
    source_filename: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Stage 2 (Part B): Text Chunking
    Splits cleaned text into overlapping segments with provenance metadata.

    Why Chunking?
    - Embedding models (like all-MiniLM-L6-v2) and LLMs have finite input limits.
    - Focused chunks yield substantially higher semantic retrieval accuracy.

    Why Overlap?
    - Sliding overlap ensures ideas split at arbitrary character limits remain
      intelligible and contiguous across adjacent passages.
    """
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
    """
    Loads, cleans, and chunks all .txt files in the given directory.
    Returns a unified list of chunk dictionaries with metadata.
    """
    raw_docs = load_documents(documents_dir)
    if not raw_docs:
        return []

    all_chunks = []
    for doc in raw_docs:
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned, doc["source"], chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    return all_chunks
