"""
Retriever Module for Saarthi AI
===============================
Handles:
Stage 3: Loading the embedding model (all-MiniLM-L6-v2) and generating embeddings.
Stage 4: Executing similarity searches directly inside Supabase using pgvector,
         with local in-memory NumPy fallback.
"""
import os
import gc
# Restrict PyTorch thread memory footprint before a model runtime is imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import psycopg2
import re
import math
from collections import Counter
from typing import Any

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """Load embeddings lazily so a blocked optional ML DLL cannot stop the API."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    model = SentenceTransformer(model_name, device="cpu")
    gc.collect()
    return model

def retrieve_relevant_chunks_supabase(
    query: str,
    db_url: str,
    model: Any,
    top_k: int = 5,
) -> list[dict]:
    """
    Queries Supabase pgvector table using the match_documents SQL function.
    """
    if not query.strip() or not db_url:
        return []

    # Generate 384-dimensional query embedding
    query_vector = model.encode(query, convert_to_numpy=True, show_progress_bar=False)
    query_vector_str = "[" + ",".join(map(str, query_vector.tolist())) + "]"

    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT id, content, source, similarity FROM match_documents(%s::extensions.vector, %s);",
            (query_vector_str, top_k)
        )
    except Exception:
        conn.rollback()
        cursor.execute(
            "SELECT id, content, source, similarity FROM match_documents(%s::vector, %s);",
            (query_vector_str, top_k)
        )
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    results = []
    for row in rows:
        results.append(
            {
                "id": str(row[0]),
                "text": row[1],
                "source": row[2],
                "score": float(row[3]),
            }
        )

    return results


def build_vector_index(
    chunks: list[dict], model: Any
) -> tuple[np.ndarray, list[dict]]:
    """Local fallback vector indexer."""
    if not chunks:
        return np.empty((0, 384), dtype=np.float32), []
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings, chunks


def retrieve_relevant_chunks(
    query: str,
    index_embeddings: np.ndarray,
    chunks: list[dict],
    model: Any,
    top_k: int = 3,
) -> list[dict]:
    """Local fallback cosine similarity retrieval."""
    if not query.strip() or len(chunks) == 0 or index_embeddings.shape[0] == 0:
        return []
    query_vector = model.encode(query, convert_to_numpy=True, show_progress_bar=False)
    chunk_norms = np.linalg.norm(index_embeddings, axis=1)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0 or np.any(chunk_norms == 0):
        scores = np.zeros(len(chunks))
    else:
        scores = np.dot(index_embeddings, query_vector) / (chunk_norms * query_norm)
    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[: min(top_k, len(chunks))]
    results = []
    for idx in top_indices:
        results.append(
            {
                "id": chunks[idx]["id"],
                "source": chunks[idx]["source"],
                "text": chunks[idx]["text"],
                "score": float(scores[idx]),
            }
        )
    return results


def retrieve_relevant_chunks_lexical(
    query: str, chunks: list[dict], top_k: int = 3
) -> list[dict]:
    """BM25 document ranking used when the embedding runtime is unavailable."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
        "does", "for", "from", "how", "i", "in", "is", "it", "me", "of", "on",
        "or", "please", "the", "to", "what", "when", "where", "which", "who",
        "why", "with", "would", "you", "your",
    }
    def tokenize(text: str) -> list[str]:
        raw = re.findall(r"[a-z0-9]+", text.lower())
        terms: list[str] = []
        index = 0
        while index < len(raw):
            token = raw[index]
            # Treat common written and spoken forms as the same programme token.
            if token == "b" and index + 1 < len(raw) and raw[index + 1] == "tech":
                token = "btech"
                index += 1
            if token not in stop_words and len(token) > 1:
                terms.append(token)
            index += 1
        return terms

    query_terms = tokenize(query)
    expansions = {
        "ai": {"artificial", "intelligence"},
        "ml": {"machine", "learning"},
        "fee": {"fees", "tuition", "charge", "charges", "cost"},
        "fees": {"fee", "tuition", "charge", "charges", "cost"},
        "charge": {"charges", "fee", "fees", "tuition", "cost"},
        "charges": {"charge", "fee", "fees", "tuition", "cost"},
        "cost": {"fee", "fees", "tuition", "charge", "charges"},
        "annual": {"yearly", "year"},
        "yearly": {"annual", "year"},
        "cse": {"computer", "science", "engineering"},
    }
    query_terms.extend(
        expanded
        for term in query_terms[:]
        for expanded in expansions.get(term, set())
    )
    terms = set(query_terms)
    if not terms or not chunks:
        return []

    tokenized = [tokenize(chunk.get("text", "")) for chunk in chunks]
    document_frequency = Counter(term for tokens in tokenized for term in set(tokens))
    average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
    ranked = []
    k1, b = 1.5, 0.75
    for chunk_index, chunk in enumerate(chunks):
        tokens = tokenized[chunk_index]
        frequencies = Counter(tokens)
        score = 0.0
        for term in terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(tokens) / max(average_length, 1))
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        if score > 0:
            ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "id": chunk.get("id", ""),
            "source": chunk.get("source", ""),
            "text": chunk.get("text", ""),
            "score": float(score),
        }
        for score, chunk in ranked[:max(0, top_k)]
    ]
