"""
Retriever Module for Saarthi AI
===============================
Handles:
Stage 3: Loading the embedding model (all-MiniLM-L6-v2) and generating chunk vectors.
Stage 4: In-memory vector storage using NumPy arrays.
Stage 5: Query embedding, cosine similarity calculation, and top-K chunk retrieval.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Lightweight, fast, high-quality embedding model (~80MB download on first run)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Stage 3 (Part A): Model Loading
    Initializes the sentence-transformers model.
    Note: On the first run, the model weights are automatically downloaded
    from Hugging Face and cached locally. Subsequent runs load from local cache.
    """
    return SentenceTransformer(model_name)


def build_vector_index(
    chunks: list[dict], model: SentenceTransformer
) -> tuple[np.ndarray, list[dict]]:
    """
    Stage 3 & 4: Embedding Generation & In-Memory Storage

    Why Embeddings?
    - Embeddings translate natural language into 384-dimensional numerical vectors.
    - Text passages with similar conceptual meaning land close together in vector space,
      enabling mathematical semantic search rather than exact word matching.

    Storage Limitation:
    - Embeddings are kept in memory as a NumPy matrix.
    - The index is recreated whenever the application starts and is not persisted to disk.
    """
    if not chunks:
        return np.empty((0, 384), dtype=np.float32), []

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    return embeddings, chunks


def retrieve_relevant_chunks(
    query: str,
    index_embeddings: np.ndarray,
    chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict]:
    """
    Stage 5: Query Embedding & Cosine Similarity Retrieval
    1. Converts user question into a vector using the same embedding model.
    2. Calculates cosine similarity against all indexed chunk vectors.
    3. Ranks chunks and returns the top K most relevant matches with similarity scores.
    """
    if not query.strip() or len(chunks) == 0 or index_embeddings.shape[0] == 0:
        return []

    # Encode query vector
    query_vector = model.encode(query, convert_to_numpy=True, show_progress_bar=False)

    # Compute cosine similarity: cos(theta) = (A · B) / (||A|| * ||B||)
    chunk_norms = np.linalg.norm(index_embeddings, axis=1)
    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0 or np.any(chunk_norms == 0):
        scores = np.zeros(len(chunks))
    else:
        scores = np.dot(index_embeddings, query_vector) / (chunk_norms * query_norm)

    # Sort indices in descending order
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
