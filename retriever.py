"""
Retriever Module for Saarthi AI
===============================
Handles:
Stage 3: Loading the embedding model (all-MiniLM-L6-v2) and generating embeddings.
Stage 4: Executing similarity searches directly inside Supabase using pgvector,
         with local in-memory NumPy fallback.
"""
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def retrieve_relevant_chunks_supabase(
    query: str,
    db_url: str,
    model: SentenceTransformer,
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
    chunks: list[dict], model: SentenceTransformer
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
    model: SentenceTransformer,
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
