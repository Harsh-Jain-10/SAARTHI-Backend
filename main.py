"""
Saarthi AI — Terminal-Based RAG Backend Prototype
=================================================
B.Tech CSE (Data Science & AI) Minor Project
Institution: SRM University, Delhi-NCR, Sonipat

Pipeline Stages:
1. Document Loading: Reads .txt files from backend/documents/
2. Text Cleaning & Chunking: Normalizes text and creates overlapping passages
3. Embedding Generation: Uses sentence-transformers (all-MiniLM-L6-v2)
4. Vector Storage: Stores embeddings in an in-memory NumPy matrix
5. Query Retrieval: Encodes user question and finds top-3 chunks via cosine similarity
6. Answer Generation: Generates grounded response using Groq LLM (if configured)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Import pipeline helper modules
from document_processor import process_documents, load_documents
from retriever import (
    load_embedding_model,
    build_vector_index,
    retrieve_relevant_chunks,
    DEFAULT_EMBEDDING_MODEL,
)

load_dotenv()

# Configurable LLM model on Groq
GROQ_MODEL = "llama-3.3-70b-versatile"
DOCUMENTS_DIRECTORY = Path(__file__).resolve().parent / "documents"


def generate_llm_answer(query: str, retrieved_chunks: list[dict]) -> str | None:
    """
    Stage 6: Grounded Answer Generation using Groq API
    Answers strictly from retrieved evidence.
    Returns None if Groq API key is not configured in .env.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key.strip())

        # Compile retrieved chunks into a factual context block
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "You are Saarthi AI, a campus assistant for SRM University, Delhi-NCR, Sonipat. "
            "Answer the user's question based strictly and exclusively on the provided context passages below. "
            "If the context does not contain the answer, reply: "
            "'The available documents do not provide enough information to answer this question.' "
            "Do not fabricate facts, extrapolate beyond what is stated, or hallucinate outside facts. "
            "Cite the source document filename where appropriate."
        )

        user_prompt = (
            f"Context passages:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Grounded Answer:"
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature ensures deterministic, factual responses
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[Groq API Error]: {e}"


def run_terminal_pipeline():
    """
    Interactive terminal workflow for Saarthi AI RAG backend.
    """
    print("\n" + "=" * 68)
    print("        SAARTHI AI -- RAG PIPELINE PROTOTYPE (TERMINAL)")
    print("      SRM University, Delhi-NCR, Sonipat -- Review 2 Prototype")
    print("=" * 68)

    # Stage 3 Pre-check: Load Embedding Model
    print(f"\n[Stage 3] Loading embedding model: '{DEFAULT_EMBEDDING_MODEL}'...")
    print("         (Weights will download on first run if not cached locally)")
    try:
        embedding_model = load_embedding_model(DEFAULT_EMBEDDING_MODEL)
        print("         Model loaded successfully.")
    except Exception as e:
        print(f"         [Error] Failed to load embedding model: {e}")
        sys.exit(1)

    # Stage 1 & 2: Document Ingestion, Cleaning & Chunking
    print(f"\n[Stage 1 & 2] Reading documents from: {DOCUMENTS_DIRECTORY}")
    raw_docs = load_documents(DOCUMENTS_DIRECTORY)

    if not raw_docs:
        print("\n" + "!" * 68)
        print(" [NOTICE] Knowledge base is currently EMPTY.")
        print(f" Place official .txt files inside: {DOCUMENTS_DIRECTORY}")
        print(" No fabricated college facts or synthetic documents are inserted.")
        print("!" * 68)

    chunks = process_documents(DOCUMENTS_DIRECTORY)
    print(f"\n[Status] Loaded {len(raw_docs)} document(s) | Generated {len(chunks)} text chunk(s).")

    # Stage 4: In-Memory Vector Indexing
    print("\n[Stage 4] Generating embeddings and building in-memory vector index...")
    if chunks:
        index_embeddings, indexed_chunks = build_vector_index(chunks, embedding_model)
        print(
            f"         Indexed {index_embeddings.shape[0]} vector(s) of dimension {index_embeddings.shape[1]}."
        )
    else:
        index_embeddings, indexed_chunks = build_vector_index([], embedding_model)
        print("         In-memory index initialized (0 vectors).")

    # Groq API check
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    if groq_configured:
        print(f"\n[Stage 6] Groq LLM integration active (Model: {GROQ_MODEL}).")
    else:
        print("\n[Stage 6] Groq LLM API key not found in .env.")
        print("         Pipeline will retrieve evidence; LLM generation remains pending.")

    print("\n" + "-" * 68)
    print(" Ready for questions! Type 'exit' or 'quit' to terminate.")
    print("-" * 68)

    # Interactive Query Loop
    while True:
        try:
            print()
            query = input("Ask a question: ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print("\nExiting Saarthi AI prototype. Goodbye!\n")
                break

            print(f"\n--- Searching for: \"{query}\" ---")

            if len(chunks) == 0:
                print("\n[Retrieval Status] No documents in knowledge base.")
                print(f"Please add official .txt files to: {DOCUMENTS_DIRECTORY}")
                continue

            # Stage 5: Similarity Retrieval
            results = retrieve_relevant_chunks(
                query=query,
                index_embeddings=index_embeddings,
                chunks=indexed_chunks,
                model=embedding_model,
                top_k=3,
            )

            if not results:
                print("\n[Stage 5: Retrieval] No relevant chunks found.")
                continue

            print(f"\n[Stage 5: Retrieval] Found {len(results)} relevant chunk(s):")
            for i, res in enumerate(results, 1):
                print(f"\n  Passage #{i} | Source: {res['source']} ({res['id']})")
                print(f"  Cosine Similarity Score: {res['score']:.4f}")
                print(f"  Excerpt: {res['text']}")

            # Stage 6: Grounded Answer Generation
            print("\n[Stage 6: Answer Generation]")
            if groq_configured:
                print(f"Synthesizing answer using Groq ({GROQ_MODEL})...")
                answer = generate_llm_answer(query, results)
                print(f"\nAnswer:\n{answer}")
            else:
                print("LLM integration pending (GROQ_API_KEY not configured in .env).")
                print("Retrieved evidence above represents the prompt context that will be supplied to Groq.")

        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user. Exiting...\n")
            break
        except Exception as e:
            print(f"\n[Error during query processing]: {e}")

if __name__ == "__main__":
    run_terminal_pipeline()
