import os
import hashlib
import uuid
from dataclasses import dataclass
from typing import List, Tuple, Optional

from dotenv import load_dotenv

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Qdrant (self-hosted vector DB)
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Load environment variables from .env file
load_dotenv()


# -----------------------------
# Config
# -----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY first: export GROQ_API_KEY='gsk_...' (or set in .env)")

# Groq OpenAI-compatible API client
groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# Local embedding model (stable)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Chat model (Groq free tier models)
CHAT_MODEL = "llama-3.1-8b-instant"

# Chunking params (tune if you want)
CHUNK_CHARS = 1200
OVERLAP_CHARS = 200

# Qdrant Cloud config
QDRANT_URL = os.getenv("QDRANT_URL")  # e.g., "https://xxxxx.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Cloud API key (required)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "condo_docs")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY for Qdrant Cloud (or set in .env file)")


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    condo_id: int
    source: str
    page: int
    text: str


# -----------------------------
# PDF -> text by page
# -----------------------------
def read_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        # normalize a bit (optional)
        txt = " ".join(txt.split())
        pages.append(txt)
    return pages


# -----------------------------
# Page text -> overlapping chunks
# -----------------------------
def chunk_page(text: str, condo_id: int, source: str, page_num_1based: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not text.strip():
        return chunks

    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        chunk_text = text[start:end].strip()
        if chunk_text:
            # deterministic ID
            h = hashlib.sha1(f"{condo_id}|{source}|{page_num_1based}|{start}|{chunk_text[:50]}".encode("utf-8")).hexdigest()[:12]
            chunks.append(Chunk(
                chunk_id=f"condo{condo_id}-{source}-p{page_num_1based}-{h}",
                condo_id=condo_id,
                source=source,
                page=page_num_1based,
                text=chunk_text
            ))
        if end == n:
            break
        start = max(0, end - OVERLAP_CHARS)

    return chunks


def build_chunks_from_pdf(pdf_path: str, condo_id: int, source_name: str) -> List[Chunk]:
    pages = read_pdf_pages(pdf_path)
    all_chunks: List[Chunk] = []
    for i, txt in enumerate(pages, start=1):
        all_chunks.extend(chunk_page(txt, condo_id, source_name, i))
    return all_chunks


# -----------------------------
# Embedding
# -----------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    # normalize_embeddings=True enables cosine similarity via dot product
    emb = embedder.encode(texts, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


# -----------------------------
# Qdrant helpers (store + retrieve)
# -----------------------------
def qdrant_client() -> QdrantClient:
    """Connect to Qdrant Cloud with URL and API key."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(vector_size: int) -> None:
    client = qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
    
    # Qdrant Cloud requires payload indexes for filtering
    # Create index for condo_id (integer filter)
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="condo_id",
            field_schema="integer",
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            print(f"Note: condo_id index - {e}")
    
    # Create index for source (keyword filter)
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="source",
            field_schema="keyword",
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            print(f"Note: source index - {e}")


def qdrant_has_source(condo_id: int, source_name: str) -> bool:
    """Return True if Qdrant already has at least one point for this condo_id + source."""
    client = qdrant_client()
    try:
        res = client.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="condo_id",
                        match=qmodels.MatchValue(value=condo_id),
                    ),
                    qmodels.FieldCondition(
                        key="source",
                        match=qmodels.MatchValue(value=source_name),
                    )
                ]
            ),
            exact=True,
        )
        return (res.count or 0) > 0
    except Exception:
        # collection may not exist yet
        return False


def upsert_chunks_to_qdrant(chunks: List[Chunk], vectors: np.ndarray) -> int:
    """Upsert chunks + vectors into Qdrant (idempotent because chunk_id is deterministic)."""
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors length mismatch")

    ensure_collection(vector_size=int(vectors.shape[1]))

    client = qdrant_client()

    # Qdrant IDs can be integers or UUID strings; convert chunk_id to UUID.
    points: List[qmodels.PointStruct] = []
    for ch, vec in zip(chunks, vectors):
        payload = {
            "chunk_id": ch.chunk_id,
            "condo_id": ch.condo_id,
            "source": ch.source,
            "page": ch.page,
            "text": ch.text,
        }
        # Convert chunk_id to UUID (deterministic)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ch.chunk_id))
        points.append(qmodels.PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

    # Batch upserts to avoid huge single requests
    BATCH = 256
    total = 0
    for i in range(0, len(points), BATCH):
        batch = points[i:i+BATCH]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        total += len(batch)
    return total


def delete_condo_from_qdrant(condo_id: int) -> int:
    """Delete all points for a specific condo_id from Qdrant."""
    client = qdrant_client()
    
    # Delete points where condo_id matches
    delete_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="condo_id",
                match=qmodels.MatchValue(value=condo_id),
            )
        ]
    )
    
    try:
        result = client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=qmodels.FilterSelector(filter=delete_filter)
        )
        return result.operation_id or 0
    except Exception as e:
        print(f"Error deleting condo {condo_id}: {e}")
        return 0


def retrieve_qdrant(query: str, condo_id: int, k: int = 6, source_filter: Optional[str] = None) -> List[Tuple[Chunk, float]]:
    """Vector search in Qdrant filtered by condo_id, returning Chunk objects + similarity score."""
    client = qdrant_client()
    qvec = embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # Always filter by condo_id
    must_conditions = [
        qmodels.FieldCondition(
            key="condo_id",
            match=qmodels.MatchValue(value=condo_id),
        )
    ]
    
    if source_filter:
        must_conditions.append(
            qmodels.FieldCondition(
                key="source",
                match=qmodels.MatchValue(value=source_filter),
            )
        )

    qfilter = qmodels.Filter(must=must_conditions)

    hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=qvec,
        limit=k,
        query_filter=qfilter,
        with_payload=True,
    )

    out: List[Tuple[Chunk, float]] = []
    for hit in hits.points:
        payload = hit.payload or {}
        ch = Chunk(
            chunk_id=str(payload.get("chunk_id", hit.id)),
            condo_id=int(payload.get("condo_id", condo_id)),
            source=str(payload.get("source", "")),
            page=int(payload.get("page", 0) or 0),
            text=str(payload.get("text", "")),
        )
        out.append((ch, float(hit.score)))
    return out


# -----------------------------
# Generation (Groq OpenAI-compatible API)
# -----------------------------
def chat_generate(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    # Build context with citations
    context_lines = []
    for ch, score in retrieved:
        context_lines.append(f"[{ch.source} p.{ch.page} | {ch.chunk_id} | score={score:.3f}]\n{ch.text}")
    context = "\n\n".join(context_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful assistant answering questions about a legal document.\n"
                "Use ONLY the provided context. If the context is insufficient, say so.\n"
                "When you make a factual claim, cite the relevant bracketed source tag (e.g., [CondoOne p.12 | ...])."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]

    try:
        resp = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    # Define condos with their PDF paths
    condos = [
        {
            "condo_id": 1,
            "name": "CondoOne",
            "pdf_path": r"/home/igorsigors26/Downloads/OceanaOne7562PublicOffering with Amendments OCR.pdf"
        },
        {
            "condo_id": 2,
            "name": "CondoTwo",
            "pdf_path": r"/home/igorsigors26/Downloads/Condo 2 Bylaws with Amendments OCR.pdf"
        }
    ]

    # 1) Build & ingest each condo to Qdrant (only if not already present)
    for condo in condos:
        condo_id = condo["condo_id"]
        source_name = condo["name"]
        pdf_path = condo["pdf_path"]
        
        if not os.path.exists(pdf_path):
            print(f"Skipping {source_name}: PDF not found at {pdf_path}")
            continue
        
        if qdrant_has_source(condo_id, source_name):
            print(f"Qdrant already contains chunks for condo_id={condo_id}, source='{source_name}'.")
        else:
            print(f"\nProcessing {source_name} (condo_id={condo_id})...")
            print("Building chunks from PDF...")
            chunks = build_chunks_from_pdf(pdf_path, condo_id, source_name)
            if not chunks:
                print(f"No text extracted from PDF for {source_name}. Skipping.")
                continue
            print(f"Embedding {len(chunks)} chunks...")
            chunk_emb = embed_texts([c.text for c in chunks])
            print(f"Upserting into Qdrant collection '{QDRANT_COLLECTION}' at {QDRANT_URL} ...")
            n = upsert_chunks_to_qdrant(chunks, chunk_emb)
            print(f"Upserted {n} chunks.")

    # 2) Select a condo and ask questions
    print("\n" + "="*50)
    print("Available condos:")
    for condo in condos:
        print(f"  {condo['condo_id']}: {condo['name']}")
    print("  delete <id>: Delete all data for a condo")
    
    while True:
        condo_choice = input("\nSelect condo (1 or 2, or 'delete <id>', or 'quit'): ").strip()
        if condo_choice.lower() in {"quit", "exit"}:
            break
        
        if condo_choice.startswith("delete "):
            try:
                delete_id = int(condo_choice.split()[1])
                confirm = input(f"Are you sure you want to delete ALL data for condo_id={delete_id}? (yes/no): ").strip().lower()
                if confirm == "yes":
                    print(f"Deleting all data for condo_id={delete_id}...")
                    operation_id = delete_condo_from_qdrant(delete_id)
                    if operation_id:
                        print(f"Delete operation completed (operation_id: {operation_id})")
                    else:
                        print("Delete operation failed or no data found")
                else:
                    print("Delete cancelled.")
                continue
            except (ValueError, IndexError):
                print("Invalid delete command. Use 'delete <id>' where <id> is a number.")
                continue
        
        try:
            selected_condo_id = int(condo_choice)
            selected_condo = next((c for c in condos if c["condo_id"] == selected_condo_id), None)
            if not selected_condo:
                print(f"Invalid condo ID: {selected_condo_id}")
                continue
        except ValueError:
            print("Please enter a valid condo ID or 'delete <id>'.")
            continue
        
        print(f"\nSelected: {selected_condo['name']} (condo_id={selected_condo_id})")
        
        # Ask questions for this condo
        while True:
            question = input("\nAsk a question (or 'back'): ").strip()
            if not question or question.lower() in {"back", "quit", "exit"}:
                if question.lower() == "back":
                    break
                else:
                    return

            retrieved = retrieve_qdrant(question, condo_id=selected_condo_id, k=6, source_filter=selected_condo['name'])
            if not retrieved:
                print("No matches found in Qdrant for this query.")
                continue

            answer = chat_generate(question, retrieved)
            print("\n--- Answer ---\n")
            print(answer)


if __name__ == "__main__":
    main()
