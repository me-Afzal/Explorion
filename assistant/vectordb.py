"""
vectordb.py
-----------
Pinecone vector store for persistent chat memory.

Public API:
  initialize()                                  → None  (call once at startup)
  create_session()                              → str (UUID session ID)
  store_interaction(session_id, query, content, is_report) → None
  retrieve_context(session_id, query, top_k)   → str (prompt-ready text)

Environment variables required in .env:
  PINECONE_API_KEY=<your_key>
"""

import os
import re
import time
import uuid

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

INDEX_NAME       = "chat-vector-db"
EMBED_DIM        = 384
REPORT_CHUNK_MAX = 400

_pc:  Pinecone              | None = None
_emb: HuggingFaceEmbeddings | None = None


def initialize() -> None:
    """
    Eagerly connect to Pinecone and load the embedding model.
    Safe to call multiple times — skips if already initialized.
    """
    global _pc, _emb

    if _pc is not None and _emb is not None:
        return

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not set in .env")

    print("[VectorDB] Connecting to Pinecone…")
    _pc = Pinecone(api_key=api_key)

    try:
        existing = {idx.name for idx in _pc.list_indexes()}
    except Exception:
        existing = set()

    if INDEX_NAME not in existing:
        print(f"[VectorDB] Creating index '{INDEX_NAME}'…")
        _pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        for _ in range(30):
            try:
                if _pc.describe_index(INDEX_NAME).status.ready:
                    break
            except Exception:
                pass
            time.sleep(2)
        print(f"[VectorDB] Index '{INDEX_NAME}' created and ready.")
    else:
        print(f"[VectorDB] Index '{INDEX_NAME}' found.")

    print("[VectorDB] Loading embedding model BAAI/bge-small-en-v1.5…")
    _emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[VectorDB] Embedding model ready.")


def _get_pc() -> Pinecone:
    if _pc is None:
        raise RuntimeError("vectordb.initialize() has not been called.")
    return _pc


def _get_emb() -> HuggingFaceEmbeddings:
    if _emb is None:
        raise RuntimeError("vectordb.initialize() has not been called.")
    return _emb


def _chunk_report(text: str, max_chars: int = REPORT_CHUNK_MAX) -> list[str]:
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:max_chars]]


def create_session() -> str:
    return str(uuid.uuid4())


def store_interaction(
    session_id: str,
    user_query: str,
    assistant_content: str,
    is_report: bool = False,
) -> None:
    try:
        pc    = _get_pc()
        emb   = _get_emb()
        index = pc.Index(INDEX_NAME)

        if is_report:
            texts = [
                f"User asked: {user_query}\n\nReport excerpt: {chunk}"
                for chunk in _chunk_report(assistant_content)
            ]
        else:
            texts = [f"User: {user_query}\nAssistant: {assistant_content}"]

        ts = int(time.time())
        vectors = [
            {
                "id":     f"{session_id}_{ts}_{i}",
                "values": emb.embed_query(text),
                "metadata": {
                    "session_id": session_id,
                    "text":       text,
                    "type":       "report" if is_report else "chat",
                },
            }
            for i, text in enumerate(texts)
        ]
        index.upsert(vectors=vectors)
    except Exception as exc:
        print(f"[VectorDB] store_interaction error (non-fatal): {exc}")


def retrieve_context(session_id: str, query: str, top_k: int = 5) -> str:
    try:
        pc    = _get_pc()
        emb   = _get_emb()
        index = pc.Index(INDEX_NAME)

        q_vec  = emb.embed_query(query)
        result = index.query(
            vector=q_vec,
            top_k=top_k,
            filter={"session_id": {"$eq": session_id}},
            include_metadata=True,
        )

        texts = [
            m["metadata"]["text"]
            for m in result.get("matches", [])
            if m.get("metadata", {}).get("text")
        ]
        return "\n---\n".join(texts) if texts else ""
    except Exception as exc:
        print(f"[VectorDB] retrieve_context error (non-fatal): {exc}")
        return ""
