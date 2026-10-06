"""Minimal RAG over a single PDF: chunk -> embed -> retrieve -> answer with page citations."""

import hashlib
import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
CACHE_DIR = Path(".cache")

SYSTEM_PROMPT = """You answer questions about a document using only the excerpts provided.
Cite the page for every claim, like (p. 3).
If the excerpts don't contain the answer, say you couldn't find it in the document."""


@dataclass
class Chunk:
    page: int
    text: str


@lru_cache
def client() -> OpenAI:
    return OpenAI()  # reads OPENAI_API_KEY from the environment


def load_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_pages(pages: list[tuple[int, str]], size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split each page into overlapping word windows. Chunks never cross pages, so citations stay exact."""
    chunks = []
    step = size - overlap
    for page, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            chunks.append(Chunk(page, " ".join(words[start : start + size])))
            if start + size >= len(words):
                break
    return chunks


def embed(texts: list[str], batch_size: int = 100) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), batch_size):
        resp = client().embeddings.create(model=EMBED_MODEL, input=texts[i : i + batch_size])
        vectors.extend(d.embedding for d in resp.data)
    v = np.array(vectors, dtype=np.float32)
    # Unit-length vectors make cosine similarity a plain dot product.
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class PdfIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def build(cls, pdf_bytes: bytes) -> "PdfIndex":
        """Index a PDF, reusing cached embeddings if this exact file was indexed before."""
        key = hashlib.sha256(pdf_bytes + EMBED_MODEL.encode()).hexdigest()[:16]
        cache = CACHE_DIR / f"{key}.npz"
        if cache.exists():
            data = np.load(cache)
            chunks = [Chunk(int(p), str(t)) for p, t in zip(data["pages"], data["texts"])]
            return cls(chunks, data["vectors"])

        chunks = chunk_pages(load_pages(pdf_bytes))
        if not chunks:
            raise ValueError("No extractable text found. Is this a scanned PDF?")
        vectors = embed([c.text for c in chunks])

        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(
            cache,
            vectors=vectors,
            pages=np.array([c.page for c in chunks]),
            texts=np.array([c.text for c in chunks]),
        )
        return cls(chunks, vectors)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ embed([query])[0]
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def answer(self, question: str, k: int = 4) -> tuple[str, list[tuple[Chunk, float]]]:
        hits = self.search(question, k)
        context = "\n\n".join(f"[p. {c.page}]\n{c.text}" for c, _ in hits)
        resp = client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"},
            ],
        )
        return resp.choices[0].message.content, hits
