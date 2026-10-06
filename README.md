# Ask your PDF

A small retrieval-augmented generation (RAG) app: upload a PDF, ask questions, get answers grounded in the document with page citations.

No RAG framework. The whole pipeline is ~100 lines in [`rag.py`](rag.py), using the OpenAI API for embeddings and answers and plain numpy for vector search.

## How it works

```
               INDEXING (once per PDF)
PDF ──► extract text per page ──► split into overlapping chunks ──► embed ──► vectors (cached in .cache/)

               QUERYING (per question)
question ──► embed ──► cosine similarity vs all chunks ──► top-k chunks
                                                              │
                                    prompt = instructions + chunks + question
                                                              │
                                                              ▼
                                                    LLM answer with (p. N) citations
```

1. **Extract.** `pypdf` pulls the text from each page.
2. **Chunk.** Each page is split into ~200-word windows with 40 words of overlap, so a sentence cut at a boundary still appears whole in one chunk. Chunks never cross pages, which keeps citations exact.
3. **Embed.** Each chunk becomes a vector with `text-embedding-3-small`. The vectors are normalized to unit length, so cosine similarity is just a dot product.
4. **Retrieve.** The question is embedded the same way and the 4 most similar chunks are selected.
5. **Generate.** The chunks are passed to the chat model with instructions to answer only from them and cite pages.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # core + CLI
pip install -r requirements-ui.txt     # optional: Streamlit web UI
cp .env.example .env                   # then add your OpenAI API key
```

## Usage

**Terminal:**

```bash
python cli.py path/to/document.pdf
```

Example with NVIDIA's *Optimizing Parallel Reduction in CUDA* slides:

```
Indexed 38 chunks. Ask a question (empty line to quit).

> What optimizations are applied to make the reduction faster?

The optimizations applied to make the reduction faster include:

- Algorithmic optimizations such as changes to addressing and algorithm cascading, resulting in an 11.84x speedup (p. 37).
- Code optimizations including loop unrolling, achieving a 2.54x speedup (p. 37).
- Kernel decomposition to avoid global synchronization by decomposing computation into multiple kernel invocations (p. 5).

Sources: p. 1 (0.62), p. 5 (0.54), p. 2 (0.54), p. 37 (0.53)

> What is the capital of France?

I couldn't find the answer in the document provided.

Sources: p. 12 (0.03), p. 20 (0.03), p. 14 (0.03), p. 30 (0.01)
```

The second question shows the grounding at work: no chunk is relevant (similarity ≈ 0), so the model declines instead of answering from general knowledge.

**Web UI:**

```bash
streamlit run app.py
```

The UI lets you upload a PDF, chat with it, and expand each answer to see which chunks were retrieved and their similarity scores.

## Configuration

Set these in `.env`:

| Variable | Default |
| --- | --- |
| `CHAT_MODEL` | `gpt-4.1-mini` |
| `EMBED_MODEL` | `text-embedding-3-small` |

Embeddings are cached per file in `.cache/`, so re-opening the same PDF costs nothing.

## Limitations

- Scanned PDFs (images without a text layer) aren't supported because there's no OCR.
- Each question is answered independently, with no memory of earlier questions.
- The index lives in memory, which is fine for one document up to a few hundred pages. A larger corpus would need a vector database.
