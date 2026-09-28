---
title: Delphion
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
short_description: Ask questions across your PDFs, with cited answers.
---

# Delphion

Retrieval-Augmented Generation over PDFs. Upload documents, ask questions, get
answers that cite the passage and page they came from.

Runs on a free CPU Space: embeddings are computed locally, the index is
in-memory, and generation goes to any OpenAI-compatible endpoint.

## How it works

```
PDF ──PyMuPDF──> page Documents ──chunker──> chunks
                                               │
                              fastembed (bge-small, CPU, in-process)
                                               │
                                      in-memory Chroma index
                                               │
question ──embed──> similarity search ──top-k passages──> LLM ──> cited answer
```

| Stage | Module | Notes |
|---|---|---|
| Extract | `src/extract_from_pdf.py` | PyMuPDF, one Document per page; attaches `source`/`page`/`title` for citations |
| Chunk | `src/chunking/` | three interchangeable strategies behind `ChunkingFactory` |
| Embed | `src/models.py` | `bge-small-en-v1.5` via fastembed (ONNX, no torch) |
| Index | `src/vectorstore.py` | Chroma, in-memory, session-scoped |
| Answer | `src/rag.py` | numbered context passages, inline `[n]` citations |

## Chunking strategies

- **recursive** — splits on paragraph → line → sentence → word boundaries. General default.
- **semantic** — embeds sentences and cuts where meaning shifts most (95th-percentile distance). Costs one embedding pass; free here because embeddings are local.
- **markdown** — splits on `#` headings, preserving section hierarchy. Only useful for documents that have them.

## Configuration

Copy `.env.example` to `.env` (locally) or set these as Space secrets:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | Groq | any OpenAI-compatible endpoint |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | check your provider's current model list |
| `LLM_API_KEY` | — | not needed for local Ollama/vLLM |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | changing it invalidates any existing index |
| `DEFAULT_K` | `4` | passages retrieved per question |

Retrieval works without an LLM key — you will get the passages, just not a
generated answer.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env    # add your key
python app.py
```

## Repository history

Earlier iterations are kept under `legacy/` for reference:

- `legacy/v1-gradio-monolith/` — the original single-file Gradio app using the OpenAI API
- `legacy/v2-modal-mcp/` — the FastAPI microservice split (embedder/retriever/generator) deployed on Modal

Each folder has a `README.md` (file-by-file reference) and a `BREAKDOWN.html`
(annotated walkthrough with diagrams). The v2 code also lives, unmodified and
still runnable, in the companion repository
[Delphion](https://github.com/AurindumBanerjee/Delphion), which is preserved as
the record of the Modal deployment.

Both are archived, not maintained. The Modal version stored its Chroma index on
container-local disk while Modal autoscaled containers, so ingestion and
retrieval could land in different containers and retrieval would silently return
nothing. Collapsing to a single process with an in-memory index removed that
class of bug by construction.

## License

MIT
