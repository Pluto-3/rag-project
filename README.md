# rag-project

A local Retrieval-Augmented Generation (RAG) system built from scratch. Ingests PDF documents, stores them as semantic vectors, and answers questions grounded in your documents — entirely offline, no API costs.

## Stack

- **Python 3.10+**
- **Ollama** — local LLM inference (`llama3.2:3b`)
- **PyMuPDF (fitz)** — PDF text extraction
- **sentence-transformers** — local embeddings (`all-MiniLM-L6-v2`)
- **ChromaDB** — persistent vector database

## Project Structure

```
rag-project/
├── ingest.py       # PDF → clean text
├── chunker.py      # text → overlapping word chunks
├── embedder.py     # chunks → vectors → ChromaDB (multi-doc, deduplication)
├── retriever.py    # semantic search against vector DB
├── rag.py          # full pipeline: query rewriting + retrieval + answer + sources
└── chroma_db/      # persisted vector database (generated, not committed)
```

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows CMD

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pymupdf sentence-transformers chromadb ollama

# Pull the LLM
ollama pull llama3.2:3b
```

## Usage

**1. Start Ollama (required every session)**

```bash
# Windows CMD — set these before ollama serve
set CUDA_VISIBLE_DEVICES=-1
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=1
ollama serve
```

**2. Embed your documents**

Drop one or more PDF files into the project root, then:

```bash
python embedder.py
```

Already-embedded documents are skipped automatically. To re-embed, delete `chroma_db/` and re-run.

**3. Ask questions**

```bash
python rag.py
```

Type any question. The system will rewrite it into an optimized search query, retrieve the most relevant chunks, and generate a grounded answer with source attribution.

## How It Works

```
PDF → text → chunks → embeddings → ChromaDB
                                       ↓
                               user question
                                       ↓
                          LLM rewrites query
                                       ↓
                           retrieve top-k chunks
                                       ↓
                              LLM generates answer
                                       ↓
                          answer + source filenames
```

## Hardware Notes (Windows, NVIDIA laptop GPU)

This project was developed on an NVIDIA RTX 4050 Laptop (6GB VRAM). The model runs on CPU due to a cuBLAS compatibility issue between CUDA 13.0 and `llama3.2:3b`'s batched attention operations.

Setting `CUDA_VISIBLE_DEVICES=-1` hides the GPU from CUDA entirely, forcing CPU mode. Context is capped at `num_ctx: 1024/2048` to keep RAM usage within limits (the model requires ~3.3GB weights + KV cache).

If running on a machine without these constraints, remove the `CUDA_VISIBLE_DEVICES` flag and increase `num_ctx` accordingly.

## Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `-1` | Disables GPU, forces CPU inference |
| `OLLAMA_NUM_PARALLEL` | `1` | Prevents concurrent request memory spikes |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Aggressively frees memory between calls |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Reduces KV cache memory usage |

## Chunking Parameters

| Parameter | Value | Notes |
|---|---|---|
| `chunk_size` | 400 words | Balances context richness vs retrieval precision |
| `overlap` | 75 words | Prevents ideas from being split at chunk boundaries |
| `k` (top results) | 3 | Number of chunks fed to the LLM as context |
