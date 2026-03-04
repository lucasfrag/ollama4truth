# 🧠 Ollama4Truth — Fact-Checking with RAG + Ollama

An automated fact-checking system that verifies claims against a curated corpus of **19,502 Brazilian fact-checking articles** using Retrieval-Augmented Generation (RAG) and local LLMs via [Ollama](https://ollama.com). Developed as part of a **CNPq-funded research project** on combating disinformation in Brazil.

## ✨ Features

- **4-Step Pipeline**: Question generation → Evidence retrieval → Per-question answering → Classification
- **3 Retrieval Modes**: RAG (local corpus), Web (Google Search), Hybrid (RAG → web fallback)
- **3 Retrieval Methods**: BM25 (keyword), Semantic (embeddings), Hybrid (combined) — switchable at runtime
- **2 Verdict Strategies**: Ollama LLM classification, Label majority voting
- **Multi-Run Consistency**: Classification runs N times (default: 3); confidence = % agreement (not LLM self-reported)
- **Per-Question Answering**: Each investigative question is answered individually by the LLM using its retrieved evidence before final classification
- **Runtime Model Selector**: Switch between any downloaded Ollama model via webapp dropdown
- **Claim History**: Every analysis is saved; browse, expand Q&A details, and clear past results
- **Dark Theme UI**: Dark navy background with white text, all labels translated to Portuguese
- **Semantic Search**: Sentence-transformer embeddings with chunk & pool encoding (500-char chunks, max-sim)
- **Embedding Caching**: First run encodes articles (~10 min), subsequent restarts are instant
- **19,502 articles** from 6 Brazilian fact-checking sources (G1, Lupa, Aos Fatos, Estadão, Boatos.org, UOL Confere)
- **Web interface** with real-time SSE streaming
- **FastAPI** backend serving both API and webapp

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (conda env `cnpq` recommended)
- [Ollama](https://ollama.com) installed with at least one model (e.g., `qwen3:14b`)
- JSONL datasets in `coleta_datasets/data/raw/`

### Setup

```bash
# 1. Install dependencies
conda activate cnpq
pip install -r requirements.txt

# 2. Start Ollama (if not running as system service)
CUDA_VISIBLE_DEVICES=0 OLLAMA_MODELS=/mnt/E-SSD/model_cache ollama serve

# 3. Pull a model (if not already downloaded)
ollama pull qwen3:14b   # or any model of your choice

# 4. Start the API server
CUDA_VISIBLE_DEVICES=0 uvicorn api:app --host 0.0.0.0 --port 8000
# ⚠️ First run encodes ~156k chunks (~10 min). Subsequent starts are instant.
```

### Remote Access (SSH)

```bash
# On your local machine, create an SSH tunnel:
ssh -L 8000:localhost:8000 <server>

# Then open: http://localhost:8000
```

## 📁 Project Structure

```
ollama4truth/
├── api.py                       # FastAPI server + static file serving
├── main.py                      # Pipeline orchestrator (init_rag, run_pipeline)
├── .env                         # Configuration (GPU, model, paths)
├── requirements.txt             # Python dependencies
├── pipeline/
│   ├── __init__.py
│   ├── generate_questions.py    # Ollama-powered question generation
│   ├── retrieve_evidence.py     # Multi-mode evidence retrieval dispatcher
│   ├── rag_retrieval.py         # BM25 + semantic RAG index (RAGIndex class)
│   ├── data_loader.py           # Unified JSONL corpus loader
│   ├── answer_questions.py      # Per-question answering using retrieved evidence
│   └── classification.py        # Verdict strategies (ollama_verdict, label_vote)
├── webapp/
│   ├── index.html               # Main UI with mode/retrieval/strategy/model dropdowns
│   ├── script.js                # SSE client, Q&A rendering, history panel
│   └── style.css                # Dark theme styling with source/label badges
└── data/
    ├── results.json             # Latest pipeline output (auto-generated, gitignored)
    ├── history.jsonl            # Claim analysis history (auto-generated, gitignored)
    └── embeddings_cache/        # Cached .npy embeddings (auto-generated, gitignored)
```

## ⚙️ Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU to use |
| `OLLAMA_MODEL` | `qwen3:14b` | Default Ollama model (can be overridden via webapp dropdown) |
| `OLLAMA_MODELS` | `/mnt/E-SSD/model_cache` | Ollama models directory (E-SSD) |
| `DATA_DIR` | `/mnt/.../data/raw` | Path to JSONL dataset directories |
| `SEMANTIC_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | Sentence-transformer model for semantic search |
| `ENCODING_STRATEGY` | `chunk_pool` | Encoding strategy: `chunk_pool`, `title_label`, `truncate` |
| `EMBEDDINGS_CACHE_DIR` | `./data/embeddings_cache` | Directory for cached embeddings (.npy) |
| `MODEL_CACHE_DIR` | `/mnt/E-SSD/mussi/model_cache` | HuggingFace/Torch model download directory |
| `CONSISTENCY_RUNS` | `3` | Number of LLM classification runs for confidence calculation |
| `GOOGLE_API_KEY` | — | For web/hybrid retrieval modes |
| `GOOGLE_CSE_ID` | — | For web/hybrid retrieval modes |

## 📖 Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for comprehensive technical documentation.

## 🔗 Data Sources

| Source | Articles | File |
|---|---|---|
| G1 Fato ou Fake | 1,907 | `g1_cleaned.jsonl` |
| Lupa | 4,141 | `lupa_cleaned.jsonl` |
| Aos Fatos | 3,537 | `aosfatos_cleaned.jsonl` |
| Estadão Verifica | 1,695 | `estadao_cleaned.jsonl` |
| Boatos.org | 6,556 | `boatos_2020_2025_cleaned.jsonl` |
| UOL Confere | 1,666 | `confere_cleaned.jsonl` |
| **Total** | **19,502** | |

## 📜 License

Part of a CNPq-funded research project on combating disinformation in Brazil.
