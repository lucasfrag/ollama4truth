# Ollama4Truth — Technical Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Layer](#3-data-layer)
4. [Pipeline Components](#4-pipeline-components)
5. [API Reference](#5-api-reference)
6. [Webapp Interface](#6-webapp-interface)
7. [Configuration](#7-configuration)
8. [Deployment Guide](#8-deployment-guide)
9. [Extending the System](#9-extending-the-system)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

Ollama4Truth is an automated fact-checking system developed as part of a **CNPq-funded research project** on combating disinformation in Brazil. It verifies user-submitted claims against a curated corpus of 19,502 fact-checked articles from 6 major Brazilian fact-checking organizations.

### Core Capabilities

- **Question Generation**: Uses a local LLM (via Ollama) to generate investigative questions from a claim
- **Evidence Retrieval**: Searches a local corpus using BM25, semantic (nomic-embed-text-v1.5 + chunk_pool), or hybrid retrieval — switchable at runtime via webapp dropdown
- **Claim Classification**: Two strategies — LLM-based verdict (negation-aware) and corpus label majority voting
- **Runtime Model Selection**: Switch between any downloaded Ollama model via webapp dropdown
- **Claim History**: Every analysis is saved to `data/history.jsonl`; browse, expand details, and clear past results
- **Real-time Streaming**: Results stream to the web interface via Server-Sent Events (SSE)

### Technology Stack

| Component | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| LLM | Ollama (local inference), default model: `qwen3:14b` |
| Retrieval | rank-bm25 (lexical), sentence-transformers (default: paraphrase-multilingual-mpnet-base-v2) |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Data | JSONL files from 6 fact-checking sources |
| GPU | NVIDIA RTX A6000 (CUDA) |

---

## 2. Architecture

### Pipeline Flow

```
User Claim
     │
     ▼
┌─────────────────────┐
│  Question Generation │  ← Ollama LLM (qwen3:14b)
│  generate_questions() │
└─────────┬───────────┘
          │ 3-5 investigative questions
          ▼
┌─────────────────────┐
│  Evidence Retrieval  │  ← Mode: rag | web | hybrid
│  retrieve_evidence() │
└─────────┬───────────┘
          │ Grouped evidence per question
          ▼
┌─────────────────────┐
│   Classification     │  ← Strategy: ollama_verdict | label_vote
│   classify_claim()   │
└─────────┬───────────┘
          │ Verdict + confidence + justification
          ▼
    JSON Result → SSE → Webapp
```

### File Structure

```
ollama4truth/
├── api.py                         # FastAPI server, SSE endpoint, static files
├── main.py                        # Pipeline orchestrator, RAG initialization
├── .env                           # All configuration variables
├── requirements.txt               # Python dependencies
│
├── pipeline/
│   ├── __init__.py                # Package marker
│   ├── generate_questions.py      # LLM question generation via Ollama
│   ├── retrieve_evidence.py       # Multi-mode retrieval dispatcher
│   ├── rag_retrieval.py           # RAGIndex class (BM25 + semantic)
│   ├── data_loader.py             # JSONL corpus loader + label normalization
│   └── classification.py          # Verdict strategies (LLM + label voting)
│
├── webapp/
│   ├── index.html                 # UI with mode/retrieval/strategy/model dropdowns + history
│   ├── script.js                  # SSE client, dynamic rendering, history panel
│   └── style.css                  # Responsive styling
│
└── data/
    ├── results.json               # Latest pipeline output (auto-generated, gitignored)
    ├── history.jsonl              # Claim analysis history (auto-generated, gitignored)
    └── embeddings_cache/          # Cached .npy embeddings (auto-generated, gitignored)
```

### Module Dependencies

```
api.py
  └─ main.py
       ├─ pipeline/generate_questions.py  (uses Ollama)
       ├─ pipeline/retrieve_evidence.py
       │    └─ pipeline/rag_retrieval.py
       │         └─ pipeline/data_loader.py
       └─ pipeline/classification.py
            └─ pipeline/data_loader.py (imports FALSE_LABELS, TRUE_LABELS)
```

---

## 3. Data Layer

### 3.1 Article Corpus

The system uses 19,502 normalized fact-checking articles from 6 sources:

| Source | Key | Articles | JSONL File | Subdirectory |
|---|---|---|---|---|
| G1 Fato ou Fake | `g1` | 1,907 | `g1_cleaned.jsonl` | `g1/` |
| Agência Lupa | `lupa` | 4,141 | `lupa_cleaned.jsonl` | `lupa/` |
| Aos Fatos | `aosfatos` | 3,537 | `aosfatos_cleaned.jsonl` | `aosfatos/` |
| Estadão Verifica | `estadao` | 1,695 | `estadao_cleaned.jsonl` | `estadao/` |
| Boatos.org | `boatos` | 6,556 | `boatos_2020_2025_cleaned.jsonl` | `boatos_org/` |
| UOL Confere | `confere` | 1,666 | `confere_cleaned.jsonl` | `confere/` |
| **Total** | | **19,502** | | |

All articles cover the period **January 2020 – December 2025**.

### 3.2 Unified Article Schema

Each article is normalized to the following schema by `data_loader.py`:

```python
{
    "url":             str,   # Original article URL
    "titulo":          str,   # Title
    "subtitulo":       str,   # Subtitle (if available)
    "texto":           str,   # Full article text
    "classificacao":   str,   # Normalized label (lowercase, no accents)
    "source":          str,   # Source key (g1, lupa, aosfatos, etc.)
    "data_publicacao": str,   # Publication date
    "tags":            list,  # Article tags/keywords
    "full_text":       str,   # Concatenation of titulo + subtitulo + texto
}
```

### 3.3 Label Normalization

Labels are normalized by:

1. Lowercasing
2. Stripping accents (e.g., `"É FALSO"` → `"e falso"`)
3. Stripping whitespace

Two label sets are defined for the label voting strategy:

**FALSE_LABELS** (22 variants): `falso`, `fake`, `enganoso`, `distorcido`, `golpe`, `manipulado`, `boato`, `nao e verdade`, `impreciso`, `exagerado`, `insustentavel`, `sem evidencia`, `sem contexto`, `descontextualizado`, `alterado`, `nao ha evidencias`, `nao e bem assim`, `falso/enganoso`

**TRUE_LABELS** (8 variants): `verdadeiro`, `fato`, `verdade`, `correto`, `real`, `comprovado`, `confirmado`, `ainda e verdade`

### 3.4 Data Loader (`pipeline/data_loader.py`)

**Key functions:**

| Function | Description |
|---|---|
| `load_corpus(data_dir)` | Loads all 6 JSONL datasets, returns unified list of dicts |
| `full_text(article)` | Concatenates titulo + subtitulo + texto for retrieval |
| `_normalize_label(label)` | Strips accents, lowercases labels |

**Usage:**

```python
from pipeline.data_loader import load_corpus
corpus = load_corpus("/path/to/data/raw")
# Returns: list of 19,502 article dicts
```

---

## 4. Pipeline Components

### 4.1 Question Generation (`pipeline/generate_questions.py`)

Generates 3-5 neutral, investigative questions from a claim using Ollama.

**Model**: Configured via `OLLAMA_MODEL` in `.env` (default: `qwen3:14b`). Can be overridden at runtime by selecting a different model in the webapp dropdown.

**Prompt strategy**: The prompt frames the task as an academic fact-checking system and uses an "INPUT TEXT" wrapper to avoid triggering safety filters on sensitive topics. Includes a concrete example to guide output format.

**Fallback behavior**: If the LLM refuses or generates no valid questions, the original claim text is used as the search query.

**Output format:**

```python
{
    "claim": "Vacinas causam autismo",
    "questions": [
        "Existem estudos científicos que associam vacinas ao autismo?",
        "O que a OMS diz sobre a relação entre vacinas e autismo?",
        "Qual é a origem da teoria de que vacinas causam autismo?"
    ],
    "timestamp": "2026-02-17T21:07:00.000000"
}
```

### 4.2 Evidence Retrieval (`pipeline/retrieve_evidence.py`)

Dispatches evidence retrieval to one of 3 modes:

#### Mode: `rag` (Local Corpus)

- Searches the local BM25/semantic index for each generated question
- Returns articles with source, label, and snippet
- Evidence is grouped per-question with deduplication across questions

#### Mode: `web` (Google Search)

- Uses Google Custom Search API (requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`)
- Returns web results with title, link, and snippet
- Original Ollama4Truth behavior (preserved)

#### Mode: `hybrid` (RAG → Web Fallback)

- First tries RAG retrieval
- If fewer than `MIN_RAG_RESULTS` (2) results, falls back to Google Search
- Merges RAG and web results, RAG first

**Evidence structure (per question):**

```python
{
    "question": "Existem estudos que associam vacinas ao autismo?",
    "results": [
        {
            "title": "É falso que vacinas causam autismo",
            "link": "https://example.com/article",
            "snippet": "Primeiro parágrafo do artigo...",
            "score": 42.5,
            "source": "boatos",
            "label": "falso"
        }
    ]
}
```

### 4.3 RAG Index (`pipeline/rag_retrieval.py`)

The `RAGIndex` class manages the retrieval indices. **Both BM25 and semantic indices are always built at startup**, allowing runtime switching via the `method` parameter.

**Retrieval methods (switchable per query):**

| Method | Description | Quality |
|---|---|---|
| `bm25` | BM25Okapi keyword matching | Good for exact terms |
| `semantic` | Cosine similarity via sentence-transformer (default: `paraphrase-multilingual-mpnet-base-v2`) with chunk_pool encoding | Better for paraphrases |
| `hybrid` | Weighted combination of BM25 + semantic scores | Best overall |

**Encoding strategy (chunk_pool):**

- Each article's `full_text` is split into 500-character chunks
- All chunks are embedded individually with the sentence-transformer model
- At retrieval: cosine similarity computed per chunk, **max-sim** taken per article
- First run encodes ~156k chunks (~10 min on A6000), cached as `.npy` files for instant restarts

**Key methods:**

| Method | Description |
|---|---|
| `retrieve(query, k=5, method=None)` | Retrieve top-k articles for a single query (method: bm25/semantic/hybrid) |
| `retrieve_multi_query(queries, k_per_query=3, k_total=5, method=None)` | Retrieve across multiple queries with deduplication |

**Result format:**

```python
{
    "title":   str,     # Article title
    "link":    str,     # Article URL
    "snippet": str,     # First 300 chars of article text
    "score":   float,   # Relevance score (method-dependent)
    "source":  str,     # Source key (g1, lupa, etc.)
    "label":   str,     # Normalized article label
}
```

**Semantic model**: `paraphrase-multilingual-mpnet-base-v2` (configurable via `SEMANTIC_MODEL` env var)

**Model cache**: All HuggingFace models are stored in `MODEL_CACHE_DIR` (default: `/mnt/E-SSD/mussi/model_cache`)

### 4.4 Classification (`pipeline/classification.py`)

Two verdict strategies with automatic fallback:

#### Strategy: `ollama_verdict`

- Sends the claim + all evidence to Ollama LLM
- **Negation-aware prompt**: Explicitly instructs the LLM to evaluate the claim as written, paying attention to negations (e.g., "X NÃO causa Y" + evidence says "falso" → SUPPORTED)
- LLM classifies as: `Supported`, `Refuted`, `Not Enough Evidence`, or `Conflicting Evidence/Cherry-picking`
- Returns JSON with classification, justification, and confidence (0-100%)
- Handles JSON parsing errors gracefully

#### Strategy: `label_vote`

- Counts article labels from RAG results
- Compares `FALSE_LABELS` count vs `TRUE_LABELS` count
- Majority wins → verdict + confidence percentage
- **Auto-fallback**: If evidence has no labels (e.g., web-only results), falls back to `ollama_verdict`

**Output format (both strategies):**

```python
{
    "claim": "Vacinas causam autismo",
    "classification": "Refuted",
    "justification": "5 de 5 artigos classificam como falso/enganoso.",
    "confidence": 100.0,
    "strategy": "label_vote",
    "label_breakdown": {"falso": 4, "enganoso": 1},
    "timestamp": "2026-02-17T21:07:30.000000"
}
```

---

## 5. API Reference

### Base URL

```
http://localhost:8000
```

### `GET /analyze-stream`

Analyzes a claim via Server-Sent Events streaming.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `claim` | string | *required* | The claim text to verify |
| `mode` | string | `rag` | Retrieval mode: `rag`, `web`, `hybrid` |
| `strategy` | string | `ollama_verdict` | Verdict strategy: `ollama_verdict`, `label_vote` |
| `retrieval_method` | string | `bm25` | RAG retrieval method: `bm25`, `semantic`, `hybrid` |
| `ollama_model` | string | *(from .env)* | Ollama model to use (empty = `.env` default) |

**Example:**

```
GET /analyze-stream?claim=Vacinas+causam+autismo&mode=rag&strategy=label_vote&retrieval_method=semantic&ollama_model=gemma3:27b
```

**Response**: SSE stream with intermediate log messages, followed by a final JSON result.

### `GET /health`

Health check endpoint.

**Response:**

```json
{"status": "ok", "message": "API está rodando e pronta para receber claims"}
```

### `GET /` (Webapp)

Serves the web interface from the `webapp/` directory.

### `GET /models`

Returns available Ollama models for the dropdown selector.

**Response:**

```json
{
  "default": "llama3.1:8b",
  "models": [
    {"name": "llama3.1:8b", "size": "4.9 GB"},
    {"name": "gemma3:27b", "size": "17 GB"}
  ]
}
```

### `GET /history`

Returns all past claim analyses, most recent first.

**Response:**

```json
{
  "count": 5,
  "entries": [{"claim": "...", "label": "Refuted", "ollama_model": "llama3.1:8b", ...}]
}
```

### `GET /history/clear`

Deletes the `data/history.jsonl` file, clearing all history.

**Response:**

```json
{"status": "ok", "message": "History cleared"}
```

---

## 6. Webapp Interface

The webapp provides a single-page interface for claim verification.

### UI Components

1. **Claim input**: Text field for entering the claim
2. **Mode selector**: Dropdown to choose retrieval mode (RAG/Web/Hybrid)
3. **Retrieval method selector**: Dropdown to choose RAG retrieval method (BM25/Semântico/Híbrido)
4. **Strategy selector**: Dropdown to choose verdict strategy (Ollama LLM/Label Vote)
5. **Model selector**: Dropdown to choose Ollama model (populated from `GET /models` on page load)
6. **Sidebar (Perguntas)**: Lists generated questions with evidence counts; clickable to filter evidence
7. **Main panel**: Displays verdict badge, confidence, justification, and evidence cards
8. **Evidence cards**: Show article title (linked), source badge, label badge, and snippet
9. **History panel**: Toggleable panel showing all past analyses, expandable to see full details (questions, evidence, rationale, model used)

### Evidence Display Features

- **Source badges**: Uppercase colored badges (BOATOS, LUPA, G1, etc.)
- **Label badges**: Color-coded — red for false/enganoso, green for verdadeiro/fato, gray for others
- **Verdict badges**: Green (Supported), Red (Refuted), Yellow (Not Enough Evidence)

---

## 7. Configuration

All configuration is in `.env`:

```bash
# GPU selection
CUDA_VISIBLE_DEVICES=0

# Ollama model for question generation and classification (default, overridable via webapp)
OLLAMA_MODEL=qwen3:14b

# Ollama models directory (E-SSD)
OLLAMA_MODELS=/mnt/E-SSD/model_cache

# Data directory containing JSONL dataset subfolders
DATA_DIR=/mnt/C-SSD/desinformacao/coleta_datasets/data/raw

# Semantic embedding model
SEMANTIC_MODEL=paraphrase-multilingual-mpnet-base-v2

# Encoding strategy: chunk_pool, title_label, or truncate
ENCODING_STRATEGY=chunk_pool

# Cache directory for pre-computed embeddings
EMBEDDINGS_CACHE_DIR=./data/embeddings_cache

# HuggingFace/Torch model download directory
MODEL_CACHE_DIR=/mnt/E-SSD/mussi/model_cache

# Google Search API keys (only needed for "web" and "hybrid" modes)
# GOOGLE_API_KEY=your_key_here
# GOOGLE_CSE_ID=your_cse_id_here
```

### Available Ollama Models (on this server)

| Model | Size | Notes |
|---|---|---|
| `llama3.2:1b` | 1.3 GB | Fast but safety filters too aggressive |
| `llama3:8b` | 4.7 GB | Good balance |
| `llama3.1:8b` | 4.9 GB | Good instruction following |
| `deepseek-r1:7b` | 4.7 GB | Reasoning model, slower |
| `qwen3:14b` | 9.3 GB | **Recommended** — very capable, current default |
| `gemma3:27b` | 17 GB | Excellent but large |

---

## 8. Deployment Guide

### Prerequisites

1. **Python 3.10+** with conda environment `cnpq`
2. **Ollama** installed and accessible via CLI
3. **NVIDIA GPU** with CUDA support (A6000 recommended)
4. **JSONL datasets** in the configured `DATA_DIR`

### Step-by-Step

#### 1. Environment Setup

```bash
conda activate cnpq
cd /mnt/C-SSD/desinformacao/mussi/rag_project/ollama4truth
pip install -r requirements.txt
```

#### 2. Start Ollama Server

```bash
# Option A: If Ollama is a system service
sudo systemctl stop ollama
CUDA_VISIBLE_DEVICES=0 OLLAMA_MODELS=/mnt/E-SSD/model_cache ollama serve
# Keep this terminal open

# Option B: If Ollama service is already configured for GPU 0
sudo systemctl start ollama
```

#### 3. Pull Model (first time only)

```bash
ollama pull qwen3:14b
```

#### 4. Start API Server

```bash
conda activate cnpq
cd /mnt/C-SSD/desinformacao/mussi/rag_project/ollama4truth
CUDA_VISIBLE_DEVICES=0 uvicorn api:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete` — BM25 builds in ~5s, semantic encoding takes ~10 min on first run (cached afterward).

#### 5. Access the Webapp

**Local**: `http://localhost:8000`

**Remote (SSH tunnel)**:

```bash
# On your local machine:
ssh -L 8000:localhost:8000 <server-hostname>
# Then open: http://localhost:8000
```

---

## 9. Extending the System

### Adding a New Retrieval Mode

1. Create a new `_retrieve_<mode>()` function in `pipeline/retrieve_evidence.py`
2. Add the mode to the dispatcher in `retrieve_evidence()`
3. Add the option to the `<select id="modeSelect">` in `webapp/index.html`

### Adding a New Verdict Strategy

1. Create a new `classify_<strategy>()` function in `pipeline/classification.py`
2. Add the strategy to the dispatcher in `classify_claim()`
3. Add the option to the `<select id="strategySelect">` in `webapp/index.html`

### Adding a New Data Source

1. Add the dataset entry to `DATASETS` in `pipeline/data_loader.py`
2. Ensure the JSONL file follows the unified schema
3. Place the file in the appropriate subdirectory under `DATA_DIR`

### Switching the Semantic Model

1. Update `SEMANTIC_MODEL` in `.env`
2. Delete any cached embeddings (if `EMBEDDINGS_CACHE_DIR` is set)
3. Restart the server

---

## 10. Troubleshooting

### "Address already in use" when starting Ollama

```bash
# Ollama is already running as a system service
sudo systemctl stop ollama
CUDA_VISIBLE_DEVICES=0 ollama serve
```

### Question generation returns the claim instead of questions

- The LLM safety filter is blocking the topic
- Try a different model via the webapp dropdown, or change `OLLAMA_MODEL` in `.env`
- The system automatically falls back to using the claim as a search query

### No evidence found in RAG mode

- Verify `DATA_DIR` is set correctly in `.env`
- Check that JSONL files exist: `ls $DATA_DIR/*/`
- Run the data loader test: `python -m pipeline.data_loader`

### GPU usage on wrong device

- Set `CUDA_VISIBLE_DEVICES=0` both when starting Ollama AND the API server
- For the Ollama system service: `sudo systemctl edit ollama` → add `Environment="CUDA_VISIBLE_DEVICES=0"`

### Webapp not accessible remotely

- Use SSH port forwarding: `ssh -L 8000:localhost:8000 <server>`
- Ensure the API server is started with `--host 0.0.0.0`

### label_vote shows "Not Enough Evidence" unexpectedly

- Only works with RAG/hybrid modes (web results don't have labels)
- Check if the article labels match the `FALSE_LABELS`/`TRUE_LABELS` sets
- Falls back to `ollama_verdict` automatically when no labels are found
