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
- **Evidence Retrieval**: Searches a local corpus using BM25, semantic, or hybrid retrieval — switchable at runtime
- **Per-Question Answering**: Each question is answered individually by the LLM based on its retrieved evidence
- **Claim Classification**: Two strategies — LLM-based verdict with multi-run consistency, and corpus label majority voting
- **Multi-Run Consistency**: Classification runs N times (default: 3); confidence = % of runs that agree (not LLM self-reported)
- **Runtime Model Selection**: Switch between any downloaded Ollama model via webapp dropdown
- **Claim History**: Every analysis is saved to `data/history.jsonl`; browse, expand Q&A details, and clear past results
- **Real-time Streaming**: Results stream to the web interface via Server-Sent Events (SSE)
- **Dark Theme UI**: Dark navy background with Portuguese labels

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

### Pipeline Flow (4 Steps)

```
User Claim
     │
     ▼
┌─────────────────────┐
│  1. Question Gen.    │  ← Ollama LLM
│  generate_questions()│
└─────────┬───────────┘
          │ 3-5 investigative questions
          ▼
┌─────────────────────┐
│  2. Evidence Retr.   │  ← Mode: rag | web | hybrid
│  retrieve_evidence() │
└─────────┬───────────┘
          │ Evidence grouped per question
          ▼
┌─────────────────────┐
│  3. Answer Questions │  ← Ollama LLM (per question)
│  answer_all_questions│
└─────────┬───────────┘
          │ Q&A pairs with evidence
          ▼
┌─────────────────────┐
│  4. Classification   │  ← Strategy: ollama_verdict | label_vote
│  classify_claim()    │  ← Multi-run consistency (3 runs)
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
│   ├── answer_questions.py        # Per-question answering via Ollama
│   └── classification.py          # Verdict strategies (LLM + label voting)
│
├── webapp/
│   ├── index.html                 # UI with mode/retrieval/strategy/model dropdowns + history
│   ├── script.js                  # SSE client, Q&A rendering, history panel
│   └── style.css                  # Dark theme styling
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
       ├─ pipeline/answer_questions.py     (uses Ollama)
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

**FALSE_LABELS** (22 variants): `falso`, `fake`, `enganoso`, `distorcido`, `golpe`, `manipulado`, `boato`, etc.

**TRUE_LABELS** (8 variants): `verdadeiro`, `fato`, `verdade`, `correto`, `real`, `comprovado`, `confirmado`, etc.

### 3.4 Data Loader (`pipeline/data_loader.py`)

**Key functions:**

| Function | Description |
|---|---|
| `load_corpus(data_dir)` | Loads all 6 JSONL datasets, returns unified list of dicts |
| `full_text(article)` | Concatenates titulo + subtitulo + texto for retrieval |
| `_normalize_label(label)` | Strips accents, lowercases labels |

---

## 4. Pipeline Components

### 4.1 Question Generation (`pipeline/generate_questions.py`)

Generates 3-5 neutral, investigative questions from a claim using Ollama.

**Model**: Configured via `OLLAMA_MODEL` in `.env`. Can be overridden at runtime.

**Prompt strategy**: The prompt frames the task as an academic fact-checking system and uses an "INPUT TEXT" wrapper to avoid triggering safety filters.

**Fallback behavior**: If the LLM refuses or generates no valid questions, the original claim text is used as the search query.

### 4.2 Evidence Retrieval (`pipeline/retrieve_evidence.py`)

Dispatches evidence retrieval to one of 3 modes:

#### Mode: `rag` (Local Corpus)

- Searches the local BM25/semantic index for each generated question
- Returns articles with source, label, and snippet
- Each question retrieves its own top-k results independently

#### Mode: `web` (Google Search)

- Uses Google Custom Search API
- Returns web results with title, link, and snippet

#### Mode: `hybrid` (RAG → Web Fallback)

- First tries RAG retrieval
- If fewer than 2 results, falls back to Google Search

### 4.3 Per-Question Answering (`pipeline/answer_questions.py`)

**NEW** — This step answers each investigative question individually using its retrieved evidence.

For each question + evidence pair, the LLM is prompted to:

- Answer **ONLY** based on the provided evidence
- Be concise (max 3 sentences)
- Say "Sem informação suficiente" if evidence doesn't cover the question

This produces structured Q&A pairs that feed into the classification step, enabling the LLM to reason from intermediate answers rather than raw article dumps.

**Key functions:**

| Function | Description |
|---|---|
| `answer_single_question(question, results, model)` | Answer one question from its evidence |
| `answer_all_questions(evidences, model)` | Answer all questions, adds `answer` field to each evidence group |

### 4.4 RAG Index (`pipeline/rag_retrieval.py`)

The `RAGIndex` class manages the retrieval indices. **Both BM25 and semantic indices are always built at startup**, allowing runtime switching.

**Retrieval methods (switchable per query):**

| Method | Description | Quality |
|---|---|---|
| `bm25` | BM25Okapi keyword matching | Good for exact terms |
| `semantic` | Cosine similarity via sentence-transformer with chunk_pool encoding | Better for paraphrases |
| `hybrid` | Weighted combination of BM25 + semantic scores | Best overall |

**Encoding strategy (chunk_pool):**

- Each article's `full_text` is split into 500-character chunks
- All chunks are embedded individually
- At retrieval: cosine similarity computed per chunk, **max-sim** taken per article
- First run encodes ~156k chunks (~10 min), cached as `.npy` files for instant restarts

### 4.5 Classification (`pipeline/classification.py`)

Two verdict strategies with automatic fallback:

#### Strategy: `ollama_verdict`

- Sends the claim + per-question Q&A pairs to Ollama LLM
- **Multi-run consistency**: Runs the classification N times (default: 3, configurable via `CONSISTENCY_RUNS`)
- **Confidence = (agreeing runs / total runs) × 100** — not LLM self-reported
- Classifies as: `Apoiada`, `Refutada`, `Insuficiente`, or `Contraditória`
- Classification prompt includes structured Q&A pairs (question + answer + article titles), not raw article text
- Article fact-checker labels are NOT shown to the LLM to prevent label bias

#### Strategy: `label_vote`

- Counts article labels from RAG results
- Compares `FALSE_LABELS` count vs `TRUE_LABELS` count
- Majority wins → verdict + confidence percentage
- **Auto-fallback**: If evidence has no labels (e.g., web-only results), falls back to `ollama_verdict`

**Output format (ollama_verdict):**

```python
{
    "claim": "Vacinas causam autismo",
    "classification": "Refutada",
    "justification": "As evidências mostram que não há relação entre vacinas e autismo.",
    "confidence": 100.0,
    "consistency_detail": ["Refutada", "Refutada", "Refutada"],
    "strategy": "ollama_verdict",
    "timestamp": "2026-02-26T18:00:00.000000"
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
| `ollama_model` | string | *(from .env)* | Ollama model to use |

**SSE stream messages:**

```
🚀 Iniciando pipeline para: "..." [modo=rag, ...]
🧩 Gerando perguntas...
✅ 3 perguntas geradas.
🔍 Buscando evidências...
✅ 15 evidências encontradas.
📝 Respondendo pergunta 1/3: ...
   ✔ A OMS recomenda o uso de máscaras...
📝 Respondendo pergunta 2/3: ...
🧠 Classificando alegação (rodada 1/3)...
   ✔ Rodada 1: Apoiada
🧠 Classificando alegação (rodada 2/3)...
✅ Classificação: Apoiada (consistência: 100% — 3/3 concordam)
🎯 Pipeline concluído com sucesso!
{...final JSON result...}
```

### `GET /health`

Health check endpoint. Returns `{"status": "ok", ...}`.

### `GET /` (Webapp)

Serves the web interface from the `webapp/` directory.

### `GET /models`

Returns available Ollama models for the dropdown selector.

### `GET /history`

Returns all past claim analyses, most recent first.

### `GET /history/clear`

Deletes the `data/history.jsonl` file, clearing all history.

---

## 6. Webapp Interface

The webapp provides a single-page interface for claim verification with a **dark navy theme**.

### UI Components

1. **Claim input**: Text field for entering the claim (label: "alegação")
2. **Mode selector**: Modo de Busca (RAG/Web/Hybrid)
3. **Retrieval method selector**: Método de Recuperação (BM25/Semântico/Híbrido)
4. **Strategy selector**: Estratégia de Veredito (Ollama LLM/Votação por Labels)
5. **Model selector**: Modelo Ollama (populated from `GET /models`)
6. **Sidebar (Perguntas)**: Lists generated questions with evidence counts; clickable to scroll to Q&A card
7. **Main panel**: Displays verdict badge, consistency score, justification, and Q&A cards
8. **Q&A cards**: Each card shows: question → LLM answer → evidence articles (with source/label badges)
9. **History panel**: Toggleable panel showing past analyses, expandable to see full Q&A details

### Display Features

- **Source badges**: Uppercase colored badges (BOATOS, LUPA, G1, etc.)
- **Label badges**: Color-coded — red for false/enganoso, green for verdadeiro/fato, gray for others
- **Verdict badges**: Green (Apoiada), Red (Refutada), Yellow (Insuficiente)
- **Consistency score**: Shows "100% (3/3)" instead of LLM self-reported confidence

---

## 7. Configuration

All configuration is in `.env`:

```bash
# GPU selection
CUDA_VISIBLE_DEVICES=0

# Ollama model (default, overridable via webapp)
OLLAMA_MODEL=qwen3:14b

# Ollama models directory
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

# Number of consistency runs for LLM classification confidence
CONSISTENCY_RUNS=3

# Google Search API keys (only needed for "web" and "hybrid" modes)
# GOOGLE_API_KEY=your_key_here
# GOOGLE_CSE_ID=your_cse_id_here
```

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

---

## 10. Troubleshooting

### "Address already in use" when starting Ollama

```bash
sudo systemctl stop ollama
CUDA_VISIBLE_DEVICES=0 ollama serve
```

### Question generation returns the claim instead of questions

- The LLM safety filter is blocking the topic
- Try a different model via the webapp dropdown
- The system automatically falls back to using the claim as a search query

### No evidence found in RAG mode

- Verify `DATA_DIR` is set correctly in `.env`
- Check that JSONL files exist: `ls $DATA_DIR/*/`

### Classification always says "Refutada" for true claims

- This was caused by article fact-checker labels ("falso") biasing the LLM
- **Fixed**: Article labels are no longer shown to the LLM in the classification prompt
- The LLM now reasons from per-question answers + article content, not labels

### Confidence score shows only 33%, 67%, or 100%

- This is expected. Confidence is computed as multi-run consistency (N=3 by default)
- Possible values: 33% (no agreement), 67% (2/3 agree), 100% (3/3 agree)
- Adjust with `CONSISTENCY_RUNS` env var (e.g., 5 for finer granularity)

### GPU usage on wrong device

- Set `CUDA_VISIBLE_DEVICES=0` both when starting Ollama AND the API server

### Webapp not accessible remotely

- Use SSH port forwarding: `ssh -L 8000:localhost:8000 <server>`
- Ensure the API server is started with `--host 0.0.0.0`
