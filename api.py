"""
api.py — FastAPI server for Ollama4Truth with RAG support.

Endpoints:
  GET /              → Webapp (served from webapp/ directory)
  GET /analyze-stream?claim=...&mode=rag&strategy=ollama_verdict&retrieval_method=bm25&ollama_model=llama3.1:8b
  GET /models        → List available Ollama models
  GET /history       → List all past analyses
  GET /history/clear → Clear history
  GET /health
"""

import os
import subprocess
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json

from main import run_pipeline_stream, init_rag, load_history, HISTORY_FILE

app = FastAPI(
    title="Ollama4Truth API",
    description="API para verificação de fatos com RAG + Ollama",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Initialize RAG index at server startup."""
    init_rag()


@app.get("/analyze-stream")
def analyze_stream(
    claim: str = Query(..., description="Claim to verify"),
    mode: str = Query("rag", description="Retrieval mode: rag, web, hybrid"),
    strategy: str = Query("ollama_verdict", description="Verdict strategy: ollama_verdict, label_vote"),
    retrieval_method: str = Query("bm25", description="RAG retrieval method: bm25, semantic, hybrid"),
    ollama_model: str = Query("", description="Ollama model name (empty = use .env default)"),
):
    """
    Analyze a claim via SSE streaming.
    """
    valid_modes = ("rag", "web", "hybrid")
    valid_strategies = ("ollama_verdict", "label_vote")
    valid_retrieval_methods = ("bm25", "semantic", "hybrid")

    if mode not in valid_modes:
        mode = "rag"
    if strategy not in valid_strategies:
        strategy = "ollama_verdict"
    if retrieval_method not in valid_retrieval_methods:
        retrieval_method = "bm25"

    # Empty string means use default from .env
    model_to_use = ollama_model if ollama_model else None

    def generate():
        for log, data in run_pipeline_stream(claim, mode=mode, strategy=strategy, retrieval_method=retrieval_method, ollama_model=model_to_use):
            yield f"data: {log}\n\n"
            if data:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API está rodando e pronta para receber claims"}


@app.get("/models")
def list_models():
    """Return list of available Ollama models."""
    default_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split()
            if parts:
                name = parts[0]  # e.g. "llama3.1:8b"
                size = parts[2] + " " + parts[3] if len(parts) >= 4 else ""  # e.g. "4.9 GB"
                models.append({"name": name, "size": size})
        return {"default": default_model, "models": models}
    except Exception as e:
        print(f"[WARN] Failed to list Ollama models: {e}")
        return {"default": default_model, "models": [{"name": default_model, "size": ""}]}


@app.get("/history")
def get_history():
    """Return all past analysis results, most recent first."""
    entries = load_history()
    entries.reverse()  # Most recent first
    return {"count": len(entries), "entries": entries}


@app.get("/history/clear")
def clear_history():
    """Clear the analysis history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return {"status": "ok", "message": "History cleared"}


# Serve the webapp at root (must be AFTER API routes)
webapp_dir = os.path.join(os.path.dirname(__file__), "webapp")
app.mount("/", StaticFiles(directory=webapp_dir, html=True), name="webapp")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
