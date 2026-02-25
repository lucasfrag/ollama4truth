"""
retrieve_evidence.py — Multi-mode evidence retrieval.

Supports 3 modes:
  - "rag":    Local corpus search (BM25 + semantic)
  - "web":    Google Custom Search (original behavior)
  - "hybrid": RAG first, fall back to Google if insufficient results
"""

import os
import json
import time
import requests
from urllib.parse import quote_plus
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CSE_ID")

# Will be set by main.py at startup
_rag_index = None

MIN_RAG_RESULTS = 2  # Minimum RAG results before falling back to web in hybrid mode


def set_rag_index(index):
    """Set the RAG index reference (called from main.py at startup)."""
    global _rag_index
    _rag_index = index


# ============================================================
# Web retrieval (original Google Search logic)
# ============================================================
def google_search(query: str, num_results: int = 5):
    """
    Executa uma busca no Google Custom Search e retorna os resultados relevantes.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        raise ValueError("As variáveis de ambiente GOOGLE_API_KEY e GOOGLE_CX precisam estar configuradas.")

    url = f"https://www.googleapis.com/customsearch/v1?q={quote_plus(query)}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&num={num_results}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
        return results

    except Exception as e:
        print(f"[ERRO] Falha ao buscar '{query}': {e}")
        return []


def _retrieve_web(claim: str, questions: list) -> dict:
    """Original web-only retrieval."""
    all_evidence = {
        "claim": claim,
        "timestamp": datetime.utcnow().isoformat(),
        "evidences": []
    }

    for q in questions:
        print(f"[WEB] Buscando evidências para: {q}")
        results = google_search(q)
        all_evidence["evidences"].append({
            "question": q,
            "results": results
        })
        time.sleep(1.5)

    print(f"[WEB] Evidências coletadas para claim: {claim}")
    return all_evidence


# ============================================================
# RAG retrieval (local corpus)
# ============================================================
def _retrieve_rag(claim: str, questions: list, retrieval_method: str = None) -> dict:
    """Retrieve evidence from local RAG corpus, grouped per question."""
    if _rag_index is None:
        raise RuntimeError("RAG index not initialized. Call init_rag() first.")

    all_evidence = {
        "claim": claim,
        "timestamp": datetime.utcnow().isoformat(),
        "evidences": [],
        "mode": "rag",
    }

    seen_urls = set()  # Deduplicate across questions

    for q in questions:
        print(f"[RAG] Buscando: {q} [method={retrieval_method or 'default'}]")
        results = _rag_index.retrieve(q, k=5, method=retrieval_method)

        # Deduplicate: skip articles already returned for earlier questions
        unique_results = []
        for r in results:
            if r["link"] not in seen_urls:
                seen_urls.add(r["link"])
                unique_results.append(r)

        all_evidence["evidences"].append({
            "question": q,
            "results": unique_results,
        })

    total = sum(len(e["results"]) for e in all_evidence["evidences"])
    print(f"[RAG] {total} evidências encontradas no corpus local")
    return all_evidence


# ============================================================
# Hybrid retrieval (RAG first → web fallback)
# ============================================================
def _retrieve_hybrid(claim: str, questions: list, retrieval_method: str = None) -> dict:
    """
    Try RAG first. If fewer than MIN_RAG_RESULTS found, 
    supplement with Google Search.
    """
    # Step 1: Try RAG
    rag_evidence = _retrieve_rag(claim, questions, retrieval_method=retrieval_method)
    total_rag = sum(len(e.get("results", [])) for e in rag_evidence.get("evidences", []))

    if total_rag >= MIN_RAG_RESULTS:
        print(f"[HYBRID] RAG retornou {total_rag} resultados — suficiente, sem fallback web")
        rag_evidence["mode"] = "hybrid (rag only)"
        return rag_evidence

    # Step 2: RAG insufficient — fall back to web
    print(f"[HYBRID] RAG retornou apenas {total_rag} resultados — buscando na web...")
    web_evidence = _retrieve_web(claim, questions)

    # Merge: RAG results first, then web results
    merged = {
        "claim": claim,
        "timestamp": datetime.utcnow().isoformat(),
        "evidences": [],
        "mode": "hybrid (rag + web fallback)",
    }

    # Add RAG results (per-question)
    for ev in rag_evidence.get("evidences", []):
        if ev.get("results"):
            merged["evidences"].append(ev)

    # Add web results
    for ev in web_evidence.get("evidences", []):
        merged["evidences"].append(ev)

    return merged


# ============================================================
# Main dispatcher
# ============================================================
def retrieve_evidence(claim: str, questions: list, mode: str = "web", retrieval_method: str = None) -> dict:
    """
    Retrieve evidence for a claim using the specified mode.

    Args:
        claim: The claim to verify
        questions: Generated fact-checking questions
        mode: "rag" | "web" | "hybrid"
        retrieval_method: RAG index method: "bm25", "semantic", "hybrid" (for index search)

    Returns:
        Dict with claim, timestamp, evidences, and mode info
    """
    if mode == "rag":
        return _retrieve_rag(claim, questions, retrieval_method=retrieval_method)
    elif mode == "hybrid":
        return _retrieve_hybrid(claim, questions, retrieval_method=retrieval_method)
    else:
        return _retrieve_web(claim, questions)


if __name__ == "__main__":
    # Exemplo de teste manual (web mode only)
    claim = "O café ajuda a melhorar a memória de longo prazo."
    questions = [
        "O consumo de café melhora a memória de longo prazo?",
        "Há estudos científicos que confirmam essa relação?",
        "A cafeína influencia processos de consolidação de memória?"
    ]

    data = retrieve_evidence(claim, questions, mode="web")
    print(json.dumps(data, indent=4, ensure_ascii=False))
