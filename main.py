"""
main.py — Ollama4Truth pipeline orchestrator with RAG support.

Supports 3 retrieval modes (rag/web/hybrid) and 2 verdict strategies 
(ollama_verdict/label_vote), configurable at runtime.
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Model cache: ensure all HuggingFace/Torch models are stored on E-SSD
# Must be set BEFORE importing any model-loading modules
# ============================================================
MODEL_CACHE_BASE = os.getenv("MODEL_CACHE_DIR", "/mnt/E-SSD/mussi/model_cache")
os.environ['HF_HOME'] = f'{MODEL_CACHE_BASE}/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = f'{MODEL_CACHE_BASE}/hub'
os.environ['TRANSFORMERS_CACHE'] = f'{MODEL_CACHE_BASE}/hub'
os.environ['TORCH_HOME'] = f'{MODEL_CACHE_BASE}/torch'
os.environ['XDG_CACHE_HOME'] = f'{MODEL_CACHE_BASE}/xet'

# Ollama models directory (so ollama CLI commands use E-SSD)
if os.getenv("OLLAMA_MODELS"):
    os.environ['OLLAMA_MODELS'] = os.getenv("OLLAMA_MODELS")

from pipeline.generate_questions import generate_questions
from pipeline.retrieve_evidence import retrieve_evidence, set_rag_index
from pipeline.answer_questions import answer_single_question, answer_all_questions
from pipeline.classification import (
    classify_claim,
    _build_classification_prompt,
    _run_single_classification,
    _aggregate_consistency,
    CONSISTENCY_RUNS,
)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "data", "history.jsonl")


def _append_to_history(result: dict):
    """Append a pipeline result to the history JSONL file."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[HISTORY] Appended to {HISTORY_FILE}")


def load_history() -> list:
    """Load all history entries from the JSONL file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    entries = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries

# ============================================================
# RAG initialization (called once at server startup)
# ============================================================
_rag_initialized = False


def init_rag():
    """
    Load the article corpus and build the RAG index.
    Called once at server startup. Safe to call multiple times (no-op after first).
    """
    global _rag_initialized
    if _rag_initialized:
        return

    data_dir = os.getenv("DATA_DIR", "/mnt/C-SSD/desinformacao/coleta_datasets/data/raw")

    print(f"\n{'='*60}")
    print(f"  Initializing RAG index (BM25 + Semantic)")
    print(f"  Data dir: {data_dir}")
    print(f"{'='*60}\n")

    from pipeline.data_loader import load_corpus
    from pipeline.rag_retrieval import RAGIndex

    corpus = load_corpus(data_dir)

    if not corpus:
        print("[ERROR] No articles loaded! Check DATA_DIR path.")
        return

    rag_index = RAGIndex(corpus)
    set_rag_index(rag_index)
    _rag_initialized = True

    print(f"\n{'='*60}")
    print(f"  RAG index ready: {len(corpus)} articles indexed")
    print(f"{'='*60}\n")


# ============================================================
# Pipeline execution
# ============================================================
def run_pipeline(claim: str, mode: str = "rag", strategy: str = "ollama_verdict", retrieval_method: str = None, ollama_model: str = None):
    """
    Executa o pipeline completo:
    1. Geração de perguntas
    2. Recuperação de evidências (rag/web/hybrid)
    3. Respostas às perguntas com base nas evidências
    4. Classificação (ollama_verdict/label_vote)
    """
    model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    print(f"\n🚀 Iniciando pipeline para a alegação:\n   \"{claim}\"")
    print(f"   Modo: {mode} | Estratégia: {strategy} | Recuperação: {retrieval_method or 'default'} | Modelo: {model}\n")

    # === 1️⃣ Gerar perguntas ===
    questions_output = generate_questions(claim, model=model)
    questions = [q for q in questions_output.get("questions", []) if isinstance(q, str)]
    print(f"\n✅ {len(questions)} perguntas geradas.")

    # === 2️⃣ Buscar evidências ===
    evidence_output = retrieve_evidence(claim, questions, mode=mode, retrieval_method=retrieval_method)

    # === 3️⃣ Responder às perguntas ===
    evidences = evidence_output.get("evidences", [])
    print(f"\n📝 Respondendo às perguntas com base nas evidências...")
    answer_all_questions(evidences, model=model)

    # === 4️⃣ Classificação ===
    classification_output = classify_claim(
        claim,
        evidences,
        strategy=strategy,
        model=model,
    )

    # === Salvar tudo em um JSON final ===
    final_result = {
        "claim": claim,
        "timestamp": datetime.now().isoformat(),
        "ollama_model": model,
        "questions": questions_output,
        "evidences": evidences,
        "label": classification_output.get("classification"),
        "rationale": classification_output.get("justification"),
        "confidence": classification_output.get("confidence"),
        "consistency_detail": classification_output.get("consistency_detail"),
        "mode": mode,
        "strategy": strategy,
        "retrieval_method": retrieval_method or "bm25",
    }

    with open("data/results.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    _append_to_history(final_result)

    print("\n🎯 Pipeline concluído com sucesso!")
    print("📁 Resultados salvos em: data/results.json")

    return final_result


def run_pipeline_stream(claim: str, mode: str = "rag", strategy: str = "ollama_verdict", retrieval_method: str = None, ollama_model: str = None):
    """
    Executa o pipeline passo a passo, emitindo logs via yield.
    Ideal para streaming em tempo real (SSE).
    """
    model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    yield f"🚀 Iniciando pipeline para: \"{claim}\" [modo={mode}, estratégia={strategy}, recuperação={retrieval_method or 'default'}, modelo={model}]", None

    # === 1️⃣ Gerar perguntas ===
    yield f"🧩 Gerando perguntas (modelo={model})...", None
    questions_output = generate_questions(claim, model=model)
    questions = [q for q in questions_output.get("questions", []) if isinstance(q, str)]
    yield f"✅ {len(questions)} perguntas geradas.", None

    # === 2️⃣ Buscar evidências ===
    yield f"🔍 Buscando evidências (modo={mode}, recuperação={retrieval_method or 'default'})...", None
    evidence_output = retrieve_evidence(claim, questions, mode=mode, retrieval_method=retrieval_method)
    evidences = evidence_output.get("evidences", [])
    total_ev = sum(len(e.get("results", [])) for e in evidences)
    yield f"✅ {total_ev} evidências encontradas.", None

    # === 3️⃣ Responder às perguntas ===
    yield f"📝 Respondendo às perguntas com base nas evidências...", None
    for i, ev_group in enumerate(evidences, 1):
        question = ev_group.get("question", "")
        results = ev_group.get("results", [])
        yield f"📝 Respondendo pergunta {i}/{len(evidences)}: {question[:60]}...", None
        answer = answer_single_question(question, results, model=model)
        ev_group["answer"] = answer
        yield f"   ✔ {answer[:100]}...", None
    yield f"✅ {len(evidences)} perguntas respondidas.", None

    # === 4️⃣ Classificação ===
    if strategy == "ollama_verdict":
        n_runs = CONSISTENCY_RUNS
        yield f"🧠 Classificando alegação ({n_runs} rodadas de consistência)...", None

        prompt = _build_classification_prompt(claim, evidences)

        runs = []
        for i in range(n_runs):
            yield f"🧠 Classificando alegação (rodada {i + 1}/{n_runs})...", None
            run_result = _run_single_classification(prompt, model=model)
            runs.append(run_result)
            yield f"   ✔ Rodada {i + 1}: {run_result['classification']}", None

        classification_output = _aggregate_consistency(runs, claim)
        majority_count = sum(1 for r in runs if r["classification"] == classification_output["classification"])
        yield (f"✅ Classificação: {classification_output['classification']} "
               f"(consistência: {classification_output['confidence']}% — {majority_count}/{n_runs} concordam)"), None
    else:
        yield f"🧠 Classificando alegação (estratégia={strategy})...", None
        classification_output = classify_claim(
            claim,
            evidences,
            strategy=strategy,
            model=model,
        )
        yield f"✅ Classificação concluída: {classification_output.get('classification')}", None

    # === Resultado final ===
    final_result = {
        "claim": claim,
        "timestamp": datetime.now().isoformat(),
        "ollama_model": model,
        "questions": questions_output,
        "evidences": evidences,
        "label": classification_output.get("classification"),
        "rationale": classification_output.get("justification"),
        "confidence": classification_output.get("confidence"),
        "consistency_detail": classification_output.get("consistency_detail"),
        "mode": mode,
        "strategy": strategy,
        "retrieval_method": retrieval_method or "bm25",
    }

    with open("data/results.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    _append_to_history(final_result)

    yield "🎯 Pipeline concluído com sucesso!", final_result
    return final_result


if __name__ == "__main__":
    # Initialize RAG for CLI usage
    init_rag()

    example_claim = "O café ajuda a melhorar a memória de longo prazo."
    run_pipeline(example_claim, mode="rag", strategy="label_vote")
