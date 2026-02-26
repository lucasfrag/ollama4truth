"""
classification.py — Claim classification with switchable verdict strategies.

Strategies:
  - "ollama_verdict": Uses Ollama LLM to classify (original behavior)
  - "label_vote":     Majority voting on article labels from RAG corpus
"""

import json
import re
import subprocess
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import label sets for label voting
from pipeline.data_loader import FALSE_LABELS, TRUE_LABELS


def run_ollama(prompt: str, model: str = None) -> str:
    """
    Executa um modelo Ollama localmente e retorna a resposta como texto.
    """
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1")

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            text=False,
            capture_output=True,
            check=True
        )
        return result.stdout.decode("utf-8", errors="ignore").strip()
    except subprocess.CalledProcessError as e:
        print("❌ Erro ao executar Ollama:", e.stderr)
        return ""


# ============================================================
# Strategy 1: Ollama LLM Verdict with Multi-Run Consistency
# ============================================================
# Max characters per article in the LLM prompt (safety net for context window)
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "20000"))

# Number of consistency runs for confidence calculation
CONSISTENCY_RUNS = int(os.getenv("CONSISTENCY_RUNS", "3"))

VALID_CLASSES = ["Apoiada", "Refutada", "Insuficiente", "Contraditória"]


def _build_evidence_text(evidences: list) -> str:
    """
    Build a structured text block from evidence results for the LLM prompt.
    Uses full_text when available, falls back to snippet.
    Truncates individual articles to MAX_EVIDENCE_CHARS.
    """
    parts = []
    article_num = 0

    for ev_group in evidences:
        for r in ev_group.get("results", []):
            article_num += 1
            title = r.get("title", "Sem título")
            text = r.get("full_text") or r.get("snippet", "")

            if len(text) > MAX_EVIDENCE_CHARS:
                text = text[:MAX_EVIDENCE_CHARS] + "... [truncado]"

            parts.append(
                f"--- Artigo {article_num} ---\n"
                f"Título: {title}\n"
                f"Texto:\n{text}"
            )

    if not parts:
        return "(Nenhuma evidência encontrada)"

    return "\n\n".join(parts)


def _build_classification_prompt(claim: str, evidences: list) -> str:
    """
    Build the classification prompt including per-question answers and evidence.
    Expects evidences enriched with 'answer' field from answer_questions step.
    """
    qa_parts = []
    for i, ev_group in enumerate(evidences, 1):
        question = ev_group.get("question", f"Pergunta {i}")
        answer = ev_group.get("answer", "Sem resposta disponível.")
        results = ev_group.get("results", [])

        # Build evidence context for this question (title + truncated text)
        evidence_parts = []
        for j, r in enumerate(results[:3], 1):  # Top 3 per question for prompt size
            title = r.get("title", "Sem título")
            text = r.get("full_text") or r.get("snippet", "")
            if len(text) > 5000:
                text = text[:5000] + "... [truncado]"
            evidence_parts.append(
                f"  Artigo {j}: {title}\n"
                f"  Texto: {text}"
            )

        evidence_list = "\n\n".join(evidence_parts) if evidence_parts else "  (Nenhuma evidência encontrada)"

        qa_parts.append(
            f"--- Pergunta {i} ---\n"
            f"Pergunta: {question}\n"
            f"Resposta baseada nas evidências: {answer}\n"
            f"Artigos consultados:\n{evidence_list}"
        )

    qa_text = "\n\n".join(qa_parts) if qa_parts else "(Nenhuma pergunta ou evidência disponível)"

    return f"""Você é um sistema de checagem de fatos acadêmico. Sua tarefa é avaliar se uma ALEGAÇÃO é verdadeira ou falsa, com base nas respostas às perguntas investigativas e nas evidências coletadas.

IMPORTANTE — LEIA COM ATENÇÃO:
- Você deve avaliar se a ALEGAÇÃO EXATA fornecida é apoiada ou refutada.
- Preste muita atenção a NEGAÇÕES na alegação.
- Analise as RESPOSTAS às perguntas investigativas — elas resumem o que as evidências dizem.
- Analise o SENTIDO SEMÂNTICO da alegação e compare com as respostas.

Alegação: "{claim}"

Perguntas investigativas e respostas:
{qa_text}

Classifique a alegação em uma das categorias:
- Apoiada: a alegação É VERDADEIRA segundo as evidências
- Refutada: a alegação É FALSA segundo as evidências
- Insuficiente: evidências insuficientes
- Contraditória: evidências contraditórias

Saída obrigatória: JSON no formato abaixo, sem texto adicional:

{{
  "classification": "<Apoiada | Refutada | Insuficiente | Contraditória>",
  "justification": "Texto explicativo."
}}
"""


def _run_single_classification(prompt: str, model: str = None) -> dict:
    """
    Run a single LLM classification and return parsed result.
    Returns dict with 'classification' and 'justification' keys.
    """
    output = run_ollama(prompt, model=model).strip()

    json_match = re.search(r'\{[\s\S]*\}', output)
    if json_match:
        output = json_match.group(0)

    try:
        parsed = json.loads(output)
        classification = parsed.get("classification", "unverified").strip()
        if classification not in VALID_CLASSES:
            classification = "unverified"
        justification = parsed.get("justification", "").strip()
    except json.JSONDecodeError:
        classification = "unverified"
        justification = output[:300] if output else "Não foi possível interpretar a resposta do modelo."

    return {"classification": classification, "justification": justification}


def _aggregate_consistency(runs: list, claim: str) -> dict:
    """
    Aggregate N classification runs into a single result with consistency-based confidence.

    Confidence = (runs agreeing with majority / total runs) × 100
    """
    from collections import Counter

    votes = [r["classification"] for r in runs]
    counter = Counter(votes)
    majority_class, majority_count = counter.most_common(1)[0]
    confidence = round((majority_count / len(runs)) * 100, 1)

    # Use the justification from the first run that returned the majority classification
    justification = next(
        (r["justification"] for r in runs if r["classification"] == majority_class),
        "Sem justificativa disponível."
    )

    result = {
        "claim": claim,
        "classification": majority_class,
        "justification": justification,
        "confidence": confidence,
        "consistency_detail": votes,
        "strategy": "ollama_verdict",
        "timestamp": datetime.utcnow().isoformat()
    }

    print(f"✅ [OLLAMA] Alegação classificada como: {majority_class.upper()} "
          f"(consistência: {confidence}% — {majority_count}/{len(runs)} concordam)")
    return result


def classify_ollama_verdict(claim: str, evidences: list, model: str = None, n_runs: int = None) -> dict:
    """
    Classifica a alegação usando Ollama LLM com múltiplas rodadas de consistência.
    Confidence is computed as the percentage of runs that agree on the majority classification.
    """
    n_runs = n_runs or CONSISTENCY_RUNS
    prompt = _build_classification_prompt(claim, evidences)

    runs = []
    for i in range(n_runs):
        print(f"  🔄 Rodada {i + 1}/{n_runs}...")
        run_result = _run_single_classification(prompt, model=model)
        runs.append(run_result)
        print(f"     → {run_result['classification']}")

    return _aggregate_consistency(runs, claim)


# ============================================================
# Strategy 2: Label Voting (uses article labels from RAG corpus)
# ============================================================
def classify_label_vote(claim: str, evidences: list) -> dict:
    """
    Classify claim by majority voting on article labels from RAG results.

    Counts how many retrieved articles have labels in FALSE_LABELS vs TRUE_LABELS.
    Only works with RAG/hybrid results that carry article labels.
    """
    false_count = 0
    true_count = 0
    other_count = 0
    label_breakdown = {}
    total = 0

    for ev_group in evidences:
        for result in ev_group.get("results", []):
            label = result.get("label", "")
            if not label:
                continue

            total += 1
            label_breakdown[label] = label_breakdown.get(label, 0) + 1

            if label in FALSE_LABELS:
                false_count += 1
            elif label in TRUE_LABELS:
                true_count += 1
            else:
                other_count += 1

    # Determine verdict
    if total == 0:
        classification = "Insuficiente"
        confidence = 0
        justification = "Nenhum artigo com classificação encontrado nas evidências."
    elif false_count > true_count and false_count > other_count:
        classification = "Refutada"
        confidence = round((false_count / total) * 100, 1)
        justification = f"{false_count} de {total} artigos classificam como falso/enganoso."
    elif true_count > false_count and true_count > other_count:
        classification = "Apoiada"
        confidence = round((true_count / total) * 100, 1)
        justification = f"{true_count} de {total} artigos classificam como verdadeiro."
    else:
        classification = "Insuficiente"
        confidence = round(max(false_count, true_count, other_count) / total * 100, 1) if total > 0 else 0
        justification = f"Sem consenso claro: {false_count} falso, {true_count} verdadeiro, {other_count} outro."

    result = {
        "claim": claim,
        "classification": classification,
        "justification": justification,
        "confidence": confidence,
        "strategy": "label_vote",
        "label_breakdown": label_breakdown,
        "timestamp": datetime.utcnow().isoformat()
    }

    print(f"✅ [LABEL_VOTE] Alegação classificada como: {classification.upper()} ({confidence}%)")
    return result


# ============================================================
# Main dispatcher
# ============================================================
def classify_claim(claim: str, evidences: list, strategy: str = "ollama_verdict", model: str = None) -> dict:
    """
    Classify a claim using the specified strategy.

    Args:
        claim: The claim to classify
        evidences: List of evidence groups from retrieve_evidence()
        strategy: "ollama_verdict" | "label_vote"
        model: Ollama model name (only for ollama_verdict strategy)

    Returns:
        Dict with classification, justification, confidence, strategy
    """
    if strategy == "label_vote":
        # Check if evidence has labels (RAG results have them, web results don't)
        has_labels = any(
            result.get("label")
            for ev_group in evidences
            for result in ev_group.get("results", [])
        )
        if not has_labels:
            print("[WARN] label_vote selected but no article labels in evidence — falling back to ollama_verdict")
            return classify_ollama_verdict(claim, evidences, model)
        return classify_label_vote(claim, evidences)
    else:
        return classify_ollama_verdict(claim, evidences, model)


if __name__ == "__main__":
    # Teste manual
    claim = "O café ajuda a melhorar a memória de longo prazo."
    evidences = [
        {
            "question": "O consumo de café melhora a memória de longo prazo?",
            "results": [
                {"title": "Estudo da Universidade X aponta melhora cognitiva", "link": "https://example.com"}
            ]
        }
    ]

    result = classify_claim(claim, evidences, strategy="ollama_verdict")
    print(json.dumps(result, indent=4, ensure_ascii=False))
