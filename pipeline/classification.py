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
# Strategy 1: Ollama LLM Verdict (original behavior)
# ============================================================
def classify_ollama_verdict(claim: str, evidences: list, model: str = None) -> dict:
    """
    Classifica a claim usando Ollama LLM com base nas evidências coletadas.
    """
    VALID_CLASSES = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking"]

    prompt = f"""Você é um sistema de checagem de fatos acadêmico. Sua tarefa é avaliar se uma CLAIM (afirmação) é verdadeira ou falsa, com base nas evidências fornecidas.

IMPORTANTE — LEIA COM ATENÇÃO:
- Você deve avaliar se a CLAIM EXATA fornecida é apoiada ou refutada pelas evidências.
- Preste muita atenção a NEGAÇÕES na claim. Exemplos:
  * "Vacinas NÃO causam autismo" → se as evidências dizem que é FALSO que vacinas causam autismo, então esta claim é SUPPORTED (apoiada), pois a claim nega algo que é de fato falso.
  * "Vacinas causam autismo" → se as evidências dizem que é FALSO, então esta claim é REFUTED.
- Os artigos de fact-checking frequentemente têm rótulos como "falso", "enganoso" etc. Esses rótulos se referem ao TÓPICO ORIGINAL da desinformação, NÃO necessariamente à claim que você está avaliando.
- Analise o SENTIDO SEMÂNTICO da claim e compare com o que as evidências dizem.

Claim: "{claim}"

Evidências coletadas:
{json.dumps(evidences, indent=2, ensure_ascii=False)}

Classifique a claim em uma das categorias:
- Supported: a claim É VERDADEIRA segundo as evidências
- Refuted: a claim É FALSA segundo as evidências
- Not Enough Evidence: evidências insuficientes
- Conflicting Evidence/Cherry-picking: evidências contraditórias

Indique um nível de confiança (0 a 100%).

Saída obrigatória: JSON no formato abaixo, sem texto adicional:

{{
  "classification": "<Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherry-picking>",
  "justification": "Texto explicativo curto",
  "confidence": <0-100>
}}
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
        confidence = parsed.get("confidence", 0)
    except json.JSONDecodeError:
        classification = "unverified"
        justification = output[:300] if output else "Não foi possível interpretar a resposta do modelo."
        confidence = 0

    result = {
        "claim": claim,
        "classification": classification,
        "justification": justification,
        "confidence": confidence,
        "strategy": "ollama_verdict",
        "timestamp": datetime.utcnow().isoformat()
    }

    print(f"✅ [OLLAMA] Claim classificada como: {classification.upper()}")
    return result


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
        classification = "Not Enough Evidence"
        confidence = 0
        justification = "Nenhum artigo com classificação encontrado nas evidências."
    elif false_count > true_count and false_count > other_count:
        classification = "Refuted"
        confidence = round((false_count / total) * 100, 1)
        justification = f"{false_count} de {total} artigos classificam como falso/enganoso."
    elif true_count > false_count and true_count > other_count:
        classification = "Supported"
        confidence = round((true_count / total) * 100, 1)
        justification = f"{true_count} de {total} artigos classificam como verdadeiro."
    else:
        classification = "Not Enough Evidence"
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

    print(f"✅ [LABEL_VOTE] Claim classificada como: {classification.upper()} ({confidence}%)")
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
