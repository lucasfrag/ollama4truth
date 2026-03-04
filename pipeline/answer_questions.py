"""
answer_questions.py — Per-question answering step.

For each investigative question + its retrieved evidence, asks the LLM
to answer the question based solely on the evidence.
Produces structured Q&A pairs for the classification step.
"""

import json
import re
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()


def run_ollama(prompt: str, model: str = None) -> str:
    """Executa um modelo Ollama localmente e retorna a resposta como texto.
    Strips thinking tokens from chain-of-thought models."""
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1")

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            text=False,
            capture_output=True,
            check=True
        )
        output = result.stdout.decode("utf-8", errors="ignore").strip()

        # Strip <think>...</think> blocks (qwen3, deepseek-r1 thinking mode)
        output = re.sub(r'<think>[\s\S]*?</think>', '', output).strip()
        # Strip "Thinking..." prefix
        output = re.sub(r'^Thinking\.\.\.[\s\S]*?\.\.\.done thinking\.\s*', '', output, flags=re.IGNORECASE).strip()

        return output
    except subprocess.CalledProcessError as e:
        print("❌ Erro ao executar Ollama:", e.stderr)
        return ""


# Max chars per article in the answering prompt
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "20000"))


def _build_answer_prompt(question: str, evidence_results: list) -> str:
    """Build a prompt to answer a single question based on its evidence."""
    evidence_parts = []
    for i, r in enumerate(evidence_results, 1):
        title = r.get("title", "Sem título")
        text = r.get("full_text") or r.get("snippet", "")
        if len(text) > MAX_EVIDENCE_CHARS:
            text = text[:MAX_EVIDENCE_CHARS] + "... [truncado]"
        evidence_parts.append(f"--- Artigo {i} ---\nTítulo: {title}\nTexto:\n{text}")

    evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(Nenhuma evidência encontrada)"

    return f"""Você é um assistente de checagem de fatos acadêmico. Responda à pergunta abaixo com base EXCLUSIVAMENTE nas evidências fornecidas.

REGRAS:
- Responda APENAS com base nas evidências. Não invente informações.
- Se as evidências não são suficientes para responder, diga "Sem informação suficiente nas evidências."
- Seja conciso e objetivo (máximo 3 frases).
- Responda em português (PT-BR).

Pergunta: "{question}"

Evidências:
{evidence_text}

Resposta:"""


def answer_single_question(question: str, evidence_results: list, model: str = None) -> str:
    """Answer a single question based on its retrieved evidence."""
    if not evidence_results:
        return "Sem evidências disponíveis para responder esta pergunta."

    prompt = _build_answer_prompt(question, evidence_results)
    answer = run_ollama(prompt, model=model).strip()

    # Clean up: remove any leading "Resposta:" prefix the model might add
    answer = re.sub(r'^(Resposta:\s*)', '', answer, flags=re.IGNORECASE).strip()

    if not answer:
        answer = "Não foi possível gerar uma resposta."

    return answer


def answer_all_questions(evidences: list, model: str = None) -> list:
    """
    Answer all questions using their respective evidence.

    Args:
        evidences: List of evidence groups from retrieve_evidence(),
                   each with 'question' and 'results' keys.
        model: Ollama model name.

    Returns:
        The same evidence list, but with an 'answer' field added to each group.
    """
    for ev_group in evidences:
        question = ev_group.get("question", "")
        results = ev_group.get("results", [])

        print(f"  📝 Respondendo: {question[:80]}...")
        answer = answer_single_question(question, results, model=model)
        ev_group["answer"] = answer
        print(f"     → {answer[:100]}...")

    return evidences


if __name__ == "__main__":
    # Manual test
    test_evidences = [
        {
            "question": "Quais órgãos recomendam o uso de máscaras?",
            "results": [
                {
                    "title": "OMS recomenda uso de máscaras",
                    "snippet": "A Organização Mundial da Saúde recomenda o uso de máscaras em locais fechados.",
                }
            ]
        }
    ]

    result = answer_all_questions(test_evidences)
    print(json.dumps(result, indent=4, ensure_ascii=False))
