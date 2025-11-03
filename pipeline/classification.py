import json
import re
import subprocess
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

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


def classify_claim(claim: str, evidences: list, model: str = None) -> dict:
    """
    Classifica a claim com base nas evidências coletadas.
    Retorna um dicionário contendo a classificação e a justificativa.
    """

    VALID_CLASSES = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking"]


    # 🔹 Criação do prompt padronizado
    prompt = f"""
Você é um sistema de checagem de fatos.

Claim: "{claim}"

Evidências coletadas:
{json.dumps(evidences, indent=2, ensure_ascii=False)}

Com base nessas evidências, classifique a claim em uma das categorias abaixo:
{VALID_CLASSES}


Explique brevemente o motivo da classificação.

Além disso, indique um nível de confiança numérico (0 a 100%) para a classificação.

Saída obrigatória: JSON no formato abaixo, sem texto adicional:

{{
  "classification": "<Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherry-picking>",
  "justification": "Texto explicativo curto",
  "confidence": <0-100>
}}
"""

    output = run_ollama(prompt, model=model).strip()

    # 🔹 Extrai apenas o JSON do retorno do modelo
    json_match = re.search(r'\{[\s\S]*\}', output)
    if json_match:
        output = json_match.group(0)

    # 🔹 Interpreta o JSON ou cria fallback
    try:
        parsed = json.loads(output)
        classification = parsed.get("classification", "unverified").strip()
        if classification not in VALID_CLASSES:
            classification = "unverified"
        justification = parsed.get("justification", "").strip()
    except json.JSONDecodeError:
        classification = "unverified"
        justification = output[:300] if output else "Não foi possível interpretar a resposta do modelo."

    # 🔹 Retorno padronizado
    result = {
        "claim": claim,
        "classification": classification,
        "justification": justification,
        "confidence": parsed.get("confidence", 0),
        "timestamp": datetime.utcnow().isoformat()
    }

    print(f"✅ Claim classificada como: {classification.upper()}")
    return result


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

    result = classify_claim(claim, evidences)
    print(json.dumps(result, indent=4, ensure_ascii=False))
