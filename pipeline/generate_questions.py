import subprocess
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def run_ollama(prompt: str, model: str = None) -> str:
    """
    Executa um modelo Ollama localmente e retorna a resposta em texto.
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

def generate_questions(claim: str, model: str = None) -> list:
    """
    Gera perguntas investigativas com base na claim e retorna uma lista de perguntas.
    """
    prompt = f"""You are a question generator for an academic fact-checking system.
A user submitted the following text for verification:

INPUT TEXT: "{claim}"

Generate exactly 3 to 5 neutral, investigative questions in Portuguese (PT-BR) that would help verify or refute this text.
Each question must:
- End with a question mark (?)
- Be neutral (do NOT assume the answer)
- Be specific and relevant to the claim

Example for "Terra é plana":
{{
  "questions": [
    "Qual é o formato da Terra segundo a comunidade científica?",
    "Existem evidências científicas que comprovem que a Terra é esférica?",
    "Quais são as principais alegações do movimento terraplanista?"
  ]
}}

Now generate questions for the INPUT TEXT above. Return ONLY valid JSON:

{{
  "questions": [
    "pergunta 1?",
    "pergunta 2?",
    "pergunta 3?"
  ]
}}
"""

    output = run_ollama(prompt, model=model).strip()
    
    # Tentar extrair apenas o bloco JSON
    json_match = re.search(r'\{[\s\S]*\}', output)
    if json_match:
        output = json_match.group(0)

    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict) and "questions" in parsed:
            questions = parsed["questions"]
        elif isinstance(parsed, list):
            questions = parsed
        else:
            questions = list(parsed.values())
    except json.JSONDecodeError:
        questions = [q.strip("-• ") for q in output.split("\n") if len(q.strip()) > 3]

    # Filter out refusals and non-question strings
    questions = [q for q in questions if isinstance(q, str) and len(q) > 5 and "não posso" not in q.lower()]

    # Fallback: if no valid questions generated, use the claim itself
    if not questions:
        print("[WARN] Question generation failed — using claim as fallback query")
        questions = [claim]

    return {
        "claim": claim,
        "questions": questions,
        "timestamp": datetime.utcnow().isoformat()
    }

# 🔹 o bloco abaixo é só para testes manuais — não interfere no pipeline
if __name__ == "__main__":
    claim = "O café ajuda a melhorar a memória de longo prazo."
    questions = generate_questions(claim)
    print(json.dumps({"claim": claim, "questions": questions}, indent=4, ensure_ascii=False))
