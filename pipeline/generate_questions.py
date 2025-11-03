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
    prompt = f"""
Você é um assistente de checagem de fatos. 
A seguinte afirmação precisa ser verificada:

Claim: "{claim}"

Gere de 3 a 5 perguntas curtas e objetivas que ajudariam a confirmar ou refutar essa claim.

Saída obrigatória: JSON no formato exato abaixo (sem texto explicativo, sem comentários):

{{
  "questions": [
    "Pergunta 1",
    "Pergunta 2",
    "Pergunta 3"
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

    # 🔹 Agora retorna um dict padronizado
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
