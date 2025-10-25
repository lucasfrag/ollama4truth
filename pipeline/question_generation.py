import subprocess
import json
import os

def run_ollama(prompt: str, model: str = None) -> str:
    """
    Executa um modelo Ollama localmente e retorna a resposta em texto.
    """
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),      # 🔹 envia prompt codificado
            text=False,                        # 🔹 desativa modo texto
            capture_output=True,
            check=True
        )

        # 🔹 decodifica saída explicitamente como UTF-8
        return result.stdout.decode("utf-8", errors="ignore").strip()

    except subprocess.CalledProcessError as e:
        print("❌ Erro ao executar Ollama:", e.stderr)
        return ""
def generate_questions(claim: str, model: str = None) -> list:
    """
    Gera perguntas investigativas com base na claim.
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

    output = run_ollama(prompt, model=model)

    # Tenta extrair JSON se o modelo gerar texto fora do formato esperado
    try:
        questions = json.loads(output)
        if isinstance(questions, dict):
            questions = list(questions.values())
    except json.JSONDecodeError:
        # fallback: separar por linhas ou traços
        questions = [q.strip("-• ") for q in output.split("\n") if len(q.strip()) > 3]

    return questions
