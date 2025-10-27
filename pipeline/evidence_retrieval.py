import os
import json
import time
import requests
from urllib.parse import quote_plus
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CSE_ID")  # ID do mecanismo de busca customizado

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

def retrieve_evidence(claim: str, questions: list, output_path: str = "data/output/evidence_results.json"):
    """
    Busca evidências online para cada pergunta relacionada à claim.
    """
    all_evidence = {
        "claim": claim,
        "timestamp": datetime.utcnow().isoformat(),
        "evidences": []
    }

    for q in questions:
        print(f"[INFO] Buscando evidências para: {q}")
        results = google_search(q)
        all_evidence["evidences"].append({
            "question": q,
            "results": results
        })
        time.sleep(1.5)  # evita limite de requisições

    # Cria pasta de saída se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva em JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_evidence, f, indent=4, ensure_ascii=False)

    print(f"[SUCESSO] Evidências salvas em {output_path}")
    return all_evidence


if __name__ == "__main__":
    # Exemplo de teste manual
    claim = "O café ajuda a melhorar a memória de longo prazo."
    questions = [
        "O consumo de café melhora a memória de longo prazo?",
        "Há estudos científicos que confirmam essa relação?",
        "A cafeína influencia processos de consolidação de memória?"
    ]
    retrieve_evidence(claim, questions)
