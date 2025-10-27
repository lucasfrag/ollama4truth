import json
from datetime import datetime
from pipeline.question_generation import generate_questions
from pipeline.evidence_retrieval import retrieve_evidence
# (futuramente) from core.classification import classify_claim

def pipeline_run(claim: str):
    """
    Executa o pipeline completo:
    1. Geração de perguntas
    2. Recuperação de evidências
    3. (futuro) Ranqueamento, justificação e classificação final
    """
    print(f"\n🚀 Iniciando pipeline para a claim:\n   \"{claim}\"\n")

    # === 1️⃣ Gerar perguntas ===
    questions_output = generate_questions(claim)
    questions = [q for q in questions_output.get("questions", []) if isinstance(q, str)]

    print(f"\n✅ {len(questions)} perguntas geradas.")

    # === 2️⃣ Buscar evidências ===
    evidence_output = retrieve_evidence(claim, questions)

    # === 3️⃣ (futuro) Classificação ===
    # result = classify_claim(claim, evidence_output)

    # === Salvar tudo em um JSON final ===
    final_result = {
        "claim": claim,
        "timestamp": datetime.now().isoformat(),
        "questions": questions_output,
        "evidences": evidence_output.get("evidences", []),
        "label": None,  # futuramente: Supported / Refuted / Unclear
    }

    with open("data/results.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    print("\n🎯 Pipeline concluído com sucesso!")
    print("📁 Resultados salvos em: data/results.json")

    return final_result


if __name__ == "__main__":
    example_claim = "O café ajuda a melhorar a memória de longo prazo."
    pipeline_run(example_claim)
