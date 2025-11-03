import json
from datetime import datetime
from pipeline.generate_questions import generate_questions
from pipeline.retrieve_evidence import retrieve_evidence
from pipeline.classification import classify_claim

import json
from datetime import datetime
from pipeline.generate_questions import generate_questions
from pipeline.retrieve_evidence import retrieve_evidence
from pipeline.classification import classify_claim

def run_pipeline(claim: str):
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

    # === 3️⃣ Classificação ===
    classification_output = classify_claim(claim, evidence_output.get("evidences", []))

    # === Salvar tudo em um JSON final ===
    final_result = {
        "claim": claim,
        "timestamp": datetime.now().isoformat(),
        "questions": questions_output,
        "evidences": evidence_output.get("evidences", []),
        "label": classification_output.get("classification"),
        "rationale": classification_output.get("justification"),
        "confidence": classification_output.get("confidence")
    }

    with open("data/results.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    print("\n🎯 Pipeline concluído com sucesso!")
    print("📁 Resultados salvos em: data/results.json")

    return final_result



def run_pipeline_stream(claim: str):
    """
    Executa o pipeline passo a passo, emitindo logs via yield.
    Ideal para streaming em tempo real.
    """
    yield f"🚀 Iniciando pipeline para a claim: \"{claim}\"", None

    # === 1️⃣ Gerar perguntas ===
    yield "🧩 Gerando perguntas...", None
    questions_output = generate_questions(claim)
    questions = [q for q in questions_output.get("questions", []) if isinstance(q, str)]
    yield f"✅ {len(questions)} perguntas geradas.", None

    # === 2️⃣ Buscar evidências ===
    yield "🔍 Buscando evidências...", None
    evidence_output = retrieve_evidence(claim, questions)
    yield f"✅ {len(evidence_output.get('evidences', []))} evidências encontradas.", None

    # === 3️⃣ Classificação ===
    yield "🧠 Classificando claim...", None
    classification_output = classify_claim(claim, evidence_output.get("evidences", []))
    yield f"✅ Classificação concluída: {classification_output.get('classification')}", None

    # === Salvar tudo em um JSON final ===
    final_result = {
        "claim": claim,
        "timestamp": datetime.now().isoformat(),
        "questions": questions_output,
        "evidences": evidence_output.get("evidences", []),
        "label": classification_output.get("classification"),
        "rationale": classification_output.get("justification"),
        "confidence": classification_output.get("confidence")
    }

    with open("data/results.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    yield "🎯 Pipeline concluído com sucesso!", final_result
    return final_result


if __name__ == "__main__":
    example_claim = "O café ajuda a melhorar a memória de longo prazo."
    run_pipeline(example_claim)
