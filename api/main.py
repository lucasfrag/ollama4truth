from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline.question_generation import generate_questions
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Carrega variáveis do .env
load_dotenv()

app = FastAPI(title="Ollama4Truth API", version="0.1")

OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ClaimRequest(BaseModel):
    claim: str

@app.post("/analyze")
async def analyze_claim(request: ClaimRequest):
    """Recebe uma claim e inicia o pipeline."""
    try:
        claim = request.claim.strip()
        if not claim:
            raise HTTPException(status_code=400, detail="Claim vazia!")

        # === Etapa 1: Geração de perguntas ===
        questions = generate_questions(claim)

        # TODO: Etapas seguintes (retrieval, reranking, justification, veracity)
        result = {
            "claim": claim,
            "questions": questions,
            "timestamp": datetime.now().isoformat()
        }

        # Salva resultado parcial
        output_file = OUTPUT_DIR / f"claim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        return {"status": "success", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "✅ Ollama4Truth API está rodando! Envie uma claim via POST /analyze"}
