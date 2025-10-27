from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.main import pipeline_run
import uvicorn

app = FastAPI(
    title="Averitec Custom Pipeline API",
    description="API para verificar claims e gerar evidências automaticamente usando Ollama + Google Search",
    version="1.0.0",
)

# Modelo de entrada
class ClaimRequest(BaseModel):
    claim: str

# Endpoint principal
@app.post("/analyze")
def analyze_claim(request: ClaimRequest):
    try:
        print(f"\n📩 Recebida claim: {request.claim}\n")
        result = pipeline_run(request.claim)
        return result
    except Exception as e:
        print(f"[ERRO] Falha ao processar claim: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint simples de saúde
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API está rodando e pronta para receber claims"}

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
