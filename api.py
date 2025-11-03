from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from main import run_pipeline_stream
import uvicorn
import json
import time

app = FastAPI(
    title="Averitec Custom Pipeline API",
    description="API para verificar claims e gerar evidências automaticamente usando Ollama + Google Search",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    claim: str

@app.get("/analyze-stream")
def analyze_stream(claim: str = Query(...)):
    def generate():
        for log, data in run_pipeline_stream(claim):
            yield f"data: {log}\n\n"
            if data:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API está rodando e pronta para receber claims"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
