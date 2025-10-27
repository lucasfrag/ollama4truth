# 🧠 Ollama4Truth

**Ollama4Truth** is an open-source framework for **misinformation detection and fact-checking** using **Large Language Models (LLMs)** through the [Ollama](https://ollama.com) ecosystem.

This project aims to explore, evaluate, and democratize the use of open-source LLMs for identifying and mitigating online disinformation — particularly in **Portuguese (PT-BR)** and **English** contexts.

---

## 🎯 Objectives

- Develop a modular and reproducible pipeline for **fact-checking and misinformation identification**.  
- Evaluate open LLMs’ capabilities in **detecting false or misleading content**.  
- Foster **open collaboration** and **transparent evaluation** within the AI research community.

---

## Configuration

1. Create a .env file:
````
OLLAMA_MODEL=llama3.1
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
GOOGLE_CSE_ID=YOUR_CSE_ID
````

2. Install requirements (with Python 3.10):
````
pip install -r requirements.txt
````

## 🚀 How to run?

1. Start the server:
````
uvicorn api:app --reload
````
---

2. Send a POST request to http://localhost:8000/analyze with the claim:
````
{
  "claim": "O café ajuda a melhorar a memória de longo prazo."
}
````

## ⚙️ Technologies

- 🦙 **[Ollama](https://ollama.com)** — local LLM inference  
- 🔍 **Google Search API** — open evidence retrieval  
- 🤗 **Transformers** — tokenization and model loading  
- 🧮 **PyTorch** — inference backend  
- 📄 **BM25 / FAISS** — ranking and document retrieval  
- 🧰 **Python (3.10+)**

---

## 🪶 License

Released under the **MIT License** — free for research and open-source use.

---

