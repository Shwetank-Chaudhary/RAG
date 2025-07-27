# 📚 RAG-based Document Search API with Qdrant & LangChain

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **Qdrant**, and **HuggingFace embeddings**. It consists of two key components:

- `ingest.py` – Ingests and indexes PDF documents into Qdrant vector database.
- `rag.py` – A FastAPI application that provides a `/search` endpoint to query indexed documents based on semantic similarity.

---

## 🧱 Project Structure
- rag.py # FastAPI server for search via LangChain + Qdrant
- ingest.py # Ingests a PDF, splits, embeds, and indexes content
- data.pdf # Sample document for ingestion (not included)
- .env # Environment variables (optional for advanced config)
- template.py # (Optional) Prompt template (commented out in code)


---

## 🚀 Features

- Embeds documents using `BAAI/bge-large-en` model (via HuggingFace).
- Stores and retrieves vectors from a local Qdrant instance.
- API endpoint to search top-k similar document chunks.
- Modular and extensible using LangChain’s components.
- Designed for local or API-based deployment.

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2. Install requirements
```bash
pip install -r requirements.txt
```
### 3. Setup Docker in you Local system
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```


