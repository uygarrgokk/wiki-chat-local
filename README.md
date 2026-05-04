# BLG483E HW3 - Local Wikipedia RAG Assistant

This project implements a local Retrieval-Augmented Generation (RAG) system that answers questions about famous people and places using only local components.

## Features
- Ingests Wikipedia pages for at least 20 people and 20 places.
- Chunks long texts with overlap for retrieval stability.
- Creates embeddings locally with SentenceTransformers.
- Stores vectors in local Chroma DB.
- Classifies query intent (person/place/both) and filters retrieval.
- Uses local Ollama model to generate grounded answers.
- Returns "I don't know." when context is missing.
- Provides simple chat-style CLI.

## Tech Stack
- Python
- Local LLM: Ollama (`llama3.2:3b` by default)
- Local embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: Chroma (persistent local storage)
- Metadata DB: SQLite

## Setup
1. Create and activate virtual environment:
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Install and run Ollama from [https://ollama.com](https://ollama.com)
4. Pull model:
   - `ollama pull llama3.2:3b`

## Run
1. Start app:
   - `python main.py`
2. In CLI, ingest data:
   - `/ingest`
3. Ask questions.

## CLI Commands
- `/ingest` : Download Wikipedia pages, chunk, embed, store.
- `/reset` : Clear vector store.
- `/sources` : Toggle source display.
- `/help` : Help text.
- `/quit` : Exit.

## Example Queries
- Who was Albert Einstein and what is he known for?
- What did Marie Curie discover?
- Where is the Eiffel Tower located?
- Compare Albert Einstein and Nikola Tesla.
- Compare the Eiffel Tower and the Statue of Liberty.
- Who is the president of Mars? (should return `I don't know`)

## Design Choice (Vector Store Strategy)
This project uses **Option B**: one vector store with metadata field `entity_type` (`person` or `place`).

Rationale:
- Keeps storage and management simple.
- Supports mixed queries naturally with one query path.
- Enables flexible filtering with metadata.

## Project Structure
- `main.py` - CLI interface
- `wikipedia_ingest.py` - Wikipedia fetch + chunk persistence
- `chunking.py` - chunking strategy
- `vector_store.py` - Chroma + embedding logic
- `retrieval.py` - query classification + retrieval
- `generation.py` - Ollama generation
- `entities.py` - people/place seed list
- `config.py` - constants and paths
- `Product_prd.md` - product requirements document
- `recommendation.md` - production deployment recommendations

## Demo Video
- Add your 5-minute demo link here: `https://...`

## Publish to GitHub
1. Initialize repository:
   - `git init`
2. Stage files:
   - `git add .`
3. Commit:
   - `git commit -m "Initial submission: local Wikipedia RAG assistant"`
4. Create GitHub repo and push:
   - `git remote add origin <your-repo-url>`
   - `git branch -M main`
   - `git push -u origin main`

Note: `.gitignore` excludes local virtual environment and generated `db/` + `data/` folders.
