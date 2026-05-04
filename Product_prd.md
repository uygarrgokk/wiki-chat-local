# Product PRD - Local Wikipedia RAG Assistant

## 1. Product Goal
Build a local ChatGPT-style assistant that answers questions about famous people and places using Wikipedia-derived knowledge, without any external LLM API.

## 2. Scope
### In Scope
- Local ingestion from Wikipedia pages.
- Document chunking for large text handling.
- Local embedding generation.
- Local vector storage and retrieval.
- Query routing by entity type (person/place/both).
- Local answer generation with Ollama.
- CLI chat interface.

### Out of Scope
- Web-scale crawling
- Fine-tuning LLMs
- Multi-user authentication
- Cloud deployment automation

## 3. Functional Requirements
1. System ingests at least 20 people and 20 places.
2. Must include minimum mandatory entities listed in homework.
3. Text is split into chunks with overlap.
4. Embeddings generated locally (no external API).
5. Vectors stored in Chroma locally.
6. Query classifier determines person/place/both.
7. Retrieval uses metadata filtering.
8. LLM answer is context-grounded.
9. If answer not supported by context, return "I don't know."
10. CLI supports ask, receive answer, optionally show sources, reset.

## 4. Non-Functional Requirements
- Runs fully on localhost.
- Easy setup with README instructions.
- Reproducible ingestion flow.
- Reasonable latency for demo usage.

## 5. Data Flow
1. User runs `/ingest`.
2. App fetches raw text from Wikipedia API.
3. Text chunking module generates chunk windows.
4. Embedding model converts chunk text to vectors.
5. Vectors + metadata stored in Chroma.
6. User asks question.
7. Query classifier sets retrieval filter.
8. Retriever returns top-k relevant chunks.
9. Prompt builder combines query + chunks.
10. Ollama model generates final answer.

## 6. Architecture Decisions
- **Single vector store + metadata** selected for simplicity and mixed-query support.
- **SentenceTransformer** selected for local, stable embeddings.
- **Ollama** selected to enforce local generation and reproducibility.
- **SQLite** added for transparent ingestion/chunk metadata persistence.

## 7. Risks and Mitigations
- **Risk:** Weak retrieval on ambiguous questions.
  - **Mitigation:** overlap chunking, higher top-k, metadata filtering.
- **Risk:** Hallucination.
  - **Mitigation:** strict system prompt and unknown fallback.
- **Risk:** Local hardware limitations.
  - **Mitigation:** small model (`llama3.2:3b`) and lightweight embed model.

## 8. Success Criteria
- Mandatory entity coverage achieved.
- Required sample queries return relevant answers.
- Failure-case queries return unknown behavior.
- Full project runs via README-only setup.
