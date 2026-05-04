# Recommendation for Production Deployment

This homework project is local-first. For production, the following architecture is recommended.

## 1. Serving Architecture
- API service layer (FastAPI) for chat and ingestion endpoints.
- Background worker queue for ingestion and embedding jobs.
- Dedicated vector DB service (managed or self-hosted with backups).
- LLM serving node(s) with autoscaling.

## 2. Model Strategy
- Keep a small default model for low-latency traffic.
- Add larger model route for complex questions.
- Use model fallback chain when generation fails.
- Version model and prompt templates together.

## 3. Retrieval Quality Improvements
- Add hybrid retrieval (BM25 + vector).
- Add re-ranking model for top-k chunks.
- Maintain entity alias dictionary (e.g., "CR7" -> Cristiano Ronaldo).
- Track retrieval precision with benchmark query set.

## 4. Data Engineering
- Use scheduled ingestion pipelines.
- Store source snapshots and content hashes.
- Add chunk deduplication and language filtering.
- Keep full provenance metadata for each answer.

## 5. Reliability and Security
- Add structured logging and tracing.
- Enforce rate limiting and auth for API users.
- Encrypt data at rest and in transit.
- Add safety filters for harmful outputs.

## 6. Evaluation and Monitoring
- Offline eval set with expected answer checks.
- Online metrics: latency, unknown-rate, citation-rate.
- Alerting for retrieval or model drift.
- Human review loop for failed queries.

## 7. Cost and Performance Tradeoffs
- Smaller models reduce cost/latency but reduce quality.
- Higher top-k improves recall but increases token load.
- Rich chunk overlap improves context continuity but increases storage.
- Hybrid retrieval improves quality but adds complexity.

## 8. Practical Next Steps
1. Wrap CLI logic into REST API.
2. Add automated test suite and CI.
3. Add benchmark dataset and eval script.
4. Containerize with Docker + compose for reproducible deployment.
