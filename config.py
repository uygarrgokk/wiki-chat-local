from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
CHROMA_DIR = DB_DIR / "chroma"
SQLITE_PATH = DB_DIR / "ingestion.db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:3b"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 120
TOP_K = 5

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia requires a descriptive User-Agent; requests without it often get HTTP 403.
WIKIPEDIA_HEADERS = {
    "User-Agent": "BLG483E-LocalWikipediaRAG/1.0 (course homework; Python requests)",
    "Accept": "application/json",
}

# --- Agentic RAG -------------------------------------------------------------
MEMORY_DB_PATH = DB_DIR / "memory.db"

# Hybrid retrieval
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # or "BAAI/bge-reranker-base"
USE_RERANKER = True
CANDIDATES_PER_RETRIEVER = 20  # fetched from BM25 and from Chroma before fusion
RRF_K = 60  # standard RRF damping constant
RERANK_POOL = 15  # fused candidates passed to the cross-encoder
FINAL_TOP_K = 4  # chunks returned to the agent per search

# ReAct agent
AGENT_MAX_STEPS = 5
MAX_SUB_QUESTIONS = 3
AGENT_TEMPERATURE = 0.1
OLLAMA_NUM_CTX = 4096
ENABLE_SELF_CORRECTION = True
OBSERVATION_CHAR_LIMIT = 600  # per chunk shown in an observation

# Memory
RECENT_MESSAGES = 6  # raw messages kept verbatim in the prompt
SUMMARY_TRIGGER = 10  # unsummarized messages that trigger compression
