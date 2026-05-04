from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
CHROMA_DIR = DB_DIR / "chroma"
SQLITE_PATH = DB_DIR / "ingestion.db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 120
TOP_K = 5

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
