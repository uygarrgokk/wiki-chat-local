from vector_store import LocalVectorStore
from wikipedia_ingest import build_local_corpus


def main() -> None:
    store = LocalVectorStore()
    rows = build_local_corpus(verbose=True)
    print("\nEmbedding + upsert to Chroma...")
    store.upsert_rows(rows)
    print(f"Done. Stored {len(rows)} chunks in vector store.")


if __name__ == "__main__":
    main()
