from vector_store import LocalVectorStore
from wikipedia_ingest import build_local_corpus


def main() -> None:
    store = LocalVectorStore()
    print("Resetting vector store before fresh ingest...")
    store.reset()
    rows = build_local_corpus(verbose=True)
    if not rows:
        print("No chunks were prepared. Retry ingest after a short wait.")
        return
    print("\nEmbedding + upsert to Chroma...")
    store.upsert_rows(rows)
    print(f"Done. Stored {len(rows)} chunks in vector store.")


if __name__ == "__main__":
    main()
