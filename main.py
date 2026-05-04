from generation import generate_answer
from retrieval import retrieve_context
from vector_store import LocalVectorStore
from wikipedia_ingest import build_local_corpus


HELP_TEXT = """
Commands:
  /ingest   Fetch Wikipedia pages, chunk, embed, store
  /reset    Reset vector store
  /sources  Toggle source display
  /help     Show commands
  /quit     Exit
"""


def run() -> None:
    print("Local Wikipedia RAG Assistant")
    print(HELP_TEXT)

    store = LocalVectorStore()
    show_sources = False

    while True:
        user_input = input("\nYou> ").strip()
        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye.")
            break
        if user_input == "/help":
            print(HELP_TEXT)
            continue
        if user_input == "/sources":
            show_sources = not show_sources
            print(f"Show sources: {show_sources}")
            continue
        if user_input == "/reset":
            store.reset()
            print("Vector store reset.")
            continue
        if user_input == "/ingest":
            print("Ingesting Wikipedia data...")
            rows = build_local_corpus()
            store.upsert_rows(rows)
            print(f"Ingestion completed. Stored {len(rows)} chunks.")
            continue

        context, sources = retrieve_context(store, user_input)
        answer = generate_answer(user_input, context)
        print(f"Assistant> {answer}")

        if show_sources and sources:
            print("\nSources:")
            for src in sources:
                print(f"- {src['title']} ({src['entity_type']}): {src['source_url']}")


if __name__ == "__main__":
    run()
