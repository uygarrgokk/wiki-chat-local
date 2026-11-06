from agentic.agent import ReActAgent
from agentic.hybrid import HybridRetriever
from agentic.llm import OllamaLLM
from agentic.memory import MemoryStore
from generation import generate_answer
from retrieval import retrieve_context
from vector_store import LocalVectorStore
from wikipedia_ingest import build_local_corpus


HELP_TEXT = """
Commands:
  /ingest   Fetch Wikipedia pages, chunk, embed, store
  /reset    Reset vector store
  /sources  Toggle source display
  /think    Toggle display of the model's <thinking> and ReAct trace
  /simple   Toggle legacy single-shot RAG (no agent loop)
  /new      Start a new conversation session (memory summaries persist)
  /memory   Show the memory injected into the prompt
  /help     Show commands
  /quit     Exit
"""


def run() -> None:
    print("Local Wikipedia RAG Assistant")
    print(HELP_TEXT)

    store = LocalVectorStore()
    show_sources = False
    show_thinking = False
    simple_mode = False

    memory = MemoryStore()
    session_id = memory.new_session()
    retriever = HybridRetriever(store)
    agent = ReActAgent(OllamaLLM(), retriever, memory)

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
        if user_input == "/think":
            show_thinking = not show_thinking
            print(f"Show thinking: {show_thinking}")
            continue
        if user_input == "/simple":
            simple_mode = not simple_mode
            print(f"Simple mode: {simple_mode}")
            continue
        if user_input == "/new":
            session_id = memory.new_session()
            print("New session started.")
            continue
        if user_input == "/memory":
            print(memory.build_context(session_id) or "(memory is empty)")
            continue
        if user_input == "/reset":
            store.reset()
            retriever.refresh()
            print("Vector store reset.")
            continue
        if user_input == "/ingest":
            print("Ingesting Wikipedia data...")
            rows = build_local_corpus()
            if not rows:
                print(
                    "Ingestion produced 0 chunks. Likely Wikipedia rate-limit (HTTP 429)."
                )
                print("Wait a minute, then run /ingest again.")
                continue
            store.upsert_rows(rows)
            retriever.refresh()
            print(f"Ingestion completed. Stored {len(rows)} chunks.")
            continue

        if simple_mode:
            try:
                context, sources = retrieve_context(store, user_input)
            except RuntimeError as exc:
                print(f"Assistant> {exc}")
                continue
            print(f"Assistant> {generate_answer(user_input, context)}")
        else:
            try:
                result = agent.run(user_input, session_id)
            except Exception as exc:  # Ollama down, model missing, ...
                print(f"Assistant> Agent error: {exc}")
                continue
            if show_thinking:
                print("\n[Sub-questions]", *result.sub_questions, sep="\n- ")
                for i, step in enumerate(result.steps, 1):
                    print(f"[Step {i}] {step.thought}\n  -> {step.action}({step.action_input})")
                if result.thinking:
                    print(f"\n<thinking>\n{result.thinking}\n</thinking>")
                if result.corrected:
                    print("[answer was revised by self-correction]")
            print(f"Assistant> {result.answer}")
            sources = result.sources

        if show_sources and sources:
            print("\nSources:")
            for src in sources:
                print(f"- {src['title']} ({src['entity_type']}): {src['source_url']}")


if __name__ == "__main__":
    run()
