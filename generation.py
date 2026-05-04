import ollama

from config import OLLAMA_MODEL


SYSTEM_PROMPT = """You are a careful QA assistant for a local Wikipedia RAG system.
Rules:
1) Answer only from provided context.
2) If context is insufficient, say exactly: I don't know.
3) Keep answer concise and factual.
"""


def generate_answer(query: str, context: str) -> str:
    if not context.strip():
        return "I don't know"

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer using only context."
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )
    except Exception:
        return "I don't know"

    answer = response["message"]["content"].strip()
    return answer or "I don't know"
