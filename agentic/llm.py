"""Thin Ollama wrapper plus JSON helpers. The agent depends only on `LLM`."""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

import ollama

from config import AGENT_TEMPERATURE, OLLAMA_MODEL, OLLAMA_NUM_CTX


class LLM(Protocol):
    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str: ...


class OllamaLLM:
    def __init__(self, model: str = OLLAMA_MODEL, temperature: float = AGENT_TEMPERATURE):
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json" if json_mode else None,
            options={"temperature": self.temperature, "num_ctx": OLLAMA_NUM_CTX},
        )
        return response["message"]["content"].strip()


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction; small models often wrap or pad JSON."""
    match = _JSON_OBJECT.search(text)
    for candidate in (text, match.group(0) if match else ""):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
