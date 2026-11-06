from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hit:
    id: str
    text: str
    title: str
    entity_type: str
    source_url: str = ""
    score: float = 0.0

    def source(self) -> dict[str, str]:
        return {
            "title": self.title,
            "entity_type": self.entity_type,
            "source_url": self.source_url,
        }


@dataclass
class Step:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    answer: str
    thinking: str = ""
    sub_questions: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    corrected: bool = False
