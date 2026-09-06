from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TermDefinition:
    term: str
    description: str
    source_page: int | None = None

    @property
    def display(self) -> str:
        return f"{self.term}: {self.description}"


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    number: int
    prompt: str
    answer: str
    choices: tuple[str, ...]
    source_page: int | None = None

