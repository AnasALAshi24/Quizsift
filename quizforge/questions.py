from __future__ import annotations

import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import QuizQuestion, TermDefinition


RANK_LABELS = {
    1: "Rank 1 · hardest distractors",
    2: "Rank 2 · challenging",
    3: "Rank 3 · balanced",
    4: "Rank 4 · accessible",
}

RANK_OFFSETS = {
    1: (0, 1, 2),
    2: (0, 1, 3),
    3: (1, 3, 4),
    4: (3, 4, 5),
}


def select_question_entries(
    entries: list[TermDefinition], *, question_count: int, seed: int
) -> list[TermDefinition]:
    selected = entries.copy()
    random.Random(seed).shuffle(selected)
    return selected[: min(question_count, len(selected))]


def similarity_neighbors(entries: list[TermDefinition]) -> dict[str, list[str]]:
    if len(entries) < 2:
        return {entry.term: [] for entry in entries}

    descriptions = [entry.description for entry in entries]
    try:
        vectors = TfidfVectorizer(stop_words="english").fit_transform(descriptions)
    except ValueError:
        vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(
            descriptions
        )
    similarities = cosine_similarity(vectors)
    terms = [entry.term for entry in entries]
    neighbors: dict[str, list[str]] = {}

    for index, term in enumerate(terms):
        ordered = similarities[index].argsort()[::-1]
        neighbors[term] = [terms[item] for item in ordered if item != index]
    return neighbors


def _ranked_distractors(
    answer: str,
    neighbors: list[str],
    all_terms: list[str],
    rank: int,
) -> list[str]:
    selected: list[str] = []
    for offset in RANK_OFFSETS[rank]:
        if offset < len(neighbors) and neighbors[offset] not in selected:
            selected.append(neighbors[offset])

    for candidate in neighbors + all_terms:
        if len(selected) >= 3:
            break
        if candidate != answer and candidate not in selected:
            selected.append(candidate)
    return selected[:3]


def build_paper(
    entries: list[TermDefinition],
    *,
    rank: int,
    question_count: int,
    seed: int = 42,
    question_prompts: dict[str, str] | None = None,
) -> list[QuizQuestion]:
    if rank not in RANK_OFFSETS:
        raise ValueError("Rank must be between 1 and 4.")
    if len(entries) < 4:
        raise ValueError("At least four term-definition pairs are required.")

    choice_rng = random.Random(seed + rank)
    selected_entries = select_question_entries(
        entries, question_count=question_count, seed=seed
    )

    neighbors = similarity_neighbors(entries)
    all_terms = [entry.term for entry in entries]
    questions: list[QuizQuestion] = []

    for number, entry in enumerate(selected_entries, start=1):
        distractors = _ranked_distractors(entry.term, neighbors[entry.term], all_terms, rank)
        choices = [entry.term, *distractors]
        choice_rng.shuffle(choices)
        prompt = (question_prompts or {}).get(
            entry.term,
            f"Which term matches this description?\n{entry.description}",
        )
        questions.append(
            QuizQuestion(
                number=number,
                prompt=prompt,
                answer=entry.term,
                choices=tuple(choices),
                source_page=entry.source_page,
            )
        )
    return questions


def build_all_papers(
    entries: list[TermDefinition],
    *,
    question_count: int,
    seed: int = 42,
    question_prompts: dict[str, str] | None = None,
) -> dict[int, list[QuizQuestion]]:
    return {
        rank: build_paper(
            entries,
            rank=rank,
            question_count=question_count,
            seed=seed,
            question_prompts=question_prompts,
        )
        for rank in RANK_OFFSETS
    }
