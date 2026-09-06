from __future__ import annotations

import json

import pytest

from quizforge.models import TermDefinition
from quizforge.ollama_extractor import _parse_questions, _parse_terms, normalize_host
from quizforge.pdf_text import PageText, chunk_pages
from quizforge.questions import build_all_papers


def sample_entries() -> list[TermDefinition]:
    return [
        TermDefinition("Encryption", "Transforms readable plaintext into protected ciphertext.", 1),
        TermDefinition("Decryption", "Transforms protected ciphertext back into readable plaintext.", 1),
        TermDefinition("Hash function", "Maps arbitrary input to a fixed length digest.", 2),
        TermDefinition("Digital signature", "Provides authenticity and integrity using asymmetric keys.", 2),
        TermDefinition("Nonce", "A value used once to prevent replay of a message.", 3),
        TermDefinition("Session key", "A temporary secret key used for one communication session.", 3),
        TermDefinition("Certificate", "Binds a public key to an identity through an issuer.", 4),
    ]


def test_parse_terms_and_render_colon_format() -> None:
    raw = json.dumps(
        {
            "terms": [
                {
                    "term": "Nonce",
                    "description": "A value used only once in a protocol exchange.",
                    "source_page": 7,
                }
            ]
        }
    )
    entries = _parse_terms(raw)
    assert entries[0].display == "Nonce: A value used only once in a protocol exchange."
    assert entries[0].source_page == 7


def test_normalize_host() -> None:
    assert normalize_host("http://localhost:11434/api/") == "http://localhost:11434"
    with pytest.raises(Exception):
        normalize_host("localhost:11434")


def test_parse_wh_questions() -> None:
    raw = json.dumps(
        {
            "questions": [
                {
                    "term": "Encryption",
                    "question": "What process converts plaintext into ciphertext",
                },
                {
                    "term": "Nonce",
                    "question": "What is a nonce?",
                },
            ]
        }
    )
    assert _parse_questions(raw) == {
        "encryption": "What process converts plaintext into ciphertext?"
    }


def test_chunk_pages_keeps_page_markers() -> None:
    chunks = chunk_pages([PageText(1, "alpha"), PageText(2, "beta")], max_characters=100)
    assert len(chunks) == 1
    assert "[PAGE 1]" in chunks[0]
    assert "[PAGE 2]" in chunks[0]


def test_four_papers_have_valid_choices() -> None:
    entries = sample_entries()
    prompts = {
        entry.term: f"Which concept is described by item {index}?"
        for index, entry in enumerate(entries)
    }
    papers = build_all_papers(entries, question_count=5, seed=10, question_prompts=prompts)
    assert set(papers) == {1, 2, 3, 4}
    expected_answers = [question.answer for question in papers[1]]
    for paper in papers.values():
        assert len(paper) == 5
        assert [question.answer for question in paper] == expected_answers
        for question in paper:
            assert len(question.choices) == 4
            assert len(set(question.choices)) == 4
            assert question.answer in question.choices
            assert question.prompt == prompts[question.answer]
