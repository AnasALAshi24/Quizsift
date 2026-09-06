from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .models import TermDefinition


SYSTEM_PROMPT = """You extract study glossary entries from educational source text.
Identify only meaningful domain concepts that the supplied text actually defines or clearly explains.
Do not invent facts, do not turn headings into terms unless they are explained, and do not copy examples as definitions.
Each description must be a concise, self-contained explanation that can be used as a quiz question.
Return JSON only."""

QUESTION_SYSTEM_PROMPT = """You write concise educational multiple-choice question prompts.
For every supplied term and description, write one WH-style question whose exact answer is the term.
Never put the answer term in the question. Do not ask for an explanation, reason, or procedure when the answer would no longer be the term.
Use the source language. For English, begin naturally with What, Which, Who, When, Where, or How.
Return JSON only."""


class OllamaError(RuntimeError):
    pass


def normalize_host(host: str) -> str:
    normalized = host.strip().rstrip("/")
    if normalized.endswith("/api"):
        normalized = normalized[:-4]
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OllamaError("Enter a valid Ollama host, such as http://localhost:11434.")
    return normalized


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def _read_json(request: Request, *, timeout: int) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            detail = body
        if exc.code == 401:
            raise OllamaError(
                "Ollama Cloud rejected the API key. It may be invalid or revoked."
            ) from exc
        if exc.code == 403:
            raise OllamaError(
                "The API key is valid, but this account cannot use the requested model or action. "
                "Test the connection again and choose a model from the available-model list."
            ) from exc
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {str(detail)[:400]}") from exc
    except URLError as exc:
        raise OllamaError(f"Could not reach Ollama: {exc.reason}") from exc
    except TimeoutError as exc:
        raise OllamaError("Ollama did not respond before the request timed out.") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned an invalid server response.") from exc


def list_models(host: str, api_key: str | None = None, *, timeout: int = 20) -> list[str]:
    request = Request(
        f"{normalize_host(host)}/api/tags",
        headers=_headers(api_key),
        method="GET",
    )
    result = _read_json(request, timeout=timeout)
    return [str(model["name"]) for model in result.get("models", []) if model.get("name")]


def _parse_json_object(raw_content: str) -> dict[str, Any]:
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        first = content.find("{")
        last = content.rfind("}")
        if first < 0 or last <= first:
            raise OllamaError("The model response was not valid JSON. Try a stronger model.")
        try:
            payload = json.loads(content[first : last + 1])
        except json.JSONDecodeError as exc:
            raise OllamaError("The model response was not valid JSON. Try a stronger model.") from exc

    if not isinstance(payload, dict):
        raise OllamaError("The model response was not a JSON object.")
    return payload


def _chat_content(
    *,
    host: str,
    model: str,
    api_key: str | None,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
) -> str:
    model_name = model.strip()
    if not model_name:
        raise OllamaError("Enter an Ollama model name.")

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    # Ollama Cloud currently relies on prompt-enforced JSON; local models can
    # additionally use JSON mode. Thinking is minimized for extraction.
    is_cloud_model = model_name.endswith(":cloud") or model_name.endswith("-cloud")
    is_direct_cloud = normalize_host(host) == "https://ollama.com"
    if not (is_cloud_model or is_direct_cloud):
        payload["format"] = "json"
    payload["think"] = "low" if model_name.startswith("gpt-oss") else False

    request = Request(
        f"{normalize_host(host)}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(api_key),
        method="POST",
    )
    response = _read_json(request, timeout=timeout)
    message = response.get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if not content:
        raise OllamaError("Ollama returned no answer content.")
    return str(content)


def _parse_terms(raw_content: str) -> list[TermDefinition]:
    payload = _parse_json_object(raw_content)

    raw_terms = payload.get("terms", [])
    if not isinstance(raw_terms, list):
        raise OllamaError("The model response did not contain a terms list.")

    terms: list[TermDefinition] = []
    for item in raw_terms:
        if not isinstance(item, dict):
            continue
        term = " ".join(str(item.get("term", "")).split()).strip(" :")
        description = " ".join(str(item.get("description", "")).split()).strip()
        page_value = item.get("source_page")
        try:
            source_page = int(page_value) if page_value is not None else None
        except (TypeError, ValueError):
            source_page = None
        if term and description and len(description.split()) >= 4:
            terms.append(TermDefinition(term, description, source_page))
    return terms


def _parse_questions(raw_content: str) -> dict[str, str]:
    payload = _parse_json_object(raw_content)
    raw_questions = payload.get("questions", [])
    if not isinstance(raw_questions, list):
        raise OllamaError("The model response did not contain a questions list.")

    questions: dict[str, str] = {}
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        term = " ".join(str(item.get("term", "")).split()).strip(" :")
        question = " ".join(str(item.get("question", "")).split()).strip()
        term_words = [word for word in "".join(
            character.casefold() if character.isalnum() else " " for character in term
        ).split() if word]
        question_words = [word for word in "".join(
            character.casefold() if character.isalnum() else " " for character in question
        ).split() if word]
        contains_answer = any(
            question_words[index : index + len(term_words)] == term_words
            for index in range(len(question_words) - len(term_words) + 1)
        ) if term_words else False
        if term and question and not contains_answer:
            if not question.endswith("?"):
                question += "?"
            questions[term.casefold()] = question
    return questions


def extract_terms_from_chunks(
    chunks: list[str],
    *,
    host: str,
    model: str,
    api_key: str | None = None,
    terms_per_chunk: int = 18,
    timeout: int = 180,
    progress: Callable[[int, int], None] | None = None,
) -> list[TermDefinition]:
    if not chunks:
        return []
    extracted: list[TermDefinition] = []

    for index, chunk in enumerate(chunks, start=1):
        prompt = f"""Extract up to {terms_per_chunk} high-quality term-definition pairs from the source below.

Output exactly one JSON object with this shape:
{{
  "terms": [
    {{"term": "the concept name", "description": "a clear standalone description", "source_page": 1}}
  ]
}}

The page number must come from the nearest [PAGE N] marker. Preserve the source language.
The rendered glossary format will be: term: description

SOURCE:
{chunk}"""
        content = _chat_content(
            host=host,
            model=model,
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            timeout=timeout,
        )
        extracted.extend(_parse_terms(content))
        if progress:
            progress(index, len(chunks))

    # Keep the first occurrence of a term, case-insensitively. The model—not
    # punctuation or capitalization rules—decides what counts as a definition.
    unique: dict[str, TermDefinition] = {}
    for entry in extracted:
        key = entry.term.casefold()
        if key not in unique:
            unique[key] = entry
    return list(unique.values())


def generate_wh_questions(
    entries: list[TermDefinition],
    *,
    host: str,
    model: str,
    api_key: str | None = None,
    batch_size: int = 20,
    timeout: int = 180,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, str]:
    """Generate term-answerable WH prompts, keyed by the canonical term."""
    if not entries:
        return {}

    batches = [entries[index : index + batch_size] for index in range(0, len(entries), batch_size)]
    generated: dict[str, str] = {}

    for batch_index, batch in enumerate(batches, start=1):
        supplied = [
            {"term": entry.term, "description": entry.description}
            for entry in batch
        ]
        prompt = f"""Write exactly one WH-style question for each supplied item.

Rules:
- The correct answer must be the exact value of "term".
- The question must not contain that term or a trivial variation of it.
- Test the meaning in "description" without adding unsupported facts.
- Keep the question concise and suitable for four term choices.
- Prefer natural forms such as "What process... ?" or "Which mechanism... ?".
- Return every supplied term exactly once.

Output exactly one JSON object with this shape:
{{
  "questions": [
    {{"term": "the unchanged supplied term", "question": "the WH-style question?"}}
  ]
}}

ITEMS:
{json.dumps(supplied, ensure_ascii=False)}"""
        content = _chat_content(
            host=host,
            model=model,
            api_key=api_key,
            system_prompt=QUESTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            timeout=timeout,
        )
        parsed = _parse_questions(content)
        canonical = {entry.term.casefold(): entry.term for entry in batch}
        for key, question in parsed.items():
            if key in canonical:
                generated[canonical[key]] = question
        if progress:
            progress(batch_index, len(batches))

    if not generated:
        raise OllamaError("Ollama did not return any usable WH questions.")
    return generated
