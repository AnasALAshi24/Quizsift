from __future__ import annotations

from html import escape
import os
from typing import Any

import pandas as pd
import streamlit as st

from quizforge.exports import glossary_to_csv, paper_to_docx, papers_to_zip
from quizforge.models import TermDefinition
from quizforge.ollama_extractor import (
    OllamaError,
    extract_terms_from_chunks,
    generate_wh_questions,
    list_models,
)
from quizforge.pdf_text import PdfTextError, PageText, chunk_pages, extract_pdf_pages
from quizforge.questions import RANK_LABELS, build_all_papers, select_question_entries


st.set_page_config(page_title="QuizSift", page_icon="✦", layout="wide")


def secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def initialize_state() -> None:
    st.session_state.setdefault("ollama_provider", "Ollama Cloud")
    st.session_state.setdefault("cloud_host", "https://ollama.com")
    st.session_state.setdefault("local_host", "http://localhost:11434")
    st.session_state.setdefault("cloud_model", "gpt-oss:20b")
    st.session_state.setdefault("local_model", "qwen3:8b")
    st.session_state.setdefault("cloud_api_key", secret("OLLAMA_API_KEY"))
    st.session_state.setdefault("local_api_key", "")
    st.session_state.setdefault("connection_models", {})
    st.session_state.setdefault("connection_notice", None)


def preferred_model(models: list[str]) -> str:
    preferred = ("gpt-oss:20b", "gpt-oss:120b", "qwen3.5:397b", "deepseek-v4-pro")
    return next((name for name in preferred if name in models), models[0])


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f2f7fc; }
        .block-container { max-width: 1120px; padding-top: 1.4rem; padding-bottom: 4rem; }
        [data-testid="stHeader"] { background: rgba(242, 247, 252, .9); }
        .brand-row { display:flex; align-items:center; margin-top:.3rem; margin-bottom:.8rem; }
        .brand { font-size:1.45rem; font-weight:800; color:#11285f; letter-spacing:-.03em; }
        .brand span { color:#28bd9f; }
        .section-label { color:#263653; font-size:.83rem; font-weight:750; margin:.3rem 0; }
        .term-line { border-left:3px solid #28bd9f; padding:.65rem .85rem; margin:.45rem 0; background:#f7fbff; border-radius:0 8px 8px 0; }
        .term-line strong { color:#132452; }
        .question-card { background:white; border:1px solid #e3ebf5; border-radius:13px; padding:1rem 1.1rem; margin:.7rem 0; }
        .question-number { color:#28a98f; font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }
        .source-note { color:#8a96a9; font-size:.75rem; }
        div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button { border-radius:9px; min-height:2.7rem; font-weight:700; }
        div[data-testid="stButton"] button[kind="primary"] { background:#22b99a; border-color:#22b99a; }
        [data-testid="stFileUploaderDropzone"] { background:#f7fbff; border:1px dashed #9fb4cf; border-radius:12px; }
        .stTabs [data-baseweb="tab-list"] { gap:.35rem; }
        .stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0; padding:0 1.2rem; }
        .stTabs [aria-selected="true"] { background:#10265e; color:white; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def terms_to_frame(entries: list[TermDefinition]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "term": item.term,
                "description": item.description,
                "source_page": item.source_page,
            }
            for item in entries
        ]
    )


def frame_to_terms(frame: pd.DataFrame) -> list[TermDefinition]:
    entries: list[TermDefinition] = []
    seen: set[str] = set()
    for row in frame.to_dict("records"):
        term = " ".join(str(row.get("term", "")).split()).strip(" :")
        description = " ".join(str(row.get("description", "")).split()).strip()
        page_value: Any = row.get("source_page")
        source_page = None if pd.isna(page_value) else int(page_value)
        key = term.casefold()
        if term and description and key not in seen:
            entries.append(TermDefinition(term, description, source_page))
            seen.add(key)
    return entries


def source_pages(uploaded_file: Any, pasted_text: str, max_pages: int) -> list[PageText]:
    if uploaded_file is not None:
        return extract_pdf_pages(uploaded_file.getvalue(), max_pages=max_pages)
    if pasted_text.strip():
        return [PageText(page_number=1, text=pasted_text.strip())]
    raise PdfTextError("Upload a PDF or paste source text first.")


initialize_state()
apply_styles()

st.markdown(
    """
    <div class="brand-row">
      <div class="brand">Quiz<span>Sift</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

with st.container(border=True):
    st.markdown('<div class="section-label">1 · Connect Ollama</div>', unsafe_allow_html=True)
    provider = st.segmented_control(
        "Provider",
        ["Ollama Cloud", "Local Ollama"],
        key="ollama_provider",
        required=True,
        label_visibility="collapsed",
    )
    is_cloud = provider == "Ollama Cloud"
    host_key = "cloud_host" if is_cloud else "local_host"
    model_key = "cloud_model" if is_cloud else "local_model"
    pending_model_key = f"pending_{model_key}"
    if pending_model_key in st.session_state:
        st.session_state[model_key] = st.session_state.pop(pending_model_key)

    config_columns = st.columns([1.25, 1, 1])
    with config_columns[0]:
        host = st.text_input("Ollama host", key=host_key)

    connection_key = f"{provider}|{host.strip().rstrip('/')}"
    known_models = st.session_state["connection_models"].get(connection_key, [])
    with config_columns[1]:
        current_model = st.session_state[model_key]
        model_options = known_models or [current_model]
        if current_model not in model_options:
            model_options = [current_model, *model_options]
        model = st.selectbox(
            "Model",
            model_options,
            key=model_key,
            help="Click Test connection to load the models available to this API key.",
        )
    with config_columns[2]:
        api_key = st.text_input(
            "API key",
            key="cloud_api_key" if is_cloud else "local_api_key",
            type="password",
            disabled=not is_cloud,
            help="Used only for this session and never written to the project.",
        )

    if st.button("Test connection and load models", width="content"):
        try:
            models = list_models(host, api_key or None)
            if models:
                st.session_state["connection_models"][connection_key] = models
                selected_model = (
                    model
                    if known_models and model in models
                    else preferred_model(models)
                )
                if selected_model != model:
                    st.session_state[pending_model_key] = selected_model
                st.session_state["connection_notice"] = (
                    f"Connected. {len(models)} model(s) available. Selected {selected_model}."
                )
                st.rerun()
            else:
                st.warning("Connected, but no models were returned. Pull or select a model first.")
        except OllamaError as exc:
            st.error(str(exc))

    connection_notice = st.session_state.pop("connection_notice", None)
    if connection_notice:
        st.success(connection_notice)

with st.container(border=True):
    st.markdown('<div class="section-label">2 · Add your content</div>', unsafe_allow_html=True)
    upload_tab, paste_tab = st.tabs(["Upload PDF", "Type or paste text"])
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Drop a PDF here",
            type=["pdf"],
            help="Text-based PDFs work directly. Scanned image PDFs require OCR.",
        )
    with paste_tab:
        pasted_text = st.text_area(
            "Source text",
            height=190,
            placeholder="Paste textbook or lecture content here…",
        )

    with st.expander("Extraction settings"):
        setting_columns = st.columns(3)
        with setting_columns[0]:
            max_pages = st.number_input("Maximum PDF pages", 1, 500, 40)
        with setting_columns[1]:
            chunk_size = st.number_input("Characters per Ollama request", 3000, 30000, 12000, 1000)
        with setting_columns[2]:
            terms_per_chunk = st.number_input("Terms per request", 4, 40, 18)

    extract_clicked = st.button(
        "Extract term: description pairs",
        type="primary",
        width="stretch",
    )

    if extract_clicked:
        if provider == "Ollama Cloud" and not api_key:
            st.error("Enter an Ollama Cloud API key.")
        elif not known_models:
            st.error("Test the connection first so the app can load your available models.")
        elif model not in known_models:
            st.error("Choose a model from the available-model dropdown before extracting.")
        else:
            try:
                pages = source_pages(uploaded_file, pasted_text, int(max_pages))
                chunks = chunk_pages(pages, max_characters=int(chunk_size))
                progress_bar = st.progress(0, text="Preparing the first Ollama request…")

                def update_progress(done: int, total: int) -> None:
                    progress_bar.progress(done / total, text=f"Analyzed chunk {done} of {total}")

                entries = extract_terms_from_chunks(
                    chunks,
                    host=host,
                    model=model,
                    api_key=api_key or None,
                    terms_per_chunk=int(terms_per_chunk),
                    progress=update_progress,
                )
                progress_bar.empty()
                st.session_state["entries"] = entries
                st.session_state.pop("papers", None)
                if entries:
                    st.success(f"Extracted {len(entries)} unique term: description pairs.")
                else:
                    st.warning("Ollama did not find any usable definitions in this content.")
            except (PdfTextError, OllamaError) as exc:
                st.error(str(exc))

entries = st.session_state.get("entries", [])
if entries:
    with st.container(border=True):
        st.markdown('<div class="section-label">3 · Review the extracted glossary</div>', unsafe_allow_html=True)
        st.caption("Edit, add, or remove rows before generating questions. Each row renders as term: description.")
        edited_frame = st.data_editor(
            terms_to_frame(entries),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "term": st.column_config.TextColumn("Term", required=True, width="medium"),
                "description": st.column_config.TextColumn("Description", required=True, width="large"),
                "source_page": st.column_config.NumberColumn("Page", min_value=1, step=1, width="small"),
            },
            key="term_editor",
        )
        reviewed_entries = frame_to_terms(edited_frame)

        with st.expander("Preview term: description format"):
            for entry in reviewed_entries[:20]:
                st.markdown(
                    f'<div class="term-line"><strong>{escape(entry.term)}:</strong> {escape(entry.description)}</div>',
                    unsafe_allow_html=True,
                )

        download_col, spacer = st.columns([1, 2])
        with download_col:
            st.download_button(
                "Download glossary CSV",
                glossary_to_csv(reviewed_entries),
                file_name="ollama_glossary.csv",
                mime="text/csv",
                width="stretch",
            )

    with st.container(border=True):
        st.markdown('<div class="section-label">4 · Build the question papers</div>', unsafe_allow_html=True)
        question_style = st.selectbox(
            "Question style",
            ["WH questions generated by Ollama", "Definition prompts"],
            help=(
                "WH questions keep the extracted term as the correct answer. "
                "The other choices remain similar terms selected by TF-IDF."
            ),
        )
        controls = st.columns([1.4, 1, 1])
        with controls[0]:
            paper_title = st.text_input("Question paper title", value="Generated Study Quiz")
        with controls[1]:
            maximum_questions = max(1, len(reviewed_entries))
            question_count = st.number_input(
                "Questions count",
                1,
                maximum_questions,
                min(20, maximum_questions),
            )
        with controls[2]:
            seed = st.number_input("Shuffle seed", 0, 999999, 42)

        if len(reviewed_entries) < 4:
            st.warning("Keep at least four terms to create four-choice questions.")
        else:
            button_label = (
                "Generate four ranked WH papers"
                if question_style == "WH questions generated by Ollama"
                else "Generate four ranked definition papers"
            )
        if len(reviewed_entries) >= 4 and st.button(
            button_label, type="primary", width="stretch"
        ):
            try:
                question_prompts: dict[str, str] = {}
                if question_style == "WH questions generated by Ollama":
                    if provider == "Ollama Cloud" and not api_key:
                        raise OllamaError("Enter an Ollama Cloud API key.")
                    if not known_models or model not in known_models:
                        raise OllamaError(
                            "Test the connection and choose an available model before generating WH questions."
                        )
                    selected_entries = select_question_entries(
                        reviewed_entries,
                        question_count=int(question_count),
                        seed=int(seed),
                    )
                    wh_progress = st.progress(0, text="Asking Ollama to write WH questions…")

                    def update_wh_progress(done: int, total: int) -> None:
                        wh_progress.progress(
                            done / total,
                            text=f"Wrote question batch {done} of {total}",
                        )

                    question_prompts = generate_wh_questions(
                        selected_entries,
                        host=host,
                        model=model,
                        api_key=api_key or None,
                        progress=update_wh_progress,
                    )
                    wh_progress.empty()
                    missing = len(selected_entries) - len(question_prompts)
                    if missing:
                        st.warning(
                            f"Ollama omitted {missing} prompt(s); those questions use the definition format."
                        )

                st.session_state["papers"] = build_all_papers(
                    reviewed_entries,
                    question_count=int(question_count),
                    seed=int(seed),
                    question_prompts=question_prompts,
                )
                st.session_state["paper_title"] = paper_title
                st.success("Four ranked papers are ready. Similar terms are used as distractors.")
            except OllamaError as exc:
                st.error(str(exc))

papers = st.session_state.get("papers")
if papers:
    title = st.session_state.get("paper_title", "Generated Study Quiz")
    with st.container(border=True):
        st.markdown('<div class="section-label">5 · Preview and download</div>', unsafe_allow_html=True)
        selected_rank = st.selectbox(
            "Difficulty",
            list(RANK_LABELS),
            format_func=lambda value: RANK_LABELS[value],
        )
        button_columns = st.columns(2)
        with button_columns[0]:
            st.download_button(
                f"Download rank {selected_rank} Word paper",
                paper_to_docx(papers[selected_rank], title=title, rank=selected_rank),
                file_name=f"rank{selected_rank}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                width="stretch",
            )
        with button_columns[1]:
            st.download_button(
                "Download all four papers (.zip)",
                papers_to_zip(papers, title=title),
                file_name="ranked_question_papers.zip",
                mime="application/zip",
                width="stretch",
            )

        for question in papers[selected_rank][:10]:
            page_note = f"Source page {question.source_page}" if question.source_page else ""
            prompt_html = "<br>".join(escape(line) for line in question.prompt.splitlines())
            st.markdown(
                f"""<div class="question-card">
                <div class="question-number">Question {question.number}</div>
                <div><strong>{prompt_html}</strong></div>
                <div class="source-note">{escape(page_note)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            for letter, choice in zip("ABCD", question.choices):
                st.write(f"{letter}. {choice}")
            with st.expander("Show answer"):
                st.success(question.answer)

        if len(papers[selected_rank]) > 10:
            st.caption("Preview shows the first 10 questions. The download contains the complete paper.")
