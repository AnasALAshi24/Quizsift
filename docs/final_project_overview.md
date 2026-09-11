# QuizSift — Final Project Overview

## Executive summary

QuizSift is a Streamlit application that converts course material into structured, editable, and export-ready assessments. A user can upload a text-based PDF or paste source text, connect to Ollama Cloud or a local Ollama server, extract a glossary, generate questions, and download multiple versions of a question paper.

The project combines generative AI with deterministic natural-language processing. Ollama identifies concepts and writes question wording, while TF-IDF and cosine similarity select plausible distractors from the verified glossary. This division keeps the workflow flexible while preserving transparent, repeatable question-paper construction.

## Video demonstration

[Watch the QuizSift demo on Google Drive](https://drive.google.com/file/d/1L5lhWHVrplSDiiQS2dvzsapZLexO_-9i/view?usp=sharing)

### Demo source material

The demonstration uses a local copy of *Cryptography and Network Security: Principles and Practice, Global Edition* as sample input. The source PDF is not redistributed in this public repository; users should process only documents they are authorized to use.

## Problem addressed

Preparing assessment material from lectures, textbooks, and technical documents is time-consuming. Instructors must identify relevant concepts, write clear questions, construct credible wrong choices, produce several paper variants, and prepare answer keys.

QuizSift reduces that preparation effort while retaining human control. Extracted concepts remain editable before any question paper is created, so the instructor can correct omissions or unsuitable AI output.

## End-to-end workflow

1. **Connect a model.** The user selects Ollama Cloud or a local Ollama server, verifies the connection, and chooses an available model.
2. **Add source material.** The user uploads a text-based PDF or pastes educational text.
3. **Extract concepts.** Source pages are divided into manageable chunks. Ollama returns structured term-definition entries with source-page references where available.
4. **Review the glossary.** The user can edit, add, or remove entries before question generation.
5. **Generate questions.** Ollama can create concise WH-style questions whose exact answers are glossary terms. A deterministic definition-prompt mode is also available.
6. **Build paper variants.** TF-IDF vectors and cosine similarity identify semantically related terms. Four ranked papers use different similarity positions to vary distractor difficulty.
7. **Export results.** The glossary can be downloaded as CSV. Individual papers can be downloaded as Word documents, and all four papers can be packaged in one ZIP archive.

## Main capabilities

### Source ingestion

- Accepts text-based PDF documents and pasted text.
- Preserves page markers during chunking so extracted entries can retain source-page context.
- Allows limits for page count, chunk size, and the number of terms requested per model call.
- Clearly identifies scanned PDFs as requiring OCR before use.

### AI-assisted glossary extraction

- Requests structured term-definition data from Ollama.
- Rejects unusable rows and removes duplicate terms case-insensitively.
- Preserves the source language.
- Presents results in an editable table before downstream generation.

### Question generation safeguards

- Requires each generated question to have the supplied term as its exact answer.
- Checks generated wording and rejects questions that expose the answer term.
- Avoids adding unsupported facts beyond the supplied definition.
- Uses deterministic definition prompts as an alternative to AI-written WH questions.

### Distractor construction

QuizSift does not ask the language model to invent wrong answers. Instead, it compares glossary descriptions using TF-IDF vectors and cosine similarity. Each wrong choice is another real term from the reviewed glossary.

Four paper variants use different similarity ranks. This produces controlled variation while keeping the same selected answers and question order for meaningful comparison.

### Professional exports

- UTF-8 CSV glossary export
- Word question papers
- Separate answer-key page in every Word paper
- Source-page references when available
- ZIP package containing all four ranked papers

## Technology and architecture

| Area | Technology or approach |
|---|---|
| User interface | Streamlit |
| PDF processing | pdfplumber |
| Data handling | pandas |
| AI provider | Ollama Cloud or local Ollama |
| Similarity engine | scikit-learn TF-IDF and cosine similarity |
| Document generation | python-docx |
| Packaging | Python ZIP utilities |
| Automated checks | pytest |

The implementation separates interface code from PDF extraction, Ollama communication, question construction, data models, and export logic. This makes the deterministic components independently testable and reduces coupling between the UI and business logic.

## Privacy and credential handling

- A cloud API key can be supplied for the current session or configured securely on the server.
- A server-side key is not copied into browser widget state.
- Local Ollama normally requires no API key.
- Uploaded content and generated working data are held in the Streamlit session; the application does not implement a persistent document database.
- Secrets should be stored in environment variables or Streamlit secrets and must never be committed to the repository.

Users should still review the privacy terms of the selected model provider before processing confidential or regulated material.

## Validation snapshot

The verified automated test suite covers:

- structured term parsing and source-page preservation;
- Ollama host normalization;
- rejection of questions that contain their answer term;
- page-marker preservation during PDF chunking;
- integrity of all four paper variants, including unique answer choices and consistent correct answers.

At the latest documentation review, all five core automated tests passed. The application modules also completed Python syntax compilation successfully.

## Limitations

- Language-model output can omit, misinterpret, or overgeneralize source material.
- Scanned or image-only PDFs require OCR before extraction.
- Distractor quality depends on the breadth and accuracy of the reviewed glossary.
- Similar terms are not automatically pedagogically equivalent; instructor review remains necessary.
- Availability, latency, and model quality depend on the selected Ollama deployment.
- The generated papers are drafts and should not be treated as validated examinations without human review.

## Responsible use

QuizSift is designed to assist educators, not replace subject-matter judgment. Before distributing a generated assessment, an instructor should verify factual accuracy, answer uniqueness, difficulty, accessibility, curriculum alignment, and copyright permissions for the source material.

## Public repository scope

The current public repository is a product-documentation release. Application source code, provider credentials, uploaded learning materials, and generated assessments are not included in this documentation package.

The verified implementation uses Python 3, Streamlit, pdfplumber, pandas, scikit-learn, python-docx, pytest, and an Ollama endpoint. A future public source release should include a reproducible dependency file, setup instructions, an example environment file without secrets, and a documented sample workflow.
