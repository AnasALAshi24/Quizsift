# QuizSift

QuizSift is the new Ollama-powered version of the question-paper builder. It accepts a PDF or pasted text, asks Ollama to extract genuine glossary entries, presents them as `term: description`, generates WH-style questions whose answers remain the terms, and creates four multiple-choice papers using TF-IDF/cosine-similarity distractors.

## Versions

- Old version: T5 and BERT.
- New QuizSift version: Ollama.

## What changed from the notebook

- Ollama decides what is a real term and description; there are no colon, capitalization, or newline extraction rules.
- PDF upload and pasted text are supported.
- Extracted terms can be reviewed and edited before question generation.
- Ollama writes term-answerable questions such as “What process converts plaintext into ciphertext?”
- The three wrong choices remain semantically similar terms selected from the glossary.
- The original four similarity ranks are retained.
- Every Word paper has a separate answer-key page.
- No BERT, T5, or NLTK model downloads are required.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

### Local Ollama

Start Ollama, pull a model such as `qwen3:8b`, then select **Local Ollama** in the app. A local server normally uses `http://localhost:11434` and does not need an API key.

### Ollama Cloud

Set the key outside the code:

```powershell
$env:OLLAMA_API_KEY="your-key"
streamlit run app.py
```

Alternatively, paste the key into the password field for the current Streamlit session. The key is not saved by the app.

## Notes

- Large PDFs are processed in chunks and can make several model requests.
- Text-based PDFs work directly. Scanned PDFs need OCR before upload.
- Review the extracted glossary before generating questions; an LLM can still omit or misinterpret material.
