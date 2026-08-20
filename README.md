# Ask My Docs

A small RAG (retrieval-augmented generation) chat app for asking questions about your own PDFs. Upload documents, and get answers grounded in their content, with sources and retrieved chunks shown alongside each answer.

Built with [Streamlit](https://streamlit.io/), [ChromaDB](https://www.trychroma.com/) for local vector storage, `sentence-transformers` for embeddings, and [Groq](https://groq.com/) for generation.

## Features

- Upload PDFs from the browser and ask questions about them in a chat interface.
- Per-session isolation — each browser session gets its own private document collection; nobody sees anyone else's uploads or chat.
- Answers include cited sources and an expandable view of the retrieved chunks.
- Per-file and per-session limits (file size, page count, extracted text size, total upload size, files per session) to bound resource usage.
- Basic rate limiting on questions per session.

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your Groq API key:

   ```
   GROQ_API_KEY=your-key-here
   ```

## Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints, upload one or more PDFs from the sidebar, and start asking questions.

## Project structure

- `app.py` — the Streamlit app (upload, index, chat, per-session isolation and limits).
- `ingest.py` — standalone CLI script to bulk-index PDFs from a local `docs/` folder into a shared Chroma collection.
- `ask.py` — standalone CLI script to ask a single question against that shared collection.
- `.streamlit/config.toml` — Streamlit theme and server configuration.

`chroma_db/` (vector store data) and `docs/` (local PDF inputs) are git-ignored, along with `.venv/` and `.env`.
