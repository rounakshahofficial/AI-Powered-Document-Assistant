# Gemini Document Assistant

A Streamlit-based document Q&A and summarization app powered by Google Gemini, LangChain, and Chroma. Upload a PDF or plain-text document, index its contents, and ask grounded questions about it without leaving the browser.

## Features

- Upload PDFs or `.txt` files
- Automatic document chunking and vector indexing with Chroma
- Retrieval-augmented generation using Google Gemini embeddings and chat models
- Question answering grounded in the uploaded document only
- Document summary mode for broader overviews
- Model fallback support for transient Google API overloads
- Optional removal of repeated PDF headers/footers to improve retrieval quality

## Tech stack

- Python 3.10+
- Streamlit
- LangChain
- Chroma
- Google Gemini API

## Project structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
└── .venv/   # local virtual environment (not committed)
```

## Prerequisites

- A Google AI API key with access to Gemini models
- Python 3.10 or newer

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Google API key.

You can either export it in your shell:

```bash
export GOOGLE_API_KEY="your-google-api-key"
```

Or create a local secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml` and set:

```toml
GOOGLE_API_KEY = "your-google-api-key"
```

## Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser and upload a document.

## How it works

- The uploaded file is parsed and split into chunks.
- Each chunk is embedded using Google Gemini embeddings.
- Chroma stores the embeddings for retrieval.
- A retrieval chain answers the user query using only the relevant document chunks.
- If the query appears to ask for a summary, the app summarizes the document's sections in batches and combines the results.

## Notes

- Some PDFs are scanned images and may require OCR before text extraction is possible.
- If the configured Gemini model returns transient overload errors, the app can retry or fall back to the next configured model.
- Keep your API key private and avoid committing secrets to version control.

## License

This project is provided as-is for local use and experimentation.
