import hashlib
import os
import random
import re
import tempfile
import time
from collections import Counter

import streamlit as st

# Make the key available to the Google SDK however it was supplied:
# a real environment variable, or Streamlit secrets (.streamlit/secrets.toml).
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error(
            "No Google API key found. Set GOOGLE_API_KEY as an environment variable, "
            "or add it to .streamlit/secrets.toml as GOOGLE_API_KEY = \"your-key\"."
        )
        st.stop()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -------------------------------------------------------------
# CONFIG - check Google's current model list if you get a 404
# -------------------------------------------------------------
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Models shown in the sidebar picker, fastest/cheapest first. Edit this list
# if Google retires or renames one.
CHAT_MODEL_OPTIONS = [
    "gemini-3.8-flash",       # best quality/speed balance
    "gemini-3.5-flash-lite",  # cheaper/faster
]
BATCH_SIZE = 50            # chunks per embedding request
BATCH_SLEEP_SECONDS = 0.2  # small pause between batches; retry handles real 429s
SUMMARY_BATCH_CHARS = 18000
MAX_RETRIES = 4
FALLBACK_ANSWER = "I cannot answer this question based on the provided document."
SUMMARY_REQUEST_PATTERN = re.compile(r"\b(summary|summarize|summarise|overview|key points|main points)\b", re.I)

SYSTEM_PROMPT = (
    "Answer using only the provided context. Combine relevant details across passages and do not "
    "require an exact wording match. If some parts are supported, answer those parts and say which "
    f"parts are not covered. Reply EXACTLY with '{FALLBACK_ANSWER}' only when the context has no "
    "relevant information at all. Do not use outside knowledge.\n\n"
    "Context:\n{context}"
)
SUMMARY_SYSTEM_PROMPT = (
    "Create a useful, faithful summary using only the supplied document sections. Preserve the main "
    "ideas, important details, and conclusions. Do not invent information or refuse just because "
    "the requested summary is not stated verbatim.\n\n"
    "Document sections:\n{context}"
)

# Error substrings worth retrying: transient server-side conditions, not
# problems retrying will ever fix (bad API key, bad model name, etc).
_RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED", "500")

# -------------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------------
st.set_page_config(page_title="Gemini Document Assistant", layout="wide")
st.title("📂 AI-Powered Document Assistant")
st.caption("RAG pipeline: LangChain + Chroma + Gemini")

st.sidebar.header("🤖 Model")
chat_model = st.sidebar.selectbox(
    "Chat model",
    CHAT_MODEL_OPTIONS,
    index=1,
    help="Switch models if you're hitting 503 (overloaded) errors on one.",
)
use_fallback = st.sidebar.checkbox(
    "Auto-fallback to next model on overload",
    value=True,
    help="If the selected model is overloaded after retries, automatically "
    "try the next model in the list above before giving up.",
)

st.sidebar.header("🔧 Retrieval Engine Tweaks")
st.sidebar.markdown("Experiment with these settings for your evaluation report.")
chunk_size = st.sidebar.slider("Chunk Size (Characters)", 200, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 400, 100, step=10)
top_k = st.sidebar.slider("Chunks retrieved (k)", 1, 8, 6)
strip_boilerplate = st.sidebar.checkbox("Strip repeated headers/footers", value=True)

if chunk_overlap >= chunk_size:
    st.sidebar.error("Overlap must be smaller than chunk size.")
    st.stop()


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def load_documents(file_bytes: bytes, filename: str):
    """Write upload to a temp file with the right extension and load it."""
    suffix = os.path.splitext(filename)[1].lower() or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        if suffix == ".txt":
            return TextLoader(tmp_path, encoding="utf-8", autodetect_encoding=True).load()
        return PyPDFLoader(tmp_path).load()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def remove_repeated_boilerplate(documents, min_page_fraction=0.4):
    """
    PDFs often repeat the same header/footer line on every page (approval
    stamps, page banners, etc). That text has no informational value but,
    once chunked, it dilutes small chunks and gets retrieved for unrelated
    queries. Strip any line that recurs on a large fraction of pages.
    """
    if len(documents) < 3:
        return documents

    line_counts = Counter()
    for doc in documents:
        # Count each distinct line once per page, not once per occurrence.
        lines = {ln.strip() for ln in doc.page_content.splitlines() if ln.strip()}
        line_counts.update(lines)

    threshold = max(2, int(len(documents) * min_page_fraction))
    boilerplate = {line for line, count in line_counts.items() if count >= threshold}

    if not boilerplate:
        return documents

    for doc in documents:
        kept = [ln for ln in doc.page_content.splitlines() if ln.strip() not in boilerplate]
        doc.page_content = "\n".join(kept)
    return documents


def group_sources_by_reference(documents):
    """Combine retrieved chunks that point to the same source page."""
    groups = {}
    for document in documents:
        page = document.metadata.get("page")
        reference = ("page", page) if isinstance(page, int) else ("text", None)
        if reference not in groups:
            label = f"Page {page + 1}" if isinstance(page, int) else "Text file"
            groups[reference] = {"label": label, "previews": []}

        preview = " ".join(document.page_content.split())
        if preview and preview not in groups[reference]["previews"]:
            groups[reference]["previews"].append(preview)

    return [
        (group["label"], " ... ".join(group["previews"]))
        for group in groups.values()
    ]


def add_batch_with_retry(store, batch):
    """Add a batch of chunks, backing off on rate-limit/availability errors."""
    for attempt in range(MAX_RETRIES):
        try:
            store.add_documents(batch)
            return
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            st.toast(f"API hiccup ({type(exc).__name__}); retrying in {wait}s...")
            time.sleep(wait)


def invoke_with_retry(chain, payload, max_retries=4, base_wait=2, status_fn=None):
    """
    Call chain.invoke with exponential backoff + jitter, retrying only on
    transient errors (model overloaded, rate limited, timed out). Anything
    else (auth, bad request, model not found) fails immediately since
    retrying it would just waste the same amount of time repeatedly.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            retryable = any(marker in msg for marker in _RETRYABLE_MARKERS)
            if not retryable or attempt == max_retries - 1:
                raise
            wait = base_wait * (2 ** attempt) + random.uniform(0, 1)
            if status_fn:
                status_fn(
                    f"Model busy (attempt {attempt + 1}/{max_retries}); "
                    f"retrying in {wait:.1f}s..."
                )
            time.sleep(wait)
    raise last_exc  # pragma: no cover - loop always returns or raises above


def is_summary_request(query):
    return bool(SUMMARY_REQUEST_PATTERN.search(query))


def group_text_by_size(texts, max_chars=SUMMARY_BATCH_CHARS):
    batches = []
    current_batch = []
    current_length = 0
    for text in texts:
        separator_length = 2 if current_batch else 0
        if current_batch and current_length + separator_length + len(text) > max_chars:
            batches.append("\n\n".join(current_batch))
            current_batch = []
            current_length = 0
            separator_length = 0
        current_batch.append(text)
        current_length += separator_length + len(text)
    if current_batch:
        batches.append("\n\n".join(current_batch))
    return batches


def summarize_document(documents, user_query, model_name, status_fn=None):
    """Summarize every document chunk in bounded batches, then combine the results."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, timeout=30, max_retries=0)
    summary_chain = (
        ChatPromptTemplate.from_messages(
            [("system", SUMMARY_SYSTEM_PROMPT), ("human", "Request: {input}")]
        )
        | llm
        | StrOutputParser()
    )

    sections = []
    for index, document in enumerate(documents, start=1):
        page = document.metadata.get("page")
        label = f"Page {page + 1}" if isinstance(page, int) else f"Section {index}"
        sections.append(f"[{label}]\n{document.page_content}")

    batches = group_text_by_size(sections)
    while batches:
        summaries = [
            invoke_with_retry(
                summary_chain,
                {"input": user_query, "context": batch},
                status_fn=status_fn,
            ).strip()
            for batch in batches
        ]
        if len(summaries) == 1:
            return summaries[0]
        batches = group_text_by_size(summaries)

    return FALLBACK_ANSWER


def build_retriever(file_bytes, filename, c_size, c_overlap, file_hash, strip_bp):
    """
    Load, chunk, embed and index the document. This is the expensive step
    (many embedding API calls), so it deliberately does NOT depend on the
    chat model: switching models never triggers a re-index.
    """
    documents = load_documents(file_bytes, filename)
    if strip_bp:
        documents = remove_repeated_boilerplate(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
    if not chunks:
        raise ValueError(
            "No extractable text found. If this is a scanned PDF, it needs OCR first."
        )

    # No task_type set: the wrapper uses retrieval_document for embed_documents
    # and retrieval_query for embed_query automatically.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # Unique collection per (file, settings) so rebuilds never mix duplicates.
    collection = f"doc_{file_hash[:12]}_{c_size}_{c_overlap}"
    store = Chroma(collection_name=collection, embedding_function=embeddings)

    progress = st.progress(0, text="Indexing document...")
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        add_batch_with_retry(store, batch)
        done = min(start + BATCH_SIZE, total)
        progress.progress(done / total, text=f"Indexed {done} of {total} chunks")
        if done < total:
            time.sleep(BATCH_SLEEP_SECONDS)
    progress.empty()

    return store, total, chunks


def make_chain(retriever, model_name):
    """Build the QA chain for a given chat model (cheap, no API calls)."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{input}")]
    )
    # max_retries=0: we handle retries ourselves in invoke_with_retry, with
    # visible status updates and markers that distinguish "worth retrying"
    # (503/429/timeout) from "retrying won't help" (bad key, bad model name).
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, timeout=30, max_retries=0)
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))


# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your source document", type=["pdf", "txt"])

if uploaded_file is None:
    st.info("Upload a document above to begin querying.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()
if st.session_state.get("chat_file_hash") != file_hash:
    st.session_state["chat_history"] = []
    st.session_state["chat_file_hash"] = file_hash

# Rebuild only when the file content or indexing settings change.
# Retrieval count and chat model do not affect the document index.
build_key = (file_hash, chunk_size, chunk_overlap, strip_boilerplate)

if (
    st.session_state.get("build_key") != build_key
    or "vector_store" not in st.session_state
    or "document_chunks" not in st.session_state
):
    try:
        vector_store, n_chunks, document_chunks = build_retriever(
            file_bytes, uploaded_file.name, chunk_size, chunk_overlap,
            file_hash, strip_boilerplate,
        )
    except Exception as exc:
        st.session_state.pop("build_key", None)
        st.error(f"Could not index the document: {exc}")
        st.stop()
    st.session_state["vector_store"] = vector_store
    st.session_state["n_chunks"] = n_chunks
    st.session_state["document_chunks"] = document_chunks
    st.session_state["build_key"] = build_key

st.success(f"Document indexed: {st.session_state['n_chunks']} chunks.")
retrieved_k = min(top_k, st.session_state["n_chunks"])
if retrieved_k < top_k:
    st.caption(
        f"Requested {top_k} chunks, but the document contains only {retrieved_k}; "
        "retrieving all available chunks."
    )
retriever = st.session_state["vector_store"].as_retriever(
    search_type="mmr", search_kwargs={"k": retrieved_k, "fetch_k": max(retrieved_k * 4, 10)}
)

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("model"):
            st.caption(f"Answered by fallback model: {message['model']}")
        if message.get("sources"):
            st.markdown("**Reference Sources**")
            for label, preview in message["sources"]:
                st.markdown(f"**📄 {label}**")
                st.caption(f'"{preview[:250]}..."')

user_query = st.chat_input("Ask a question about your document")

if user_query:
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    status_placeholder = st.empty()
    summary_request = is_summary_request(user_query)

    # Selected model first, then (optionally) each following model in the list.
    start = CHAT_MODEL_OPTIONS.index(chat_model)
    candidates = CHAT_MODEL_OPTIONS[start:] if use_fallback else [chat_model]

    output, used_model, last_exc = None, None, None
    spinner_text = "Summarizing the full document..." if summary_request else "Searching knowledge base..."
    with st.spinner(spinner_text):
        for model_name in candidates:
            try:
                if summary_request:
                    answer = summarize_document(
                        st.session_state["document_chunks"],
                        user_query,
                        model_name,
                        status_fn=lambda msg: status_placeholder.info(msg),
                    )
                    output = {"answer": answer, "context": []}
                else:
                    chain = make_chain(retriever, model_name)
                    output = invoke_with_retry(
                        chain,
                        {"input": user_query},
                        status_fn=lambda msg: status_placeholder.info(msg),
                    )
                used_model = model_name
                break
            except Exception as exc:
                last_exc = exc
                if not any(m in str(exc) for m in _RETRYABLE_MARKERS):
                    break  # bad key / bad model name: trying another model won't help
                status_placeholder.info(f"{model_name} overloaded, trying next model...")

    status_placeholder.empty()

    if output is None:
        if last_exc is not None and any(m in str(last_exc) for m in _RETRYABLE_MARKERS):
            error_message = (
                "Gemini is overloaded right now and didn't recover after several "
                "retries. This is on Google's side, not your code. Wait a minute "
                "and try again, or pick a different model in the sidebar."
            )
        else:
            error_message = f"Query failed: {type(last_exc).__name__}: {last_exc}"
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": error_message}
        )
        st.rerun()

    answer = output["answer"].strip()
    sources = output.get("context", [])
    is_fallback = answer.rstrip(".").lower() == FALLBACK_ANSWER.rstrip(".").lower()
    source_references = group_sources_by_reference(sources) if sources and not is_fallback else []

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": answer,
            "model": used_model if used_model != chat_model else None,
            "sources": source_references,
        }
    )
    st.rerun()