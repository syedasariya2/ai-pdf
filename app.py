import os
import re
import tempfile
from typing import List, Dict, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 350       # tokens
CHUNK_OVERLAP = 60     # tokens
TOP_K = 5


# -----------------------------
# Load open-source embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


# -----------------------------
# Groq client
# -----------------------------
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]

    if not api_key:
        return None

    return Groq(api_key=api_key)


# -----------------------------
# PDF text extraction
# -----------------------------
def extract_pdf(uploaded_file) -> List[Dict]:
    """Extract text page-by-page from the uploaded PDF."""
    pages = []

    pdf_bytes = uploaded_file.getvalue()

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    return pages


# -----------------------------
# Tokenization + chunking
# -----------------------------
def create_chunks(
    pages: List[Dict],
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Tokenize page text and create overlapping token-based chunks.
    The original page number is preserved as metadata.
    """
    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        start = 0

        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            chunk_token_ids = token_ids[start:end]

            chunk_text = tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= len(token_ids):
                break

            start = end - overlap

    return chunks


# -----------------------------
# FAISS vector database
# -----------------------------
def build_faiss_index(
    chunks: List[Dict],
    embedding_model,
) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Create embeddings and store them in a FAISS index."""
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    # Normalize vectors so inner product becomes cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve(
    question: str,
    index,
    chunks: List[Dict],
    embedding_model,
    top_k: int = TOP_K,
) -> List[Dict]:
    """Retrieve the most relevant PDF chunks for a question."""
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    faiss.normalize_L2(question_embedding)

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        result = chunks[idx].copy()
        result["score"] = float(score)
        results.append(result)

    return results


# -----------------------------
# Groq LLM generation
# -----------------------------
def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    chat_history: List[Dict],
    client: Groq,
) -> str:
    """Generate an answer using retrieved PDF context."""
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful PDF question-answering assistant.

Your job is to answer the user's question using ONLY the provided PDF context.

Rules:
1. Use the retrieved PDF context as your primary and only source of factual information.
2. Do not invent facts that are not supported by the PDF.
3. If the answer cannot be found in the provided context, clearly say:
   "I couldn't find that information in the uploaded PDF."
4. Keep the answer clear and concise, but include enough explanation to be useful.
5. When possible, mention the relevant page number(s).
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Keep a small amount of conversation history so the app can handle follow-up questions.
    for message in chat_history[-6:]:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    user_prompt = f"""PDF context:

{context}

User question:
{question}

Answer using the PDF context. Include page references when appropriate."""

    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
    )

    return completion.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# -----------------------------
# UI
# -----------------------------
st.title("📚 PDF RAG Assistant")
st.write(
    "Upload a PDF, extract and tokenize its text, create embeddings with an "
    "open-source embedding model, store them in FAISS, and ask questions about the document."
)

with st.sidebar:
    st.header("⚙️ RAG Settings")
    top_k = st.slider("Retrieved chunks", min_value=2, max_value=10, value=TOP_K)

    st.markdown("### Models")
    st.caption(f"Embedding: `{EMBEDDING_MODEL}`")
    st.caption(f"LLM: `{LLM_MODEL}`")
    st.caption("Vector store: FAISS")

    if st.button("🗑️ Clear current document"):
        st.session_state.chunks = None
        st.session_state.faiss_index = None
        st.session_state.document_name = None
        st.session_state.messages = []
        st.rerun()

uploaded_file = st.file_uploader(
    "📄 Upload a PDF document",
    type=["pdf"],
    help="Upload one PDF to build the RAG knowledge base.",
)

if uploaded_file is not None:
    if st.session_state.document_name != uploaded_file.name:
        st.session_state.messages = []
        st.session_state.chunks = None
        st.session_state.faiss_index = None

        with st.status("Building your RAG knowledge base...", expanded=True) as status:
            st.write("1. Extracting PDF text...")
            pages = extract_pdf(uploaded_file)

            if not pages:
                status.update(label="No readable text found", state="error")
                st.error(
                    "No selectable text was found in this PDF. "
                    "This version of the app expects a text-based PDF."
                )
                st.stop()

            st.write(f"Extracted text from {len(pages)} page(s).")

            st.write("2. Loading open-source tokenizer...")
            tokenizer = load_tokenizer()

            st.write("3. Tokenizing and creating chunks...")
            chunks = create_chunks(
                pages,
                tokenizer,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            if not chunks:
                status.update(label="Chunking failed", state="error")
                st.error("Could not create text chunks from this PDF.")
                st.stop()

            st.write(f"Created {len(chunks)} chunks.")

            st.write("4. Creating embeddings with an open-source model...")
            embedding_model = load_embedding_model()

            st.write("5. Storing embeddings in FAISS...")
            index, _ = build_faiss_index(chunks, embedding_model)

            st.session_state.chunks = chunks
            st.session_state.faiss_index = index
            st.session_state.document_name = uploaded_file.name

            status.update(
                label="RAG knowledge base is ready!",
                state="complete",
            )

    st.success(
        f"Document ready: **{uploaded_file.name}** | "
        f"{len(st.session_state.chunks)} chunks indexed in FAISS"
    )

    st.divider()

    # Display previous conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        client = get_groq_client()

        if client is None:
            st.error(
                "GROQ_API_KEY is not configured. Add it to Streamlit Cloud "
                "Secrets or set it as an environment variable."
            )
            st.stop()

        # Show user question
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        embedding_model = load_embedding_model()

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and generating an answer..."):
                retrieved_chunks = retrieve(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    embedding_model,
                    top_k=top_k,
                )

                answer = generate_answer(
                    question,
                    retrieved_chunks,
                    st.session_state.messages[:-1],
                    client,
                )

            st.markdown(answer)

            with st.expander("🔎 Retrieved sources"):
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    st.markdown(
                        f"**Source {i} — Page {chunk['page']} "
                        f"(similarity: {chunk['score']:.3f})**"
                    )
                    st.write(chunk["text"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("👆 Upload a PDF to start building the RAG pipeline.")

st.caption(
    "RAG pipeline: PDF → Text Extraction → Tokenization → Chunking → "
    "Open-Source Embeddings → FAISS → Retrieval → Groq-hosted Llama"
)
