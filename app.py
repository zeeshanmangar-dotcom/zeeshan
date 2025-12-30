import streamlit as st
import tempfile
import os
import numpy as np
from typing import List, Tuple

# Lazy imports with graceful fallbacks
try:
    import pymupdf  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ========================= Page Config =========================
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 PDF Chat Assistant")
st.markdown(
    """
Upload one or more PDF files and ask questions about their content.  
The assistant uses **semantic search** over document chunks and a local LLM to answer your questions — no API keys needed!
"""
)

# ========================= Dependency Check =========================
if not all([PYMUPDF_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE, TRANSFORMERS_AVAILABLE]):
    st.error("⚠️ Missing required packages. Install them with:")
    st.code(
        "pip install pymupdf sentence-transformers transformers torch",
        language="bash",
    )
    missing = []
    if not PYMUPDF_AVAILABLE:
        missing.append("pymupdf")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing.append("sentence-transformers")
    if not TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    st.error(f"Missing packages: {', '.join(missing)}")
    st.stop()

# ========================= Sidebar =========================
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once",
    )

    process_button = st.button("Process PDFs", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(
        """
        **Models Used**  
        • Embeddings: `all-MiniLM-L6-v2` (fast & lightweight)  
        • LLM: `google/flan-t5-base` (runs locally)  
        
        Fully offline after initial download — no data leaves your machine.
        """
    )

# ========================= Session State Initialization =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Tuple[str, str]] = []
if "chunks" not in st.session_state:
    st.session_state.chunks: List[str] = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files: List[str] = []
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "qa_pipeline" not in st.session_state:
    st.session_state.qa_pipeline = None

# ========================= Model Loading (Cached) =========================
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading QA model (this may take a minute)...")
def load_qa_pipeline():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # You can upgrade to 'flan-t5-large' or 'flan-t5-xl' if VRAM allows
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        device=-1,  # CPU; change to 0 for GPU if available
    )

# Load models if not already
if st.session_state.embedding_model is None:
    st.session_state.embedding_model = load_embedding_model()

if st.session_state.qa_pipeline is None:
    st.session_state.qa_pipeline = load_qa_pipeline()

# ========================= PDF Processing Functions =========================
def extract_text_from_pdf(pdf_bytes: bytes, filename: str) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        doc = pymupdf.open(tmp_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        st.error(f"Failed to extract text from {filename}: {e}")
        return ""

def recursive_chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """Better chunking: splits on sentence/paragraph boundaries when possible."""
    separators = ["\n\n", "\n", ". ", "? ", "! "]
    chunks = [text]

    for sep in separators:
        new_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                new_chunks.append(chunk)
            else:
                parts = chunk.split(sep)
                current = ""
                for part in parts:
                    if len(current + part + sep) <= chunk_size:
                        current += (part + sep) if current else part
                    else:
                        if current:
                            new_chunks.append(current.strip())
                        current = part
                if current:
                    new_chunks.append(current.strip())
        chunks = new_chunks
        if all(len(c) <= chunk_size for c in chunks):
            break

    # Overlap handling
    final_chunks = []
    for i, chunk in enumerate(chunks):
        final_chunks.append(chunk)
        if i < len(chunks) - 1:
            overlap_start = max(0, len(chunk) - chunk_overlap)
            final_chunks.append(chunk[overlap_start:])

    return [c.strip() for c in final_chunks if c.strip()]

def compute_embeddings(chunks: List[str], model: SentenceTransformer):
    """Compute and cache embeddings."""
    with st.spinner(f"Computing embeddings for {len(chunks)} chunks..."):
        return model.encode(chunks, normalize_embeddings=True)

def find_relevant_chunks(query: str, chunks: List[str], embeddings, model, top_k: int = 4) -> List[str]:
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(embeddings, query_emb)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_answer(question: str, context: str, qa_pipeline) -> str:
    prompt = f"""Based only on the following context, answer the question concisely. 
If the answer is not present, respond with "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    try:
        result = qa_pipeline(prompt)[0]["generated_text"]
        # Clean up common FLAN-T5 artifacts
        return result.strip()
    except Exception as e:
        return f"Error during generation: {e}"

# ========================= Process PDFs =========================
if process_button and uploaded_files:
    all_text = ""
    processed_names = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(uploaded_files):
        status_text.text(f"Extracting text from {file.name}...")
        text = extract_text_from_pdf(file.getvalue(), file.name)
        if text:
            all_text += f"\n\n--- Document: {file.name} ---\n\n{text}"
            processed_names.append(file.name)
        progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("Chunking text...")
    st.session_state.chunks = recursive_chunk_text(all_text, chunk_size=800, chunk_overlap=100)

    if not st.session_state.chunks:
        st.error("No text chunks created. Please check your PDFs.")
        st.stop()

    status_text.text("Generating embeddings...")
    st.session_state.embeddings = compute_embeddings(st.session_state.chunks, st.session_state.embedding_model)
    st.session_state.processed_files = processed_names

    st.session_state.chat_history = []  # Reset chat on new documents
    progress_bar.empty()
    status_text.empty()

    st.success(f"✅ Processed {len(processed_names)} PDF(s): {', '.join(processed_names)}")
    st.info(f"📊 Created {len(st.session_state.chunks)} chunks for retrieval.")

# ========================= Main Chat Interface =========================
st.markdown("---")

if not st.session_state.chunks:
    st.info("👈 Upload PDF files in the sidebar and click **Process PDFs** to begin chatting!")
else:
    st.success(f"📄 Active documents: {', '.join(st.session_state.processed_files)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input
    if prompt := st.chat_input("Ask a question about the uploaded PDFs..."):
        # Add user message
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and thinking..."):
                relevant_chunks = find_relevant_chunks(
                    prompt,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    st.session_state.embedding_model,
                    top_k=4,
                )
                context = "\n\n".join(relevant_chunks)
                answer = generate_answer(prompt, context, st.session_state.qa_pipeline)
                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
