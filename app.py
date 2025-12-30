import streamlit as st
import tempfile
import os
import re
import numpy as np

# Lazy imports to handle missing packages gracefull
try:
    import pymupdf
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

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Chat Assistant")
st.markdown("""
Upload one or more PDF files and chat with an AI assistant about their content.
The assistant uses semantic search to find relevant content and answer your questions.
""")

# Check dependencies
if not all([PYMUPDF_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE, TRANSFORMERS_AVAILABLE]):
    st.error("⚠️ Missing required packages. Please ensure all dependencies are installed.")
    missing = []
    if not PYMUPDF_AVAILABLE:
        missing.append("PyMuPDF")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing.append("sentence-transformers")
    if not TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    st.error(f"Missing: {', '.join(missing)}")
    st.stop()

# Sidebar for file upload
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat about"
    )
    
    # Process button
    process_button = st.button("Process PDFs", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("""
    **Note:** This app uses free, open-source models:
    - Embeddings: all-MiniLM-L6-v2
    - LLM: Google Flan-T5
    
    No API key required!
    """)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "model" not in st.session_state:
    st.session_state.model = None

if "qa_pipeline" not in st.session_state:
    st.session_state.qa_pipeline = None


@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model."""
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None


@st.cache_resource
def load_qa_model():
    """Load and cache the QA model."""
    try:
        qa = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            device=-1
        )
        return qa
    except Exception as e:
        st.error(f"Error loading QA model: {str(e)}")
        return None


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        doc = pymupdf.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        os.unlink(tmp_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_file.name}: {str(e)}")
        return ""


def split_text(text, chunk_size=500, overlap=50):
    """Split text into chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_len:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_relevant_chunks(query, chunks, embeddings, model, top_k=3):
    """Find most relevant chunks for a query."""
    query_embedding = model.encode([query])[0]
    
    similarities = []
    for i, chunk_emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, chunk_emb)
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:top_k]]
    
    return [chunks[i] for i in top_indices]


def process_pdfs(files, model):
    """Process uploaded PDF files."""
    if not files:
        st.error("⚠️ Please upload at least one PDF file.")
        return None
    
    if model is None:
        st.error("⚠️ Embedding model failed to load.")
        return None
    
    try:
        with st.spinner("Processing PDFs... This may take a moment."):
            all_text = ""
            file_names = []
            
            for file in files:
                text = extract_text_from_pdf(file)
                if text.strip():
                    all_text += f"\n\n--- Content from {file.name} ---\n\n{text}"
                    file_names.append(file.name)
                else:
                    st.warning(f"⚠️ No text extracted from {file.name}.")
            
            if not all_text.strip():
                st.error("❌ No text could be extracted from the uploaded PDFs.")
                return None
            
            chunks = split_text(all_text)
            
            if not chunks:
                st.error("❌ Could not split text into chunks.")
                return None
            
            # Create embeddings
            embeddings = model.encode(chunks)
            
            st.success(f"✅ Successfully processed {len(file_names)} PDF(s): {', '.join(file_names)}")
            st.info(f"📊 Created {len(chunks)} text chunks.")
            
            return chunks, embeddings, file_names
    
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {str(e)}")
        return None


def generate_answer(question, context, qa_pipeline):
    """Generate an answer using the QA pipeline."""
    try:
        prompt = f"""Answer the question based on the context below. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

Context: {context}

Question: {question}

Answer:"""
        
        result = qa_pipeline(prompt, max_length=512, do_sample=False)
        answer = result[0]['generated_text']
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# Load models
if st.session_state.model is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.model = load_embedding_model()

if st.session_state.qa_pipeline is None:
    with st.spinner("Loading language model..."):
        st.session_state.qa_pipeline = load_qa_model()

# Process PDFs
if process_button:
    result = process_pdfs(uploaded_files, st.session_state.model)
    if result:
        st.session_state.chunks, st.session_state.embeddings, st.session_state.processed_files = result
        st.session_state.chat_history = []

# Main chat interface
st.markdown("---")

if not st.session_state.chunks:
    st.info("""
    👈 **Get Started:**
    1. Upload one or more PDF files
    2. Click "Process PDFs"
    3. Start chatting about your documents!
    """)
else:
    st.success(f"📄 Currently chatting about: {', '.join(st.session_state.processed_files)}")

# Display chat history
for message in st.session_state.chat_history:
    role, content = message
    with st.chat_message(role):
        st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs...", disabled=(not st.session_state.chunks)):
    if st.session_state.qa_pipeline is None:
        st.error("⚠️ Language model failed to load.")
    else:
        # Add user message
        st.session_state.chat_history.append(("user", prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Find relevant chunks
                    relevant_chunks = find_relevant_chunks(
                        prompt,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        st.session_state.model,
                        top_k=3
                    )
                    
                    context = "\n\n".join(relevant_chunks)
                    answer = generate_answer(prompt, context, st.session_state.qa_pipeline)
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append(("assistant", answer))
                    
                except Exception as e:
                    error_message = f"❌ Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append(("assistant", error_message))

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
