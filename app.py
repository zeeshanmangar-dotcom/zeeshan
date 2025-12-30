import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, AIMessage
import pymupdf  # PyMuPDF for PDF processing
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Chat Assistant")
st.markdown("""
Upload one or more PDF files and chat with an AI assistant about their content.
The assistant uses RAG (Retrieval-Augmented Generation) with free, open-source models to provide accurate answers based on your documents.
""")

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
    - Embeddings: sentence-transformers/all-MiniLM-L6-v2
    - LLM: Google Flan-T5 (running locally)
    
    No API key required! The first run may take a moment to download models.
    """)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "llm" not in st.session_state:
    st.session_state.llm = None


@st.cache_resource
def load_embeddings():
    """Load and cache the embedding model."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None


@st.cache_resource
def load_llm():
    """Load and cache the language model."""
    try:
        from transformers import pipeline
        
        # Use a smaller, local model for faster inference
        qa_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            device=-1  # CPU
        )
        
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Open and extract text from PDF
        doc = pymupdf.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_file.name}: {str(e)}")
        return ""


def process_pdfs(files, embeddings):
    """Process uploaded PDF files and create vector store."""
    if not files:
        st.error("⚠️ Please upload at least one PDF file.")
        return None
    
    if embeddings is None:
        st.error("⚠️ Embedding model failed to load.")
        return None
    
    try:
        with st.spinner("Processing PDFs... This may take a moment."):
            # Extract text from all PDFs
            all_text = ""
            file_names = []
            
            for file in files:
                text = extract_text_from_pdf(file)
                if text.strip():
                    all_text += f"\n\n--- Content from {file.name} ---\n\n{text}"
                    file_names.append(file.name)
                else:
                    st.warning(f"⚠️ No text extracted from {file.name}. It might be empty or image-based.")
            
            if not all_text.strip():
                st.error("❌ No text could be extracted from the uploaded PDFs.")
                return None
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(all_text)
            
            if not chunks:
                st.error("❌ Could not split text into chunks.")
                return None
            
            # Create vector store
            vectorstore = FAISS.from_texts(chunks, embeddings)
            
            st.success(f"✅ Successfully processed {len(file_names)} PDF(s): {', '.join(file_names)}")
            st.info(f"📊 Created {len(chunks)} text chunks for retrieval.")
            
            return vectorstore, file_names
    
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {str(e)}")
        return None


def generate_answer(question, context, qa_pipeline):
    """Generate an answer using the QA pipeline."""
    try:
        # Format the prompt for the model
        prompt = f"""Answer the question based on the context below. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

Context: {context}

Question: {question}

Answer:"""
        
        # Generate answer
        result = qa_pipeline(prompt, max_length=512, do_sample=False)
        answer = result[0]['generated_text']
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# Load models on first run
if st.session_state.embeddings is None:
    with st.spinner("Loading embedding model... (first time only)"):
        st.session_state.embeddings = load_embeddings()

if st.session_state.llm is None:
    with st.spinner("Loading language model... (first time only)"):
        st.session_state.llm = load_llm()

# Process PDFs when button is clicked
if process_button:
    result = process_pdfs(uploaded_files, st.session_state.embeddings)
    if result:
        st.session_state.vectorstore, st.session_state.processed_files = result
        # Clear chat history when processing new PDFs
        st.session_state.chat_history = []

# Main chat interface
st.markdown("---")

# Display instructions if no PDFs processed
if st.session_state.vectorstore is None:
    st.info("""
    👈 **Get Started:**
    1. Upload one or more PDF files
    2. Click "Process PDFs"
    3. Start chatting about your documents!
    
    ⚡ **Note:** The first run will download required models (~400MB). This happens once and future runs will be faster.
    """)
else:
    st.success(f"📄 Currently chatting about: {', '.join(st.session_state.processed_files)}")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs...", disabled=(st.session_state.vectorstore is None)):
    if st.session_state.llm is None:
        st.error("⚠️ Language model failed to load. Please refresh the page.")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant documents
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.invoke(prompt)
                    
                    # Combine context from retrieved documents
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Generate answer
                    answer = generate_answer(prompt, context, st.session_state.llm)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                except Exception as e:
                    error_message = f"❌ Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append(AIMessage(content=error_message))

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
