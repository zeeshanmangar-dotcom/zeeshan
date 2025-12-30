
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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
The assistant uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on your documents.
""")

# Sidebar for API key and file upload
with st.sidebar:
    st.header("Configuration")
    
    # OpenAI API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to use the chatbot"
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    
    # File uploader
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat about"
    )
    
    # Process button
    process_button = st.button("Process PDFs", type="primary", use_container_width=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


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


def process_pdfs(files, api_key):
    """Process uploaded PDF files and create vector store."""
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        return None
    
    if not files:
        st.error("⚠️ Please upload at least one PDF file.")
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
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(all_text)
            
            if not chunks:
                st.error("❌ Could not split text into chunks.")
                return None
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectorstore = FAISS.from_texts(chunks, embeddings)
            
            st.success(f"✅ Successfully processed {len(file_names)} PDF(s): {', '.join(file_names)}")
            st.info(f"📊 Created {len(chunks)} text chunks for retrieval.")
            
            return vectorstore, file_names
    
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {str(e)}")
        return None


def create_conversation_chain(vectorstore, api_key):
    """Create a conversational retrieval chain."""
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=api_key
        )
        
        # Create memory for conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create conversational retrieval chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        return conversation_chain
    
    except Exception as e:
        st.error(f"❌ Error creating conversation chain: {str(e)}")
        return None


# Process PDFs when button is clicked
if process_button:
    result = process_pdfs(uploaded_files, api_key)
    if result:
        st.session_state.vectorstore, st.session_state.processed_files = result
        st.session_state.conversation_chain = create_conversation_chain(
            st.session_state.vectorstore,
            api_key
        )
        # Clear chat history when processing new PDFs
        st.session_state.chat_history = []

# Main chat interface
st.markdown("---")

# Display instructions if no PDFs processed
if st.session_state.vectorstore is None:
    st.info("""
    👈 **Get Started:**
    1. Enter your OpenAI API key in the sidebar
    2. Upload one or more PDF files
    3. Click "Process PDFs"
    4. Start chatting about your documents!
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
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
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
                    # Get response from conversation chain
                    response = st.session_state.conversation_chain({
                        "question": prompt
                    })
                    
                    answer = response["answer"]
                    
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
