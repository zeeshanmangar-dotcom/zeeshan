import os
import streamlit as st
from typing import List
from PyPDF2 import PdfReader

from openai import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import AIMessage, HumanMessage

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="PDF Chatbot (RAG – HF Router)",
    page_icon="📄🤖",
    layout="wide"
)

# =========================
# Hugging Face OpenAI-Compatible Client
# =========================
def get_hf_client(hf_token: str) -> OpenAI:
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

# =========================
# Custom Embeddings (HF via OpenAI-compatible API)
# =========================
class HFEmbeddings(Embeddings):
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            resp = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(resp.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return resp.data[0].embedding

# =========================
# Helper Functions
# =========================
def extract_text_from_pdfs(pdf_files: List) -> str:
    all_text = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    return "\n".join(all_text)


def build_vectorstore(text: str, client: OpenAI) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    embeddings = HFEmbeddings(
        client=client,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings)


def generate_answer(
    client: OpenAI,
    context_docs: List[str],
    chat_history: List,
    question: str
) -> str:
    system_prompt = (
        "You are a helpful AI assistant. Answer strictly using the provided context. "
        "If the answer is not in the context, say you don't know."
    )

    context = "\n\n".join(context_docs)

    messages = [{"role": "system", "content": system_prompt}]
    for user, bot in chat_history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{question}"
    })

    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=messages
    )

    return completion.choices[0].message.content

# =========================
# Session State
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "client" not in st.session_state:
    st.session_state.client = None

# =========================
# Sidebar
# =========================
st.sidebar.title("🔑 HuggingFace API")

hf_token = st.sidebar.text_input(
    "Enter HF_TOKEN",
    type="password",
    help="Your HuggingFace access token"
)

st.sidebar.markdown("---")

pdf_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

process_btn = st.sidebar.button("Process PDFs")

# =========================
# Main UI
# =========================
st.title("📄🤖 Chat with PDFs (RAG – HuggingFace Router)")
st.markdown(
    """
**Instructions**
1. Enter your HuggingFace `HF_TOKEN`
2. Upload one or more PDFs
3. Click **Process PDFs**
4. Ask questions or request summaries/explanations
"""
)

# =========================
# Process PDFs
# =========================
if process_btn:
    if not hf_token:
        st.error("❌ HF_TOKEN is required.")
    elif not pdf_files:
        st.error("❌ Upload at least one PDF.")
    else:
        with st.spinner("Processing PDFs..."):
            os.environ["HF_TOKEN"] = hf_token
            client = get_hf_client(hf_token)

            raw_text = extract_text_from_pdfs(pdf_files)
            if not raw_text.strip():
                st.error("❌ No text found in PDFs.")
            else:
                st.session_state.vectorstore = build_vectorstore(raw_text, client)
                st.session_state.client = client
                st.session_state.chat_history = []
                st.success("✅ PDFs indexed successfully.")

# =========================
# Chat Interface
# =========================
st.markdown("---")
st.subheader("💬 Chat")

for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

user_input = st.chat_input("Ask a question about the PDFs...")

if user_input:
    if st.session_state.vectorstore is None:
        st.error("❌ Please process PDFs first.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            docs = st.session_state.vectorstore.similarity_search(user_input, k=4)
            context_docs = [d.page_content for d in docs]

            answer = generate_answer(
                client=st.session_state.client,
                context_docs=context_docs,
                chat_history=st.session_state.chat_history,
                question=user_input
            )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append((user_input, answer))
