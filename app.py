import os
import streamlit as st
from typing import List
from PyPDF2 import PdfReader

from openai import OpenAI

# ✅ FIXED IMPORT (LangChain v0.2+)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="PDF Chatbot (RAG – HF Router)",
    page_icon="📄🤖",
    layout="wide"
)

# =========================
# HuggingFace OpenAI-Compatible Client
# =========================
def get_hf_client(hf_token: str) -> OpenAI:
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

# =========================
# Custom Embeddings via HF Router
# =========================
class HFEmbeddings(Embeddings):
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            resp = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            vectors.append(resp.data[0].embedding)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return resp.data[0].embedding

# =========================
# PDF Utilities
# =========================
def extract_text_from_pdfs(pdf_files: List) -> str:
    text_data = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_data.append(text)
    return "\n".join(text_data)

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

def generate_answer(client: OpenAI, context: List[str], history, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Answer ONLY using the provided context. If not found, say you don't know."
        }
    ]

    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    messages.append({
        "role": "user",
        "content": f"Context:\n{chr(10).join(context)}\n\nQuestion:\n{question}"
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
hf_token = st.sidebar.text_input("Enter HF_TOKEN", type="password")

st.sidebar.markdown("---")
pdf_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

process_btn = st.sidebar.button("Process PDFs")

# =========================
# Main UI
# =========================
st.title("📄🤖 Chat with PDFs (RAG – HuggingFace)")
st.markdown("""
**Steps**
1. Enter HuggingFace `HF_TOKEN`
2. Upload PDFs
3. Click **Process PDFs**
4. Ask questions, summaries, or explanations
""")

# =========================
# Process PDFs
# =========================
if process_btn:
    if not hf_token:
        st.error("HF_TOKEN is required.")
    elif not pdf_files:
        st.error("Upload at least one PDF.")
    else:
        with st.spinner("Indexing PDFs..."):
            os.environ["HF_TOKEN"] = hf_token
            client = get_hf_client(hf_token)

            text = extract_text_from_pdfs(pdf_files)
            if not text.strip():
                st.error("No readable text found.")
            else:
                st.session_state.vectorstore = build_vectorstore(text, client)
                st.session_state.client = client
                st.session_state.chat_history = []
                st.success("PDFs processed successfully!")

# =========================
# Chat UI
# =========================
st.markdown("---")
st.subheader("💬 Chat")

for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

query = st.chat_input("Ask about the PDFs...")

if query:
    if st.session_state.vectorstore is None:
        st.error("Please process PDFs first.")
    else:
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            docs = st.session_state.vectorstore.similarity_search(query, k=4)
            context = [d.page_content for d in docs]

            answer = generate_answer(
                client=st.session_state.client,
                context=context,
                history=st.session_state.chat_history,
                question=query
            )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append((query, answer))
