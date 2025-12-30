import os
import streamlit as st
from typing import List
from PyPDF2 import PdfReader
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="PDF Chatbot (HF RAG)",
    page_icon="📄🤖",
    layout="wide"
)

# =========================
# HF OpenAI-Compatible Client
# =========================
def get_client(hf_token: str) -> OpenAI:
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

# =========================
# HF Embeddings (NO sentence-transformers)
# =========================
class HFEmbeddings(Embeddings):
    def __init__(self, client: OpenAI):
        self.client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model="intfloat/e5-small-v2",
            input=texts
        )
        return [e.embedding for e in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="intfloat/e5-small-v2",
            input=text
        )
        return response.data[0].embedding

# =========================
# PDF Processing
# =========================
def extract_text(files) -> str:
    text = []
    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text.append(page.extract_text())
    return "\n".join(text)

def build_vectorstore(text: str, client: OpenAI):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    embeddings = HFEmbeddings(client)
    return FAISS.from_texts(chunks, embeddings)

# =========================
# RAG Answer
# =========================
def answer_question(client, context, history, question):
    messages = [
        {
            "role": "system",
            "content": "Answer ONLY using the provided context. If the answer is not present, say you don't know."
        }
    ]

    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

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
if "history" not in st.session_state:
    st.session_state.history = []
if "client" not in st.session_state:
    st.session_state.client = None

# =========================
# Sidebar
# =========================
st.sidebar.title("🔑 HuggingFace Token")
hf_token = st.sidebar.text_input("HF_TOKEN", type="password")

pdfs = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process PDFs"):
    if not hf_token or not pdfs:
        st.error("HF_TOKEN and PDFs required")
    else:
        with st.spinner("Processing PDFs..."):
            client = get_client(hf_token)
            text = extract_text(pdfs)

            if not text.strip():
                st.error("No text found in PDFs.")
            else:
                st.session_state.vectorstore = build_vectorstore(text, client)
                st.session_state.client = client
                st.session_state.history = []
                st.success("PDFs indexed successfully!")

# =========================
# Main UI
# =========================
st.title("📄🤖 Chat with PDFs (RAG)")

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

query = st.chat_input("Ask a question about the PDFs...")

if query:
    if not st.session_state.vectorstore:
        st.error("Please upload and process PDFs first.")
    else:
        with st.spinner("Thinking..."):
            docs = st.session_state.vectorstore.similarity_search(query, k=4)
            context = "\n\n".join(d.page_content for d in docs)

            answer = answer_question(
                st.session_state.client,
                context,
                st.session_state.history,
                query
            )

        st.session_state.history.append((query, answer))

        with st.chat_message("assistant"):
            st.markdown(answer)
