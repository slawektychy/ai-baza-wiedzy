# app.py
import streamlit as st
import os
import hashlib
import tempfile
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader

st.set_page_config(page_title="AI Baza Wiedzy", layout="wide")

st.title("📚 AI Baza Wiedzy dla Twojej firmy")
openai.api_key = st.secrets["OPENAI_API_KEY"]

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# Upload PDF i przetwarzanie
with st.sidebar:
    st.header("📄 Wgraj dokument")
    uploaded_file = st.file_uploader("PDF z dokumentacją", type="pdf")

    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        save_path = os.path.join(UPLOAD_DIR, file_hash + ".pdf")

        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(file_bytes)

            # Przetwarzanie PDF → tekst → embeddingi
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = os.path.join(tmpdir, "tmp.pdf")
                with open(tmppath, "wb") as f:
                    f.write(file_bytes)
                loader = PyMuPDFLoader(tmppath)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
                chunks = splitter.split_documents(documents)

                embed = OpenAIEmbeddings()
                vectordb = FAISS.from_documents(chunks, embed)

                if st.session_state.vectordb:
                    st.session_state.vectordb.merge_from(vectordb)
                else:
                    st.session_state.vectordb = vectordb

            st.success("✅ Dokument został dodany do bazy wiedzy!")
        else:
            st.info("⚠️ Ten dokument został już wcześniej załadowany.")

# Czat
st.subheader("💬 Zadaj pytanie dotyczące dokumentów")

query = st.text_input("Twoje pytanie:", placeholder="np. Jak wystawić fakturę w Enova?")
if query and st.session_state.vectordb:
    llm = ChatOpenAI(model_name="gpt-4o")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.vectordb.as_retriever())
    response = qa.run(query)
    st.write(response)
elif query:
    st.warning("📂 Najpierw wgraj przynajmniej jeden dokument PDF.")
