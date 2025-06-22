import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import tempfile
import os

st.set_page_config(page_title="AI Baza Wiedzy", layout="wide")

st.title("📚 AI Baza Wiedzy dla Twojej firmy")
st.markdown("Wgraj plik PDF i zadawaj pytania do jego treści.")

# Wybór i wczytanie pliku PDF
uploaded_file = st.file_uploader("Wgraj dokument PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmppath = tmp.name

    try:
        loader = PyPDFLoader(tmppath)
        documents = loader.load()

        # Dziel tekst na fragmenty
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        # Utwórz wektorową bazę danych
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)

        # Interfejs zadawania pytań
        st.success("📄 Plik załadowany i przetworzony.")
        query = st.text_input("Zadaj pytanie dotyczące dokumentu:")

        if query:
            chain = load_qa_chain(ChatOpenAI(model_name="gpt-4", temperature=0), chain_type="stuff")
            matching_docs = db.similarity_search(query)
            answer = chain.run(input_documents=matching_docs, question=query)
            st.markdown("### 📌 Odpowiedź:")
            st.write(answer)

    except Exception as e:
        st.error(f"❌ Błąd przy analizie pliku: {e}")

