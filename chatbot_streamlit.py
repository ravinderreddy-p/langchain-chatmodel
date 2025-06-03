import os
import tempfile
from langchain_openai import OpenAI, OpenAIEmbeddings
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("PDF Chatbot with Langchain")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None and st.button("Process PDF"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    st.write("Extracting and Embedding content...")
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
    retriever = vector_store.as_retriever()

    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever = retriever,
        return_source_documents = False
    )

    st.success("PDF processed, you can ask questions now")
    os.unlink(tmp_path)

if st.session_state.qa_chain:
    user_question = st.text_input("Ask a question about the PDF")
    if user_question:
        with st.spinner("Generating Answer.."):
            answer = st.session_state.qa_chain.run(user_question)
            st.markdown(f"**Answer: ** {answer}")
    else:
        st.info("Please upload and process a PDF to enable question answering.")