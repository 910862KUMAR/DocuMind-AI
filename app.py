import streamlit as st

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.qa_chain import get_qa_chain
from utils.helpers import save_uploaded_file

st.set_page_config(page_title="DocuMind AI")

st.title("📚 DocuMind AI")
st.write("Upload a PDF and ask questions")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        file_path = save_uploaded_file(uploaded_file)

        documents = load_pdf(file_path)

        chunks = split_documents(documents)

        embeddings = get_embeddings()

        db = create_vector_store(
            chunks,
            embeddings
        )

        retriever = get_retriever(db)

        llm = get_qa_chain(retriever)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question")

    if question:

        with st.spinner("Generating answer..."):

            docs = retriever.invoke(question)

            context = "\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
            Answer the question based on the context below.

            Context:
            {context}

            Question:
            {question}
            """

            response = llm.invoke(prompt)

        st.subheader("Answer")

        st.write(response.content)