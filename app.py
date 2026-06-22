import streamlit as st

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.qa_chain import get_qa_chain
from utils.helpers import save_uploaded_file

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

# ---------------- SESSION STATE ---------------- #

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- SIDEBAR ---------------- #

with st.sidebar:
    st.title("📚 DocuMind AI")

    st.markdown("""
    ### Features
    ✅ PDF Upload  
    ✅ Semantic Search  
    ✅ Gemini Powered  
    ✅ RAG Pipeline  
    """)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []

    st.markdown("---")
    st.caption("Built with LangChain + ChromaDB + Gemini")

# ---------------- MAIN UI ---------------- #

st.title("📄 Chat with Your PDF")
st.write("Upload a PDF and ask questions in natural language.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# ---------------- PROCESS PDF ---------------- #

if uploaded_file:

    # Process only when a new file is uploaded
    if st.session_state.current_file != uploaded_file.name:

        with st.spinner("Processing PDF..."):

            file_path = save_uploaded_file(uploaded_file)

            documents = load_pdf(file_path)

            chunks = split_documents(documents)

            embeddings = get_embeddings()

            db = create_vector_store(
                chunks,
                embeddings
            )

            st.session_state.retriever = get_retriever(db)

            st.session_state.llm = get_qa_chain(
                st.session_state.retriever
            )

            st.session_state.current_file = uploaded_file.name

        st.success("✅ PDF processed successfully!")

# ---------------- ASK QUESTION ---------------- #

if st.session_state.retriever:

    question = st.chat_input(
        "Ask something about your document..."
    )

    if question:

        with st.spinner("Generating answer..."):

            docs = st.session_state.retriever.invoke(
                question
            )

            context = "\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
            Answer ONLY from the provided context.

            Context:
            {context}

            Question:
            {question}

            If the answer is not found in the context,
            say:

            'The information is not available in the uploaded document.'
            """

            response = st.session_state.llm.invoke(
                prompt
            )

            answer = response.content

            st.session_state.chat_history.append(
                ("You", question)
            )

            st.session_state.chat_history.append(
                ("DocuMind AI", answer)
            )

# ---------------- DISPLAY CHAT ---------------- #

for sender, message in st.session_state.chat_history:

    with st.chat_message(
        "user" if sender == "You" else "assistant"
    ):
        st.markdown(message)
