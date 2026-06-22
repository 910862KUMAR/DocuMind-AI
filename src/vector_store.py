from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DB_DIR

def create_vector_store(chunks, embeddings):

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    return db