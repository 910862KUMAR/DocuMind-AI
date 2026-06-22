import shutil
import os
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DB_DIR


def create_vector_store(chunks, embeddings):

    # Delete old database before creating a new one
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    return db
