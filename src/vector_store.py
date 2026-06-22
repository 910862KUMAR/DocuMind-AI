from langchain_community.vectorstores import Chroma


def create_vector_store(chunks, embeddings):

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return db
