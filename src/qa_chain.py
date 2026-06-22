from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_API_KEY, LLM_MODEL


def get_qa_chain(retriever):

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    return llm
