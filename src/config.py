import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Local development
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Streamlit Cloud
if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

LLM_MODEL = "gemini-2.5-flash"
