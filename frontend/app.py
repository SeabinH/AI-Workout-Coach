import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from backend.rag_retrieval import qa_chain

@st.cache_resource
def load_chain():
    from backend.rag_retrieval import qa_chain
    return qa_chain

qa_chain = load_chain()

st.set_page_config(page_title="AI Workout Coach")

st.title("💪 AI Workout Coach")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask your workout coach anything:")

if st.button("Ask") and question:
    response = qa_chain.invoke(question)
    answer = response["result"]

    st.session_state.history.append((question, answer))

for q, a in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Coach:** {a}")