import streamlit as st
from backend.rag_retrieval import qa_chain

st.title("AI Workout Coach")

question = st.text_input("Ask your workout coach anything:")

if question:
    answer = qa_chain.invoke(question)
    st.write(answer)