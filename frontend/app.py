import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from backend.rag_retrieval import build_chain

# @st.cache_resource
# def load_chain():
#     from backend.rag_retrieval import qa_chain
#     return qa_chain

# qa_chain = load_chain()

qa_chain = build_chain()

st.set_page_config(page_title="AI Workout Coach", layout="centered")

st.title("💪 AI Workout Coach")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your workout coach anything..."):
    
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    response = qa_chain.invoke(prompt)
    answer = response["result"]

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save AI response
    st.session_state.messages.append({"role": "assistant", "content": answer})