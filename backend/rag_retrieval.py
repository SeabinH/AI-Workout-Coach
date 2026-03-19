import os

from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import OPENAI_API_KEY

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(persist_directory=os.path.join(BASE_DIR, "db"), embedding_function= embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key = OPENAI_API_KEY)

    prompt_template = """
    You are a professional fitness coach.

    Use the following context to answer the question.
    If you don't know, give your best helpful fitness advice anyway.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = retriever, chain_type_kwargs= {'prompt': PROMPT})

    return qa_chain