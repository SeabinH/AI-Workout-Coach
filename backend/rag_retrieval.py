import os

from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import OPENAI_API_KEY

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory=os.path.join(BASE_DIR, "db"), embedding_function= embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key = OPENAI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = retriever)


if __name__ == "__main__":
    result = qa_chain.invoke("What are the primary muscles for bench press?")
    print(result)