from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from backend.config import OPENAI_API_KEY

vectorstore = Chroma(persist_directory="../db", embedding_function= None)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key = OPENAI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = retriever)