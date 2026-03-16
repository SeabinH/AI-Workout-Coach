from fastapi import FastAPI
from backend.rag_retrieval import qa_chain

app = FastAPI()

@app.post("/ask")
def ask(question: str):
    answer = qa_chain.invoke(question)
    return {"answer": answer}