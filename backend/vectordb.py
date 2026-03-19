import json
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import OPENAI_API_KEY

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Load text documents
documentsLoader = DirectoryLoader('./data/informative_texts',glob="**/*.txt", loader_cls= TextLoader)
documents = documentsLoader.load()

# Load JSON exercises
with open("data/exercises/exercises.json") as f:
    exercises = json.load(f)

exercise_docs = []
for ex in exercises:
    text = f"""
            Exercise: {ex['name']}
            Primary Muscles: {', '.join(ex['primaryMuscles'])}
            Equipment: {ex['equipment']}
            Instructions: {' '.join(ex['instructions'])}
            """
    exercise_docs.append(Document(page_content=text))

# Combine and split into chunks
all_docs = documents + exercise_docs
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunked_docs = splitter.split_documents(all_docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build vector DB
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory=os.path.join(BASE_DIR, "db")
)

vectorstore.persist()
print("Vector database created!")