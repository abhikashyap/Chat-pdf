from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from langchain_openai import OpenAIEmbeddings
file_path = "python.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_docs = text_splitter.split_documents(documents=docs)


embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
QdrantVectorStore.from_documents(splitted_docs,
                                 url="http://localhost:6333",
                                 collection_name="learning_vectors",
                                 embedding=embeddings_model)
print("upload done")