import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

st.set_page_config(layout="wide")
load_dotenv()
OPENAI_API_KEY = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
if OPENAI_API_KEY:
    st.sidebar.success("you can chat")
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Ask follow-up questions if needed."}
    ]

existing_collections = [col.name for col in qdrant_client.get_collections().collections]
st.sidebar.title("\U0001F4C1 Upload & Settings")

collection_mode = st.sidebar.radio("Collection Mode", ["Use Existing", "Create New"])

if collection_mode == "Use Existing":
    selected_collection = (
        st.sidebar.selectbox("Select a collection", existing_collections)
        if existing_collections else None
    )
    if not existing_collections:
        st.sidebar.warning("No existing collections found.")
else:
    selected_collection = st.sidebar.text_input("Enter new collection name")

chunk_size = st.sidebar.number_input("Chunk Size", value=1000)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", value=200)
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
upload_btn = st.sidebar.button("Submit")

if uploaded_file and upload_btn:
    if not selected_collection:
        st.error("\u274C Please enter a collection name.")
    else:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            split_docs = splitter.split_documents(docs)

            if selected_collection not in existing_collections:
                qdrant_client.create_collection(
                    collection_name=selected_collection,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
                )

            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            QdrantVectorStore.from_documents(
                split_docs,
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=selected_collection,
                embedding=embeddings
            )

            os.remove(tmp_path)
            st.success(f"\u2705 PDF uploaded and embedded into `{selected_collection}` collection.")
elif upload_btn and not uploaded_file:
    st.warning("\u26A0\uFE0F Please upload a PDF file before submitting.")

with st.container():
    col1, col2 = st.columns([6, 1])
    with col1:
        query = st.text_input("Ask a question", placeholder="Ask a question based on the selected collection", label_visibility="collapsed", key="chat_input")
    with col2:
        go_btn = st.button("Go", use_container_width=True, key="go_button")


if go_btn and query and selected_collection:
    with st.spinner("Searching and generating answer..."):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectordb = QdrantVectorStore.from_existing_collection(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=selected_collection,
            embedding=embeddings
        )

        results = vectordb.similarity_search(query=query)
        context = "\n\n\n".join([
            f"Page Content: {r.page_content}\nPage Number: {r.metadata.get('page_label', '?')}\nFile: {r.metadata.get('source', '?')}"
            for r in results
        ])

        st.session_state.messages.append({"role": "user", "content": query})

        system_prompt = {
            "role": "system",
            "content": f"""Use the context below to answer the user's question,don't answer anything out of the context

        Context:
        {context}

        Only answer based on the context. Point the user to the correct page number if needed."""
                }

        chat = ChatOpenAI(model="gpt-4.1-mini")
        response = chat.invoke([system_prompt] + st.session_state.messages[1:])
        st.session_state.messages.append({"role": "assistant", "content": response.content})


if st.button("Clear Conversation"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Ask follow-up questions if needed."}
    ]
    st.success("Conversation cleared.")

st.markdown("### Conversation")

def render_message(role, content):
    if role == "user":
        align = "right"
        bg_color = "#DCF8C6"
        sender = " You"
    else:
        align = "left"
        bg_color = "#F1F0F0"
        sender = " Bot"
    html = f"""
    <div style='display: flex; justify-content: {align}; margin: 10px 0;'>
        <div style='background-color: {bg_color}; padding: 10px 15px; border-radius: 15px; max-width: 75%;'>
            <strong>{sender}:</strong><br>{content}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

for msg in reversed(st.session_state.messages[1:]):  # reverse to show latest on top
    render_message(msg["role"], msg["content"])