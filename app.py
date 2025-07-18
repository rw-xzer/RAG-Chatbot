import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")
st.title("RAG Chatbot")

MODEL = "deepseek-coder:6.7b"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state or st.session_state.get("prev_context_size") != 16384:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.prev_context_size = 16384

llm = ChatOllama(model=MODEL, streaming=True)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

pdf_files_path = "C:/Users/admin/RAG-Chatbot/pdf_files/"

print("Loading data...")
print(os.listdir(pdf_files_path))

loaders = [PyPDFLoader(os.path.join(pdf_files_path, fn)) for fn in os.listdir(pdf_files_path)]
print(loaders)

def PDFLoader():
    docs = []
    for loader in loaders:
        print("Loading raw document..." + loader.file_path)
        docs = loader.load()
    return docs

all_pdfs = PDFLoader()

llm = ChatOllama(model="deepseek-coder:6.7b" )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

splited_documents = text_splitter.split_documents(all_pdfs)
print(splited_documents)

persist_directory = "./chromadb"

vectorstore = Chroma.from_documents(
    documents=splited_documents,
    embedding=embeddings,
    persist_directory=persist_directory
)

vectorstore.persist()
print("Data stored in Chromadb")

retriever = vectorstore.as_retriever(search_type="similarity")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def trim_memory():
    while len(st.session_state.chat_history) > 10 * 2:
        st.session_state.chat_history.pop(0)

if prompt := st.chat_input("Ask something"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    trim_memory()

    with st.chat_message("assistant"):
        response_container = st.empty()

        retrieve_docs = retriever.get_relevant_documents(prompt)
        response = (
            "No relevant documents found." if not retrieve_docs
            else qa({"query": prompt}).get("result", "No response generated.")
        )

        response_container.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        trim_memory()