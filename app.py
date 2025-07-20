import streamlit as st
import os
import chromadb
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")
st.title("RAG Chatbot")

MODEL = "deepseek-coder"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state or st.session_state.get("prev_context_size") != 16384:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.prev_context_size = 16384

llm = ChatOllama(model=MODEL, streaming=True)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

pdf_files_path = "/Users/rou/RAG-Chatbot/files/"

def PDFLoader(pdf_dir):
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

all_pdfs = PDFLoader(pdf_files_path)

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

def trim_memory():
    while len(st.session_state.chat_history) > 10 * 2:
        st.session_state.chat_history.pop(0)

custom_prompt_template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know, say you don't know.

{context}

Question: {question}
Helpful Answer:"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_prompt_template,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

if user_input := st.chat_input("Ask something"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    trim_memory()

    with st.chat_message("assistant"):
        response_container = st.empty()
        try:
            print("Prompt received:", user_input)
            retrieve_docs = retriever.invoke(user_input)
            print("Documents retrieved:", retrieve_docs)
            if not retrieve_docs:
                response = "No relevant documents found."
            else:
                print("Passing to QA chain...")
                qa_result = qa.invoke({"query": user_input})
                print("QA chain result:", qa_result)
                response = qa_result.get("result", "No response generated.")
            print("Response:", response)

            response_container.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            trim_memory()

        except Exception as e:
            print("Error during response generation:", e)
            response_container.markdown(f"**Error:** {e}")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
