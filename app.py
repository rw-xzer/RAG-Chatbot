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

#deepseek-r1:7b
MODEL = "deepseek-r1:1.5b"

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

custom_prompt_template = """
Answer the question only based on the following context if it is relevant to the question.
Context: {context}
Question: {question}
If the context is not relevant to the question, answer based on your own knowledge.
Always be truthful and clear. If you don't know the answer, say that you don't know and do not fabricate an answer.
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

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask something"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        response_placeholder.markdown("Thinking...")
        
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
            
            response_placeholder.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            trim_memory()
            
        except Exception as e:
            print("Error during response generation:", e)
            error_response = f"**Error:** {e}"
            response_placeholder.markdown(error_response)
            st.session_state.chat_history.append({"role": "assistant", "content": error_response})
