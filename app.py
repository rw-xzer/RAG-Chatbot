import streamlit as st
import os
import chromadb
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from chromadb.config import Settings

from dotenv import load_dotenv

load_dotenv()

client = chromadb.PersistentClient(path="/Users/rou/RAG-Chatbot/chromadb")
collection = client.get_or_create_collection(name="chroma.sqlite3")

collection.delete(where={"source": "chromadb"})

st.set_page_config(layout="wide")
st.title("RAG Chatbot")

MODEL = "deepseek-r1:7b"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state or st.session_state.get("prev_context_size") != 16384:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.prev_context_size = 16384

llm = ChatOllama(model=MODEL, streaming=True, temperature=0.6)
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
    chunk_size=1500,
    chunk_overlap=950,
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

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate 3
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by new lines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vectorstore.as_retriever(), llm, prompt=QUERY_PROMPT
)

def trim_memory():
    while len(st.session_state.chat_history) > 10 * 2:
        st.session_state.chat_history.pop(0)

custom_prompt_template = """
Answer the question only with the following context. 
Context: {context}
Question: {question}
If the context does not answer the question, just say "I don't have enough information to answer that."
If you are unsure, just say "I don't have enough information to answer that."
When possible always directly use the relevant part of the context to answer the question.
Figures refer to images which should be explained in words when referred to.
Assume that all grammar and spelling mistakes in the context are intentional and should not be corrected.
Always be truthful and clear while using simple language.
"""

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
