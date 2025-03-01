# import os
# import json
# import streamlit as st

# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA

# # Set working directory
# working_dir = os.getcwd()

# # Load API Key from config.json
# config_data = json.load(open(f"{working_dir}/config.json"))
# GROQ_API_KEY = config_data["GROQ_API_KEY"]
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# # Load embedding model
# embedding = HuggingFaceEmbeddings()

# # Load LLM from Groq
# llm = ChatGroq(
#     model="deepseek-r1-distill-llama-70b",
#     temperature=0
# )


# def process_document_to_chroma_db(file_name):
#     """Process PDF, split text, and store in ChromaDB"""
#     loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
#     documents = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
#     texts = text_splitter.split_documents(documents)

#     # Check if the collection exists
#     try:
#         vectordb = Chroma(persist_directory=f"{working_dir}/doc_vectorstore", embedding_function=embedding)
#         existing_collections = vectordb._client.list_collections()

#         if "doc_vectorstore" in [col.name for col in existing_collections]:
#             print("Collection already exists. Skipping creation.")
#             return vectordb  # Return existing collection
        
#         # Create a new collection if not exists
#         vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=f"{working_dir}/doc_vectorstore")
    
#     except Exception as e:
#         print(f"Error processing document: {e}")
    
#     return vectordb


# def answer_question(user_question):
#     """Retrieve answer from the stored document using DeepSeek-R1"""
#     vectordb = Chroma(
#         persist_directory=f"{working_dir}/doc_vectorstore",
#         embedding_function=embedding
#     )

#     retriever = vectordb.as_retriever()

#     # Create a QA Chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#     )

#     response = qa_chain.invoke({"query": user_question})
#     answer = response["result"]

#     return answer


# # 🔵 Streamlit UI
# st.set_page_config(page_title="DeepSeek-R1 - Multi-Document RAG", layout="wide")

# st.title("🐋 DeepSeek-R1 - Multi-Document RAG")

# # File uploader for multiple PDFs
# uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         save_path = os.path.join(working_dir, uploaded_file.name)

#         # Save file
#         with open(save_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Process document
#         process_document_to_chroma_db(uploaded_file.name)

#     st.success("✅ Documents uploaded and processed successfully!")

# # User input
# user_question = st.text_area("Ask your question about the documents")

# if st.button("Get Answer"):
#     answer = answer_question(user_question)
    
#     st.markdown("### 💡 DeepSeek-R1 Response")
#     st.markdown(answer)


import os
import json
import streamlit as st

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set working directory
working_dir = os.getcwd()

# Load API Key from config.json
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load embedding model
embedding = HuggingFaceEmbeddings()

# Load LLM from Groq
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0
)

# Define the prompt template
transport_prompt = PromptTemplate.from_template(
    "I need the cheapest transport option for shipping from {from_location} to {to_location}. "
    "The delivery must be within {days} days, and the shipment weighs {weight} kg. "
    "Return ONLY the price and transport mode in this exact format: 'Price: $XXX, Transport: YYY'. "
    "No additional information. If no exact match is found, provide the closest available option in the same format."
    " you want give any random value or  closest available if you do not know it  "
)

def process_document_to_chroma_db(file_name):
    """Process PDF, split text, and store in ChromaDB"""
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    try:
        vectordb = Chroma(persist_directory=f"{working_dir}/doc_vectorstore", embedding_function=embedding)
        existing_collections = vectordb._client.list_collections()

        if "doc_vectorstore" in [col.name for col in existing_collections]:
            print("Collection already exists. Skipping creation.")
            return vectordb 
        
        vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=f"{working_dir}/doc_vectorstore")
    except Exception as e:
        print(f"Error processing document: {e}")
    
    return vectordb

def answer_question(user_question):
    """Retrieve answer from the stored document using DeepSeek-R1"""
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer

# 🔵 Streamlit UI
st.set_page_config(page_title="DeepSeek-R1 - Multi-Document RAG", layout="wide")

st.title("🐋 DeepSeek-R1 - Multi-Document RAG")

# File uploader for multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(working_dir, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        process_document_to_chroma_db(uploaded_file.name)

    st.success("✅ Documents uploaded and processed successfully!")

# User input
user_question = st.text_area("Ask your question about the documents")

if st.button("Get Answer"):
    answer = answer_question(user_question)
    
    st.markdown("### 💡 DeepSeek-R1 Response")
    st.markdown(answer)
