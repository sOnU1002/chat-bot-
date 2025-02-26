import streamlit as st
import os
from dotenv import load_dotenv
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

st.title("CDP Support Agent Chatbot")

# Define the LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)

# Prompt for chatbot
prompt = ChatPromptTemplate.from_template(
    """
    You are a Support Agent Chatbot for Customer Data Platforms (CDPs).
    You help users by providing accurate information from official documentation.
    Answer only based on the provided context and do not generate general responses.
    
    <context>
    {context}
    <context>
    
    Question: {input}
    """
)

# CDP Documentation Links
documentation_links = {
    "Segment": "https://segment.com/docs/?ref=nav",
    "mParticle": "https://docs.mparticle.com/",
    "Lytics": "https://docs.lytics.com/",
    "Zeotap": "https://docs.zeotap.com/home/en-us/"
}

# Function to process documentation
def vector_embedding(cdp_name, cdp_link):
    if f"vectors_{cdp_name}" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = WebBaseLoader(cdp_link)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state[f"vectors_{cdp_name}"] = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Dropdown to select CDP
cdp_name = st.selectbox("Select a CDP", list(documentation_links.keys()))

# Load documentation button
if st.button("Load Documentation"):
    vector_embedding(cdp_name, documentation_links[cdp_name])
    st.write(f"Documentation for {cdp_name} has been embedded.")

# Input for user question
question = st.text_input("Enter Your How-to Question")

if question and cdp_name:
    vector_key = f"vectors_{cdp_name}"
    if vector_key in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state[vector_key].as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': question})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # Show document sources
        with st.expander("Relevant Documentation Excerpts"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write(f"Please load the documentation for {cdp_name} first.")
