import os
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_groq import ChatGroq
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings






logger = get_logger('Langchain-Chatbot')

# Load the Groq API key from the secrets file
groq_api_key = st.secrets["GROQ_API_KEY"]





#decorator
def enable_chat_history(func):
    # to clear chat history after switching chatbot
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
        except:
            pass

    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_llm():
    available_llms = ["deepseek-r1", "llama-4-scout-17b"]
    llm_opt = st.sidebar.radio(
        label="LLM",
        options=available_llms,
        key="SELECTED_LLM"
    )

    if llm_opt == "deepseek-r1":
        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=groq_api_key,
            model_name="deepseek-r1-distill-llama-70b",
            streaming=True
        )
    elif llm_opt == "llama-4-scout-17b":
        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=groq_api_key,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            streaming=True
        )
    return llm

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    return embedding_model



def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v



