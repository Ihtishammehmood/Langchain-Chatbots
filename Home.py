import streamlit as st

st.set_page_config(
    page_title="Langchain Chatbot",
    page_icon='🏠',
    layout='wide'
)

st.header("A Multifaceted Chatbot")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/Ihtishammehmood/Langchain-Chatbots.git)
[![linkedin ](https://img.shields.io/badge/Ihtisham%20Mehmood-blue?logo=linkedin&color=gray)](https://www.linkedin.com/in/ihtishammehmood/)

""")
st.write("""
A robust AI-powered chatbot using LangChain that offers a range of powerful features to cater to various user needs:

- **Basic Chatbot**:A simple, conversational chatbot that allows users to interact and ask general questions with real-time responses.
- **Context aware chatbot**:he bot retains conversational context, making interactions more natural and coherent, providing relevant follow-up responses based on the user's previous
- **Chatbot with Internet Access**:The bot can access real-time information from the web, ensuring users get up-to-date answers by searching the internet when necessary.
- **Chat with your documents**: Users can upload documents (PDFs, text files, etc.), and the chatbot will parse the content, allowing for interactive Q&A based on the document's contents.
- **Chat with SQL database**: The bot can query SQL databases directly, making it easy for users to retrieve or manipulate data through simple conversational commands.
- **Chat with Websites**: Enable chatbot to interact and answer questions based on a site's content, enabling efficient and detailed information extraction from web pages.

This chatbot combines multiple advanced capabilities, making it highly adaptable and useful across various scenarios—from casual conversations to complex data retrieval tasks.
""")