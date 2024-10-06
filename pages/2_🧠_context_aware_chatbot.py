import utils
import streamlit as st
from streaming import StreamHandler

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Context aware chatbot", page_icon="🧠")
st.header('Context Aware Chatbot')
st.write('Enhancing Chatbot Interactions through Context Awareness')

class ContextChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
    
    @st.cache_resource
    def setup_chain(_self):
        memory = ConversationBufferMemory()
        chain = ConversationChain(llm=_self.llm, memory=memory, verbose=False)
        return chain
    
    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    {"input":user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["response"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(ContextChatbot, user_query, response)

if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
