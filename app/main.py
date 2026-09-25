"""
app/main.py
Streamlit chat UI for the product recommendation chatbot.
Wraps the dialogue/intent/retrieval pipeline in a conversational interface.
"""

import sys
import os

# Allow importing from src/ when running via `streamlit run app/main.py`
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from dialogue import ConversationState, handle_turn
from retrieval import ProductRetriever


st.set_page_config(
    page_title="Virtual Sales Associate",
    page_icon="🛍️",
    layout="centered",
)

st.title("🛍️ Virtual Sales Associate")
st.caption(
    "Ask me for product recommendations or comparisons — phones, laptops, "
    "clothing, or footwear. Try: *\"suggest shirts for a Goa trip\"* "
    "or *\"compare iPhone 16 Pro and Galaxy S25 Ultra\"*."
)


@st.cache_resource
def get_retriever():
    """Load the embedding model + build the FAISS index once per app session."""
    return ProductRetriever()


if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = ConversationState()

if "messages" not in st.session_state:
    st.session_state.messages = []


retriever = get_retriever()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("What are you shopping for today?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = handle_turn(
                user_input,
                st.session_state.conversation_state,
                retriever,
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant recommends products from a 480-item catalog "
        "spanning smartphones, laptops, clothing, and footwear, using "
        "hybrid retrieval (semantic search + filters) and LLM reasoning."
    )
    if st.button("🔄 Reset conversation"):
        st.session_state.conversation_state = ConversationState()
        st.session_state.messages = []
        st.rerun()