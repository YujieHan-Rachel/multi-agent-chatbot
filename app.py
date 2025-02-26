import os
import openai
import pinecone
import streamlit as st
from agents import Head_Agent

# Streamlit setup
st.set_page_config(page_title="Multi-Agent Chatbot", layout="wide")

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "miniproject2-multi-agent-chatbot"

# Check if API keys are set
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API keys are missing! Please set `OPENAI_API_KEY` and `PINECONE_API_KEY` in Hugging Face Secrets.")
    st.stop()

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone with exception handling
from pinecone import Pinecone, ServerlessSpec

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        st.warning(f"⚠️ Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # Ensure it matches your embedding model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
    st.stop()

# Initialize the chatbot agent
if "chatbot_agent" not in st.session_state:
    st.session_state.chatbot_agent = Head_Agent(openai.api_key, pinecone_index)

# Sidebar UI
with st.sidebar:
    st.title("🔧 Chatbot Settings")
    st.markdown("This is an AI-powered multi-agent chatbot using OpenAI and Pinecone.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# Main UI
st.title("🤖 Multi-Agent Chatbot")
st.markdown("Ask me anything, and I'll retrieve relevant information and generate an intelligent response.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input with improved text area
user_input = st.text_area("💬 Your question:", height=100)

if st.button("Send"):
    if user_input:
        # Display user input in chat
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process the user query using the chatbot agent
        chatbot_response = st.session_state.chatbot_agent.process_query(user_input)

        # Store chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

        # Display chatbot response
        with st.chat_message("assistant"):
            st.markdown(chatbot_response)

        # Clear text area after sending
        st.experimental_rerun()
    else:
        st.warning("⚠️ Please enter a question before clicking send!")
