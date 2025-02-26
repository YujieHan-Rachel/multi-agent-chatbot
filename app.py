import os
import openai
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from agents import Head_Agent

# Streamlit setup
st.set_page_config(page_title="Multi-Agent Chatbot", layout="wide")

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "miniproject2-multi-agent-chatbot"

# Check if API keys are set
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API keys are missing! Please set them in Hugging Face Secrets.")
    st.stop()

# ✅ Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        st.warning(f"⚠️ Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

except Exception as e:
    st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
    st.stop()

# ✅ Initialize chatbot agent
if "chatbot_agent" not in st.session_state:
    st.session_state.chatbot_agent = Head_Agent(openai_client, pinecone_index)

# ✅ Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("🤖 Multi-Agent Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.text_area("💬 Your question:", height=100)

if st.button("Send"):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        chatbot_response = st.session_state.chatbot_agent.process_query(user_input)

        st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

        st.experimental_rerun()
    else:
        st.warning("⚠️ Please enter a question before clicking send!")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()
