import os
import pinecone
import openai
import streamlit as st
from agents import Head_Agent

# Streamlit setup
st.set_page_config(page_title="Multi-Agent Chatbot", layout="centered")

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "miniproject2-multi-agent-chatbot"

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone using the new method
from pinecone import Pinecone, ServerlessSpec

# Create Pinecone instance with the API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists, and create it if not
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # Set the dimension based on your embeddings
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust to your region
    )

# Connect to the Pinecone index
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Initialize the chatbot agent
if "chatbot_agent" not in st.session_state:
    st.session_state.chatbot_agent = Head_Agent(openai_client, pinecone_index)

# Streamlit UI
st.title("Multi-Agent Chatbot")
st.write("Ask me anything, and I'll retrieve relevant information and generate an intelligent response.")

# Chat history storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.text_input("Your question:", "")

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

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
