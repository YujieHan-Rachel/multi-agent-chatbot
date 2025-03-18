import os
import openai
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from agents import Head_Agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Multi-Agent Chatbot System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "miniproject2-multi-agent-chatbot"

# Header section
st.title("🤖 Multi-Agent Chatbot System")
st.markdown("""
This chatbot uses multiple specialized agents to answer your questions about machine learning.
Each agent has a specific role in processing your query and generating a response.
""")

# Check if API keys are set
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API keys are missing! Please set them in environment variables.")
    st.stop()

# Initialize OpenAI client
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    st.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
    logger.error(f"OpenAI initialization error: {e}")
    st.stop()

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # First, check if the index exists
    try:
        all_indexes = pc.list_indexes().names()
        index_exists = PINECONE_INDEX_NAME in all_indexes
    except Exception as e:
        logger.error(f"Failed to list Pinecone indexes: {e}")
        index_exists = False
        st.warning("⚠️ Could not verify if the Pinecone index exists. Will attempt to create it.")

    if not index_exists:
        st.warning(f"⚠️ Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
        logger.warning(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")

        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Successfully created Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            st.error(f"❌ Failed to create Pinecone index: {str(e)}")
            logger.error(f"Pinecone index creation error: {e}")
            st.stop()

    # Connect to the index
    try:
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' connected")
    except Exception as e:
        st.error(f"❌ Failed to connect to Pinecone index: {str(e)}")
        logger.error(f"Pinecone index connection error: {e}")
        st.stop()

    # Check if the index has any vectors
    try:
        index_stats = pinecone_index.describe_index_stats()
        vector_count = index_stats.get('total_vector_count', 0)

        # Display vector count in sidebar
        st.sidebar.info(f"📊 Vector count in Pinecone index: {vector_count}")

        if vector_count == 0:
            st.warning(
                "⚠️ Your Pinecone index is empty. You need to add document embeddings before using the chatbot effectively.")
            logger.warning("Pinecone index is empty - no vectors found")
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        st.warning("⚠️ Could not verify if the Pinecone index contains vectors.")

except Exception as e:
    st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
    logger.error(f"Pinecone initialization error: {e}")
    st.stop()

# Sidebar with mode selection and information
with st.sidebar:
    st.header("Agent Configuration")

    # Mode selection
    if "chatbot_mode" not in st.session_state:
        st.session_state.chatbot_mode = "precise"

    mode = st.radio(
        "Select Agent Mode:",
        ["precise", "chatty"],
        help="Precise mode gives concise, factual answers. Chatty mode is more conversational."
    )

    if mode != st.session_state.chatbot_mode:
        st.session_state.chatbot_mode = mode
        # If we already have a chatbot agent, update its mode
        if "chatbot_agent" in st.session_state:
            try:
                st.session_state.chatbot_agent.set_mode(mode)
                st.success(f"✅ Mode changed to '{mode}'")
                logger.info(f"Agent mode changed to: {mode}")
            except Exception as e:
                st.error(f"Failed to change mode: {str(e)}")
                logger.error(f"Mode change error: {e}")

    st.header("Agent Information")

    if st.button("👁️ Show Agent Architecture"):
        with st.expander("Agent Architecture", expanded=True):
            st.markdown("""
            ### Multi-Agent System Architecture

            1. **Head Agent (Controller)**
               - Coordinates all other agents
               - Manages conversation history

            2. **Obnoxious Agent**
               - Checks if user input is appropriate
               - Filters out harmful or offensive content

            3. **Pinecone Query Agent**
               - Determines if query relates to domain knowledge
               - Routes irrelevant questions appropriately

            4. **Query Agent**
               - Converts user query to embeddings
               - Retrieves potential relevant documents

            5. **Relevant Documents Agent**
               - Filters retrieved documents by relevance
               - Ensures responses are backed by appropriate context

            6. **Answering Agent**
               - Generates final response based on relevant documents
               - Fallback to general knowledge when needed
            """)

    st.header("Actions")
    if st.button("🗑️ Clear Chat History"):
        # Reset conversation but keep the agent
        if "messages" in st.session_state:
            st.session_state.messages = []
            logger.info("Chat history cleared")
        if "chatbot_agent" in st.session_state:
            # Reset agent's conversation history too
            st.session_state.chatbot_agent.conv_history = []
        st.success("Chat history cleared!")
        st.experimental_rerun()

    # In the sidebar, add PDF processing section
    st.header("PDF Processing")

    if st.button("Process Machine Learning PDF"):
        try:
            st.info("Starting PDF processing...")

            # Import data_loader module
            import data_loader

            # Show progress information
            with st.spinner("Step 1/3: Loading PDF file..."):
                page_texts, page_numbers = data_loader.load_pdf()
                st.success(f"✓ Loaded {len(page_texts)} pages from PDF")

            with st.spinner("Step 2/3: Chunking text and generating embeddings..."):
                chunks, chunk_page_numbers = data_loader.chunk_text(page_texts, page_numbers)
                df = data_loader.prepare_data(chunks, chunk_page_numbers)
                st.success(f"✓ Generated embeddings for {len(df)} text chunks")

            with st.spinner("Step 3/3: Uploading to Pinecone..."):
                stats = data_loader.create_pinecone_index(df)
                vector_count = stats.get('total_vector_count', 0)
                st.success(f"✓ Successfully uploaded to Pinecone. Total vectors: {vector_count}")

            # Reload the page to update index status
            st.success("PDF processing complete! Refreshing page...")
            st.experimental_rerun()

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.error("Please check logs for more details")

# Test direct query to Pinecone if index is empty
if "vector_count" in locals() and vector_count == 0:
    if st.button("🧪 Test Pinecone Connection"):
        try:
            st.info("Testing Pinecone connection with a sample query...")

            # Generate an embedding
            test_response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=["machine learning basics"]
            )
            test_embedding = test_response.data[0].embedding

            # Query Pinecone
            test_results = pinecone_index.query(
                vector=test_embedding,
                top_k=1,
                include_values=False,
                include_metadata=True
            )

            st.json(test_results)
            st.success("Pinecone query test completed successfully!")
        except Exception as e:
            st.error(f"Test query failed: {str(e)}")
            logger.error(f"Pinecone test query error: {e}")

# Initialize or update chatbot agent with current mode
if "chatbot_agent" not in st.session_state:
    try:
        st.session_state.chatbot_agent = Head_Agent(
            openai_client,
            pinecone_index,
            domain="machine learning",
            mode=st.session_state.chatbot_mode
        )
        logger.info(f"Initialized chatbot agent in {st.session_state.chatbot_mode} mode")
    except Exception as e:
        st.error(f"Failed to initialize chatbot agent: {str(e)}")
        logger.error(f"Chatbot agent initialization error: {e}")
        st.stop()

# Initialize conversation history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.info("Initialized empty message history")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Ask me about machine learning...")

if user_query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Add to message history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get response (with spinner to show processing)
    with st.spinner("Thinking..."):
        logger.info(f"Processing user query: {user_query[:50]}...")
        try:
            chatbot_response = st.session_state.chatbot_agent.process_query(user_query)
            logger.info(f"Generated response of length {len(chatbot_response)}")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            chatbot_response = "I encountered an error while processing your query. Please try again or ask something different."

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(chatbot_response)

    # Add to message history
    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})