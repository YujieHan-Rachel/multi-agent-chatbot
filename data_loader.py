import os
import logging
import pandas as pd
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration variables - update these as needed
PDF_PATH = "machine_learning.pdf"  # Path to your PDF file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "miniproject2-multi-agent-chatbot"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 50


def load_pdf():
    """Task 1: Load PDF file and extract text."""
    logger.info(f"Loading PDF file: {PDF_PATH}")

    try:
        # Load the PDF document
        loader = PyMuPDFLoader(PDF_PATH)
        documents = loader.load()

        # Extract text and page numbers
        page_texts = [doc.page_content for doc in documents]
        page_numbers = [doc.metadata["page"] + 1 for doc in documents]  # Pages are 0-indexed in PyMuPDF

        logger.info(f"Successfully loaded {len(page_texts)} pages from PDF")
        return page_texts, page_numbers
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise


def chunk_text(page_texts, page_numbers):
    """Task 2: Break down the extracted text into smaller chunks."""
    logger.info("Breaking text into chunks")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Storage for chunks and their page numbers
    chunks = []
    chunk_page_numbers = []
    previous_page_tail = ""

    # Process each page
    for i, (text, page_num) in enumerate(zip(page_texts, page_numbers)):
        # Append previous page's tail to current page
        if previous_page_tail:
            text = previous_page_tail + " " + text
            previous_page_tail = ""

        # Split text into chunks
        page_chunks = text_splitter.split_text(text)

        # Store chunks with page numbers
        chunks.extend(page_chunks)
        chunk_page_numbers.extend([page_num] * len(page_chunks))

        # Save the tail of the current page
        if len(page_chunks) > 0:
            previous_page_tail = page_chunks[-1][-CHUNK_OVERLAP:]

    logger.info(f"Created {len(chunks)} chunks from {len(page_texts)} pages")
    return chunks, chunk_page_numbers


def prepare_data(chunks, chunk_page_numbers):
    """Task 2: Prepare the data and generate embeddings."""
    logger.info("Preparing data and generating embeddings")

    # Check if OpenAI API key is set
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set")

    # Create DataFrame
    df = pd.DataFrame({
        'text': chunks,
        'page_number': chunk_page_numbers
    })

    # Preprocess text
    df['processed_text'] = df['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Function to generate embeddings
    def get_embedding(text):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    # Generate embeddings for each chunk
    logger.info("Generating embeddings. This may take some time...")
    df['embedding'] = df['processed_text'].apply(get_embedding)

    logger.info(f"Generated embeddings for {len(df)} chunks")
    return df


def create_pinecone_index(df):
    """Task 3: Create Pinecone index and insert data."""
    logger.info("Creating Pinecone index and inserting data")

    # Check if Pinecone API key is set
    if not PINECONE_API_KEY:
        raise ValueError("Pinecone API key is not set")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    index_exists = PINECONE_INDEX_NAME in pc.list_indexes().names()

    # Create index if it doesn't exist
    if not index_exists:
        logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Connect to index
    index = pc.Index(PINECONE_INDEX_NAME)

    # Insert data in batches
    batch_size = 100
    total_rows = len(df)

    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        batch_df = df.iloc[i:end_idx]

        vectors = []
        for j, row in batch_df.iterrows():
            # Create metadata dictionary
            metadata = {
                "text": row['text'],
                "page_number": int(row['page_number'])
            }

            # Create vector
            vector = {
                "id": f"chunk_{j}",
                "values": row['embedding'],
                "metadata": metadata
            }
            vectors.append(vector)

        # Upsert batch
        index.upsert(vectors=vectors)
        logger.info(f"Inserted batch {i // batch_size + 1}/{(total_rows - 1) // batch_size + 1} into Pinecone")

    # Get index statistics
    stats = index.describe_index_stats()
    logger.info(f"Pinecone index stats: {stats}")

    return stats


def main():
    """Main function to execute all tasks."""
    try:
        # Task 1: Load PDF and extract text
        page_texts, page_numbers = load_pdf()

        # Task 2: Break text into chunks
        chunks, chunk_page_numbers = chunk_text(page_texts, page_numbers)

        # Task 2: Prepare data and generate embeddings
        df = prepare_data(chunks, chunk_page_numbers)

        # Task 3: Create Pinecone index and insert data
        stats = create_pinecone_index(df)

        logger.info("Data processing complete! Your chatbot is ready to use.")
        logger.info(f"Total vectors in Pinecone: {stats.get('total_vector_count', 0)}")

        return True
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {e}")
        return False


if __name__ == "__main__":
    main()