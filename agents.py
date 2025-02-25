import openai
import pinecone
import logging
from typing import List, Dict


# Initialize OpenAI and Pinecone clients
class Obnoxious_Agent:
    """Detects inappropriate content in user queries."""

    def __init__(self, client: openai.OpenAI, mode: str = "precise"):
        self.client = client  # OpenAI client

    def check_query(self, query: str) -> bool:
        """Checks if the query contains inappropriate content."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Does this query contain offensive content? {query}"}],
                temperature=0.2,
                max_tokens=5
            )
            return "yes" in response.choices[0].message.content.lower()
        except Exception as e:
            logging.error(f"Obnoxious check failed: {e}")
            return False


class Query_Agent:
    """Retrieves relevant documents from Pinecone."""

    def __init__(self, pinecone_index, openai_client: openai.OpenAI, mode: str = "precise"):
        self.index = pinecone_index
        self.client = openai_client

    def query_vector_store(self, query: str, k: int = 5) -> List[Dict]:
        """Converts the query into an embedding and retrieves relevant documents."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            embedding = response.data[0].embedding

            results = self.index.query(vector=embedding, top_k=k, include_metadata=True)

            if results and results.matches:
                return [
                    {"text": match.metadata["text"], "score": match.score}
                    for match in results.matches
                ]
            return []
        except Exception as e:
            logging.error(f"Query failed: {e}")
            return []


class Relevant_Documents_Agent:
    """Filters retrieved documents based on relevance."""

    def __init__(self, openai_client: openai.OpenAI, mode: str = "precise"):
        self.client = openai_client

    def get_relevance(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Checks document relevance to the query."""
        relevant_docs = []
        for doc in docs:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user",
                               "content": f"Is this document relevant to the query? {query}\n{doc['text'][:500]}"}],
                    temperature=0.2,
                    max_tokens=5
                )
                if "yes" in response.choices[0].message.content.lower():
                    relevant_docs.append(doc)
            except Exception as e:
                logging.error(f"Relevance check failed: {e}")
        return relevant_docs


class Answering_Agent:
    """Generates responses to the user's query using OpenAI."""

    def __init__(self, openai_client: openai.OpenAI):
        self.client = openai_client  # OpenAI client

    def generate_response(self, query: str, docs: List[Dict]) -> str:
        """Generates a response directly using OpenAI."""
        if not docs:
            # Directly generate an explanation using OpenAI if no documents are found
            return self.fallback_to_openai(query)

        context = "\n\n".join([f"Document {i + 1}: {doc['text']}" for i, doc in enumerate(docs)])

        messages = [{"role": "system", "content": "Use the following context to answer the user's question."}]
        messages.append({"role": "user", "content": f"Question: {query}\nContext:\n{context}"})

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.5,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to generate response: {str(e)}"

    def fallback_to_openai(self, query: str) -> str:
        """Fallback to OpenAI to generate an explanation when no relevant documents are found."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}],
                temperature=0.5,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to generate fallback response: {str(e)}"


class Head_Agent:
    """Manages the workflow between all agents."""

    def __init__(self, openai_client: openai.OpenAI, pinecone_index, mode: str = "precise"):
        self.client = openai_client
        self.index = pinecone_index
        self.mode = mode
        self.conv_history = []

        self.obnoxious_agent = Obnoxious_Agent(self.client, mode)
        self.query_agent = Query_Agent(self.index, self.client, mode)
        self.doc_agent = Relevant_Documents_Agent(self.client, mode)
        self.answering_agent = Answering_Agent(self.client)

    def process_query(self, query: str) -> str:
        """Processes user queries through multiple agents."""
        if self.obnoxious_agent.check_query(query):
            return "Sorry, I cannot answer inappropriate questions."

        # Check if it's the second question and use the previous context
        if len(self.conv_history) > 0 and 'logistic regression' in self.conv_history[-1]['content'].lower():
            query = "Logistic regression: " + query  # Modify query to ensure context is clear

        # Retrieve relevant documents
        docs = self.query_agent.query_vector_store(query)
        if not docs:
            return "No relevant documents found."

        valid_docs = self.doc_agent.get_relevance(query, docs)
        response = self.answering_agent.generate_response(query, valid_docs)

        # Add the user and assistant messages to the conversation history
        self.conv_history.append({"role": "user", "content": query})
        self.conv_history.append({"role": "assistant", "content": response})

        return response

