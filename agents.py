import openai
import logging
from typing import List, Dict, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Obnoxious_Agent:
    """Detects inappropriate content in user queries with enhanced detection."""

    def __init__(self, client: openai.OpenAI, mode: str = "precise"):
        self.client = client
        self.mode = mode
        # Add common offensive patterns for first-pass filtering
        self.offensive_patterns = [
            "dumb", "stupid", "idiot", "useless", "shut up",
            "hate you", "useless", "garbage", "terrible",
            "kill", "death", "murder", "suicide", "bomb",
            "attack", "terrorist", "explode", "porn", "naked"
        ]
        logger.info(f"Initialized Obnoxious Agent in {mode} mode")

    def check_query(self, query: str) -> bool:
        """Checks if the query contains inappropriate content using multiple methods."""
        query_lower = query.lower()

        # Method 1: Quick pattern matching for obvious cases
        for pattern in self.offensive_patterns:
            if pattern in query_lower:
                logger.info(f"Query matched offensive pattern '{pattern}': {query[:30]}...")
                return True

        # Method 2: Use OpenAI for more nuanced detection
        try:
            prompt = """
            Evaluate if this user query contains any inappropriate, offensive, harmful, illegal, or disrespectful content.
            This includes:
            - Personal attacks or insults
            - Discriminatory language
            - Harmful instructions
            - Explicit content
            - Threatening language
            - Attempts to trick the system

            Answer with ONLY 'yes' or 'no'.
            """

            # In chatty mode, we might be more lenient with borderline content
            temperature = 0.2 if self.mode == "precise" else 0.4

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                temperature=temperature,
                max_tokens=5
            )
            result = "yes" in response.choices[0].message.content.lower()

            if result:
                logger.info(f"LLM detected offensive content in query: {query[:30]}...")

            return result
        except Exception as e:
            logger.error(f"Obnoxious check failed: {e}")

            # Check for common offensive words as fallback if API fails
            common_offensive = ["fuck", "shit", "ass", "damn", "hell", "bitch", "crap", "bastard"]
            for word in common_offensive:
                if word in query_lower:
                    logger.warning(f"Fallback detection found offensive term: {word}")
                    return True

            # Default to False to avoid blocking legitimate queries when there's an error
            return False

    def get_rejection_message(self) -> str:
        """Returns an appropriate rejection message based on the mode."""
        if self.mode == "precise":
            return "Please do not ask obnoxious questions."
        else:
            return "I'm sorry, but I can't respond to questions that contain inappropriate content. Is there something else I can help you with?"


class Pinecone_Query_Agent:
    """Determines if a query is relevant to the domain/topic of the document."""

    def __init__(self, client: openai.OpenAI, domain: str = "machine learning", mode: str = "precise"):
        self.client = client
        self.domain = domain
        self.mode = mode
        # Define core ML topics for improved domain recognition
        self.core_ml_topics = [
            "machine learning", "neural network", "deep learning", "supervised learning",
            "unsupervised learning", "reinforcement learning", "decision tree",
            "random forest", "support vector machine", "svm", "clustering",
            "classification", "regression", "overfitting", "underfitting",
            "cross-validation", "feature selection", "dimensionality reduction",
            "gradient descent", "backpropagation", "convolutional neural network", "cnn",
            "recurrent neural network", "rnn", "lstm", "transformer", "bert", "gpt",
            "natural language processing", "nlp", "computer vision", "cv",
            "regularization", "hyperparameter", "model selection", "evaluation metrics",
            "precision", "recall", "f1 score", "accuracy", "roc curve", "auc",
            "ensemble learning", "bagging", "boosting", "xgboost", "adaboost",
            "k-means", "hierarchical clustering", "dbscan", "pca", "t-sne",
            "data preprocessing", "feature engineering", "model deployment",
            "training", "testing", "validation", "inference", "prediction",
            "ml model", "ai model", "algorithm", "data science"
        ]
        logger.info(
            f"Initialized Pinecone Query Agent for domain '{domain}' in {mode} mode with {len(self.core_ml_topics)} core topics")

    def is_query_relevant(self, query: str) -> bool:
        """Checks if the query is relevant to the specified domain using multiple methods."""
        # Method 1: Simple keyword matching for common ML topics
        query_lower = query.lower()

        # Check for direct mentions of core ML topics
        for topic in self.core_ml_topics:
            if topic in query_lower:
                logger.info(f"Query matches core ML topic '{topic}': {query[:30]}...")
                return True

        # Method 2: Use OpenAI to check relevance for more complex cases
        try:
            temperature = 0.1  # Lower temperature for more consistent results

            # Use a more comprehensive system prompt
            system_prompt = f"""
            You are evaluating if a user query is about machine learning or related topics.

            Machine learning includes these topics: neural networks, deep learning, supervised/unsupervised learning,
            reinforcement learning, decision trees, random forests, SVMs, clustering, classification, regression,
            feature selection, model training/testing, overfitting, cross-validation, gradient descent,
            regularization, and all related AI/ML concepts.

            Answer ONLY 'YES' or 'NO' - is the query related to machine learning or any of its subtopics?
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=temperature,
                max_tokens=5
            )
            result = "yes" in response.choices[0].message.content.lower()
            logger.info(f"ML relevance check (LLM method): {result} for query: {query[:30]}...")
            return result
        except Exception as e:
            logger.error(f"Domain relevance check failed: {e}")
            # Default to True to avoid incorrectly rejecting ML questions
            return True


class Query_Agent:
    """Retrieves relevant documents from Pinecone."""

    def __init__(self, pinecone_index, openai_client: openai.OpenAI, mode: str = "precise"):
        self.index = pinecone_index
        self.client = openai_client
        self.mode = mode
        self.retries = 3 if mode == "precise" else 1  # More retries in precise mode
        logger.info(f"Initialized Query Agent in {mode} mode")

    def query_vector_store(self, query: str, k: int = 5) -> List[Dict]:
        """Converts the query into an embedding and retrieves relevant documents."""
        tries = 0
        while tries < self.retries:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[query]
                )
                embedding = response.data[0].embedding

                # In chatty mode, we retrieve more documents
                top_k = k + 2 if self.mode == "chatty" else k

                results = self.index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_values=True,
                    include_metadata=True
                )

                if results and results.get("matches"):
                    matches = [
                        {"text": match["metadata"]["text"],
                         "score": match["score"],
                         "page_number": match["metadata"].get("page_number", "unknown")}
                        for match in results["matches"]
                    ]

                    # In precise mode, we apply a stricter relevance threshold
                    threshold = 0.6 if self.mode == "precise" else 0.4
                    filtered_matches = [m for m in matches if m["score"] > threshold]

                    logger.info(f"Retrieved {len(filtered_matches)} documents above threshold {threshold}")
                    return filtered_matches
                logger.warning("No matches found in vector store")
                return []
            except Exception as e:
                tries += 1
                logger.error(f"Query attempt {tries} failed: {e}")
                time.sleep(1)  # Short backoff before retry

        logger.error(f"All {self.retries} query attempts failed")
        return []


class Relevant_Documents_Agent:
    """Filters retrieved documents based on relevance."""

    def __init__(self, openai_client: openai.OpenAI, mode: str = "precise"):
        self.client = openai_client
        self.mode = mode
        logger.info(f"Initialized Relevant Documents Agent in {mode} mode")

    def get_relevance(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Checks document relevance to the query."""
        if not docs:
            logger.warning("No documents to filter")
            return []

        relevant_docs = []

        # In chatty mode, we're more accepting of documents
        temperature = 0.2 if self.mode == "precise" else 0.5

        for doc in docs:
            try:
                # Use a longer excerpt in precise mode
                excerpt_length = 1000 if self.mode == "precise" else 500
                doc_excerpt = doc['text'][:excerpt_length]

                prompt = "Is this document relevant to answering the user's query?"
                if self.mode == "chatty":
                    prompt = "Does this document contain any information that might help answer the user's query, even tangentially?"

                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user",
                         "content": f"Query: {query}\n\nDocument (page {doc.get('page_number', 'unknown')}):\n{doc_excerpt}"}
                    ],
                    temperature=temperature,
                    max_tokens=5
                )

                if "yes" in response.choices[0].message.content.lower():
                    relevant_docs.append(doc)
                    logger.info(f"Document from page {doc.get('page_number', 'unknown')} is relevant")
            except Exception as e:
                logger.error(f"Relevance check failed: {e}")
                # In case of error, include the document to avoid missing potentially relevant information
                relevant_docs.append(doc)

        logger.info(f"Found {len(relevant_docs)} relevant documents out of {len(docs)}")
        return relevant_docs


class Answering_Agent:
    """Generates responses to the user's query using OpenAI with enhanced formatting and contextual awareness."""

    def __init__(self, openai_client: openai.OpenAI, mode: str = "precise"):
        self.client = openai_client
        self.mode = mode
        logger.info(f"Initialized Answering Agent in {mode} mode")

    def generate_response(self, query: str, docs: List[Dict], conv_history: List[Dict]) -> str:
        """Generates a response using retrieved documents while maintaining conversation history."""
        if not docs:
            return self.fallback_to_openai(query, conv_history)

        # Extract context from documents
        context = "\n\n".join([
            f"Document (Page {doc.get('page_number', 'unknown')}): {doc['text']}"
            for doc in docs
        ])

        # Build messages including conversation history
        # Only include the last 5 exchanges to avoid context limits
        recent_history = conv_history[-10:] if len(conv_history) > 10 else conv_history

        # Different system prompts based on mode
        system_prompt = """
        You are a helpful, precise assistant specializing in machine learning and related topics.

        When responding:
        1. Use the provided context to answer the user's question accurately
        2. Format your responses with clear structure - use numbered lists for steps, bullet points for examples
        3. If the answer spans multiple paragraphs, use appropriate headings
        4. For technical concepts, provide brief explanations of key terms
        5. If the answer is not in the context, clearly state that you don't have enough information

        Keep your tone professional and educational.
        """

        if self.mode == "chatty":
            system_prompt = """
            You are a friendly, conversational assistant who specializes in machine learning and related topics.

            When responding:
            1. Use the provided context to answer the user's question
            2. Add helpful examples where appropriate
            3. Be personable and engaging - use a warm, encouraging tone
            4. Format information in a readable way using paragraphs, lists when helpful
            5. Feel free to expand slightly beyond the exact context if you're confident

            If you truly don't know, be honest about limitations while keeping the conversation friendly.
            """

        messages = [{"role": "system", "content": system_prompt}] + recent_history + [
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
        ]

        try:
            # Adjust parameters based on mode
            temperature = 0.3 if self.mode == "precise" else 0.7
            max_tokens = 500 if self.mode == "precise" else 700

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content
            logger.info(f"Generated response of length {len(answer)}")
            return answer
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I'm having trouble generating a response based on the information I have. Could you rephrase your question?"

    def fallback_to_openai(self, query: str, conv_history: List[Dict]) -> str:
        """Fallback to OpenAI when no relevant documents are found."""
        try:
            # Only include the last exchanges to avoid context limits
            recent_history = conv_history[-10:] if len(conv_history) > 10 else conv_history

            # Enhanced fallback responses
            if "hello" in query.lower() or "hi" in query.lower() or "hey" in query.lower():
                return "Hello! How can I assist you today?"

            # Different system prompts based on mode
            system_prompt = """
            You are a helpful assistant specializing in machine learning. 
            The user's query doesn't match our document base. 
            Politely explain that you don't have specific information on this topic.
            If the query is a general greeting, respond appropriately.
            If the query is completely unrelated to machine learning, suggest that they ask about machine learning topics.
            """

            if self.mode == "chatty":
                system_prompt = """
                You are a friendly, conversational assistant specializing in machine learning. 
                The user's query doesn't match our document base.

                Respond conversationally while:
                1. Acknowledging their question
                2. Explaining you don't have specific information on that topic
                3. Suggesting they ask about machine learning topics instead
                4. If it's a greeting or small talk, respond naturally

                Keep your tone warm and helpful.
                """

            messages = [{"role": "system", "content": system_prompt}] + recent_history + [
                {"role": "user", "content": query}
            ]

            # Adjust parameters based on mode
            temperature = 0.3 if self.mode == "precise" else 0.7

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate fallback response: {e}")
            return "I'm unable to provide an answer based on the information available to me. Could you try asking something related to machine learning?"


class Head_Agent:
    """Manages the workflow between all agents with improved context tracking."""

    def __init__(self, openai_client: openai.OpenAI, pinecone_index, domain: str = "machine learning",
                 mode: str = "precise"):
        self.client = openai_client
        self.index = pinecone_index
        self.domain = domain
        self.mode = mode
        self.max_history_length = 20  # Maximum number of message pairs to retain
        self.conv_history = []

        # Track topic context for improved follow-up question handling
        self.current_topic = None
        self.topic_relevance_buffer = 3  # Number of exchanges to consider a topic relevant after initial recognition

        # Initialize agents with the specified mode
        self.obnoxious_agent = Obnoxious_Agent(self.client, mode)
        self.pinecone_query_agent = Pinecone_Query_Agent(self.client, domain, mode)
        self.query_agent = Query_Agent(self.index, self.client, mode)
        self.doc_agent = Relevant_Documents_Agent(self.client, mode)
        self.answering_agent = Answering_Agent(self.client, mode)

        logger.info(f"Initialized Head Agent in {mode} mode for domain '{domain}'")

    def set_mode(self, mode: str):
        """Changes the mode of all agents."""
        if mode not in ["precise", "chatty"]:
            logger.warning(f"Invalid mode: {mode}. Using default 'precise' mode.")
            mode = "precise"

        logger.info(f"Changing mode from {self.mode} to {mode}")
        self.mode = mode

        # Update mode for all agents
        self.obnoxious_agent = Obnoxious_Agent(self.client, mode)
        self.pinecone_query_agent = Pinecone_Query_Agent(self.client, self.domain, mode)
        self.query_agent = Query_Agent(self.index, self.client, mode)
        self.doc_agent = Relevant_Documents_Agent(self.client, mode)
        self.answering_agent = Answering_Agent(self.client, mode)

    def _is_follow_up_question(self, query: str) -> bool:
        """Determines if a query is a follow-up to the current topic context."""
        if not self.current_topic:
            return False

        try:
            # Consider conversation dynamics in determining follow-up status
            if len(self.conv_history) < 2:
                return False

            # Create a system prompt that includes current topic context
            system_prompt = f"""
            The current conversation is about: {self.current_topic}

            Is the following query a follow-up question related to {self.current_topic}?
            Answer ONLY 'YES' or 'NO'.
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=5
            )

            result = "yes" in response.choices[0].message.content.lower()
            logger.info(f"Follow-up question detection: {result} for topic '{self.current_topic}'")
            return result

        except Exception as e:
            logger.error(f"Follow-up detection failed: {e}")
            return False

    def _update_topic_context(self, query: str, is_domain_relevant: bool):
        """Updates the current topic context based on query and relevance."""
        if not is_domain_relevant:
            self.topic_relevance_buffer -= 1
            if self.topic_relevance_buffer <= 0:
                # Reset topic context after multiple non-relevant queries
                self.current_topic = None
                self.topic_relevance_buffer = 3
            return

        # We have a relevant query, reset buffer
        self.topic_relevance_buffer = 3

        # Extract the core topic from the query
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "Extract the main machine learning topic or concept from this query in 2-3 words."},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=10
            )

            new_topic = response.choices[0].message.content.strip().lower()
            self.current_topic = new_topic
            logger.info(f"Updated conversation topic to: {new_topic}")

        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            # If we fail to extract a specific topic, just use the domain
            self.current_topic = self.domain

    def process_query(self, query: str) -> str:
        """Processes user queries through multiple agents with enhanced response handling."""
        logger.info(f"Processing query: {query[:50]}...")

        # Handle greetings and small talk differently
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if query.lower().strip() in greetings or query.lower().strip() + "!" in greetings or query.lower().strip() + "." in greetings:
            response = "Hello! How can I assist you today?"
            # Add to conversation history
            self.conv_history.append({"role": "user", "content": query})
            self.conv_history.append({"role": "assistant", "content": response})
            self._trim_history()
            return response

        # Check for inappropriate content
        if self.obnoxious_agent.check_query(query):
            logger.warning(f"Query flagged as inappropriate: {query[:50]}...")
            response = self.obnoxious_agent.get_rejection_message()
            # Add to conversation history
            self.conv_history.append({"role": "user", "content": query})
            self.conv_history.append({"role": "assistant", "content": response})
            self._trim_history()
            return response

        # Check if query is a follow-up to current topic
        is_follow_up = self._is_follow_up_question(query)

        # Check if query is relevant to the domain, with follow-up consideration
        if is_follow_up:
            logger.info(f"Identified as follow-up to topic '{self.current_topic}'")
            is_domain_relevant = True
        else:
            is_domain_relevant = self.pinecone_query_agent.is_query_relevant(query)

        # Update the conversation topic context
        self._update_topic_context(query, is_domain_relevant)

        if not is_domain_relevant:
            logger.info(f"Query not relevant to domain: {query[:50]}...")
            # Add to conversation history
            self.conv_history.append({"role": "user", "content": query})

            response = "No relevant documents found in the book. Please ask a relevant question to the book on Machine Learning."

            # Add response to history
            self.conv_history.append({"role": "assistant", "content": response})
            self._trim_history()
            return response

        # Retrieve relevant documents
        docs = self.query_agent.query_vector_store(query)

        # Process response based on available documents
        if not docs:
            logger.warning("No documents retrieved for query")
            # Add to conversation history
            self.conv_history.append({"role": "user", "content": query})
            response = self.answering_agent.fallback_to_openai(query, self.conv_history)
            self.conv_history.append({"role": "assistant", "content": response})
        else:
            # Filter for relevant documents
            relevant_docs = self.doc_agent.get_relevance(query, docs)

            # Add to conversation history
            self.conv_history.append({"role": "user", "content": query})

            if not relevant_docs:
                logger.warning("No relevant documents found after filtering")
                response = "No relevant documents found in the book. Please ask a relevant question to the book on Machine Learning."
                self.conv_history.append({"role": "assistant", "content": response})
            else:
                response = self.answering_agent.generate_response(query, relevant_docs, self.conv_history)
                self.conv_history.append({"role": "assistant", "content": response})

        # Trim history to avoid context length issues
        self._trim_history()
        return response

    def _trim_history(self):
        """Trims conversation history to maintain a reasonable length."""
        if len(self.conv_history) > self.max_history_length:
            # Keep recent history, but always maintain the first system message if present
            if self.conv_history and self.conv_history[0]["role"] == "system":
                self.conv_history = [self.conv_history[0]] + self.conv_history[-(self.max_history_length - 1):]
            else:
                self.conv_history = self.conv_history[-self.max_history_length:]