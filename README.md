# 🤖 Multi-Agent Knowledge Retrieval Chatbot

 **An AI-powered chatbot using OpenAI GPT-3.5 and Pinecone for intelligent knowledge retrieval.**  
This chatbot efficiently handles user queries by orchestrating multiple AI agents, retrieving relevant documents, and generating context-aware responses.

---

## Project Overview

In this project, I built a chatbot that:

- Retrieves relevant documents using **Pinecone vector search**  
- Generates intelligent responses with **OpenAI GPT-3.5**  
- Uses a **multi-agent architecture** to ensure modular and scalable processing  

The system consists of multiple agents:

1. **Query Agent**  - Converts user queries into embeddings and retrieves relevant documents.  
2. **Relevance Agent**  - Filters out unrelated search results.  
3. **Answering Agent**  - Generates intelligent, context-aware responses using OpenAI LLM.  
4. **Obnoxious Agent**  - Ensures safety by moderating inappropriate queries.  
5. **Head Agent**  - Coordinates the workflow between all agents, ensuring efficiency.  

---

## Key Challenges & Solutions

### 1. Handling Irrelevant Questions
![Handling irrelevant questions](https://imgur.com/1bjnijr.png)  
- If no relevant documents are found, the bot **politely informs** the user instead of generating misleading responses.

### 2. Filtering Inappropriate Queries
![Handling obnoxious questions](https://imgur.com/qkeXAXY.png)  
- The chatbot detects **offensive or inappropriate** queries and responds with a predefined safe message.

### 3. Managing General Conversations
![Responding to general greetings](https://imgur.com/qSOcKWE.png)  
- The chatbot handles **general interactions**, such as greetings, ensuring a user-friendly experience.

### 4. Multi-Turn Conversation Handling
![Multi-turn conversation handling](https://imgur.com/mg10wRS.png)  
- Users can ask follow-up questions, and the chatbot **maintains context** to provide deeper insights.

---

## Project Architecture

├── agents.py # Implements the multi-agent system 
├── app.py # Streamlit-based UI for chatbot 
├── requirements.txt # Dependencies required for the project 
├── README.md # Project documentation 
└── .gitignore # Ensures sensitive files aren't committed


---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YujieHan-Rachel/multi-agent-chatbot.git
cd multi-agent-chatbot
```
### 2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install required dependencies
pip install -r requirements.txt
```

### 3. Set Up API Keys (Security Best Practice)
Ensure your OpenAI API Key and Pinecone API Key are stored as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
```
(For Windows, use set instead of export.)

## Run the Chatbot
```bash
streamlit run app.py
```
Open your browser and go to http://localhost:8501 to interact with the chatbot.
## Live Demo
🔗 Hugging Face Space: https://huggingface.co/spaces/yujierachel/multi-agent-chatbot
📂 GitHub Repository: https://github.com/YujieHan-Rachel/multi-agent-chatbot

## Customization & Enhancements
If you want to modify the chatbot behavior:

Optimize Retrieval: Modify Query_Agent in agents.py for better search results.
Enhance Response Quality: Experiment with prompt engineering in Answering_Agent.
Improve Document Filtering: Adjust ranking logic in Relevance_Agent.
## Contributing
Pull requests are welcome! If you find a bug or want to suggest improvements:

Fork the repo
Create a new branch (git checkout -b feature-new-feature)
Commit your changes (git commit -m "Added new feature")
Push to your branch (git push origin feature-new-feature)
Open a Pull Request
## License
This project is licensed under the MIT License. Feel free to use and modify!

## Contact
For questions or collaboration, reach out at:

Email: [yujierachel@gmail.com]
LinkedIn: [[LinkedIn Profile](https://www.linkedin.com/in/yujie-rachel-han/)]

## Final Notes
- This project showcases a scalable, modular AI chatbot with efficient knowledge retrieval.
- Ideal for AI-driven information retrieval, Q&A systems, and interactive chatbots.
- Can be expanded to support domain-specific datasets or company knowledge bases.
- Enjoy building AI-powered conversational systems! 🚀
