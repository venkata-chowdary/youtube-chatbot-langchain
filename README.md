# YouTube Chatbot using LangChain

A chatbot that answers questions about YouTube video transcripts using LangChain, Google Gemini, and RAG (Retrieval-Augmented Generation).

## Features

- Fetches YouTube video transcripts automatically
- Chunks and embeds transcript text using Gemini embeddings
- Stores embeddings in FAISS vector database
- Retrieves relevant content for user queries
- Generates answers grounded in video content only

## Tech Stack

- LangChain
- Google Generative AI (Gemini)
- YouTube Transcript API
- FAISS (vector search)
- Python

## Installation

```bash
pip install langchain langchain-google-genai langchain-community youtube-transcript-api python-dotenv faiss-cpu
```

Set your Google API key in `.env`:
```
GOOGLE_API_KEY=your_api_key
```

## Usage

```python
result = chain.invoke({"question": "Who is the speaker?"})
print(result)
```

Run the chatbot:
```bash
python cahtbot_chain.py
```
