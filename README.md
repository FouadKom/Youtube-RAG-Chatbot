# YouTube Transcript Summarization & RAG Pipeline

This project allows you to fetch YouTube video transcripts, summarize them, split them into chunks, embed them into a vector database, and query them using a **Retrieval-Augmented Generation (RAG)** pipeline with a large language model (LLM).  

It combines **YouTubeTranscriptApi**, **Hugging Face Transformers**, **LangChain**, and **Chroma** to create a full-text search and summarization pipeline.

---

## Features

- Fetch YouTube video transcripts directly from video URLs.
- Split transcripts into manageable chunks using `RecursiveCharacterTextSplitter`.
- Summarize each chunk individually using `facebook/bart-large-cnn`.
- Combine chunk summaries into a final overall summary.
- Embed chunks into a **Chroma vector store** for semantic search.
- Query transcripts using an LLM (`Gemini-2.5-Flash`) with session-based chat history.
- Fully interactive CLI or FastAPI integration.

---

## Requirements

- Python 3.10+
- Required environment variables:
  - `GOOGLE_API_KEY` — Google API Key (for LLM access)
  - `NGROK_AUTHTOKEN` — Ngrok auth token (if exposing via ngrok)

## Install dependencies:

```bash
pip install -r requirements.txt
```

## How It Works

### 1. Fetch YouTube Transcript
- The video URL is parsed to extract the video ID.
- `YouTubeTranscriptApi` fetches all transcript snippets.
- Snippets are combined into a single transcript string.

### 2. Split Transcript into Chunks
- The transcript is split into chunks of 1000 characters with 200 characters overlap.
- This improves semantic search and ensures each chunk contains meaningful context.

### 3. Summarize Chunks
- Each chunk is summarized individually using the `facebook/bart-large-cnn` model.
- Summaries are concatenated into a single **final summary** of the video.

### 4. Vector Store Embedding
- Each chunk is embedded using `SentenceTransformerEmbeddings`.
- Chunks are stored in a **Chroma vector database** for semantic search and retrieval.

### 5. Querying with LLM
- A large language model (`Gemini-2.5-Flash`) is initialized via LangChain.
- A RAG chain allows users to query the video transcript interactively.
- Session-based chat history ensures context-aware responses.
