# -----------------------------
# Imports
# -----------------------------
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

# -----------------------------
# Environment Setup
# -----------------------------
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

# -----------------------------
# YouTube Transcript Fetching
# -----------------------------
video_url = ""  # Add your YouTube URL here
video_id = video_url.split("=")[1]

ytt_api = YouTubeTranscriptApi()
fetched_snippets = ytt_api.fetch(video_id)

transcript = "\n".join(snippet['text'] for snippet in fetched_snippets)
print("Fetched transcript length:", len(transcript))

# -----------------------------
# Split Transcript into Chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.create_documents([transcript])
print(f"Splitting the transcript into {len(chunks)} chunks")

# -----------------------------
# Summarize Chunks Using Transformers
# -----------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summaries = []
for chunk in chunks:
    summary = summarizer(chunk.page_content, max_length=60, min_length=30, do_sample=False)
    summaries.append(summary[0]['summary_text'])

final_summary = " ".join(summaries)
print("Final Summary:", final_summary)

# -----------------------------
# Vector Store & Embeddings
# -----------------------------
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="youtube_transcript",
    persist_directory="./chroma_langchain.db",
    embedding_function=embeddings
)

_ = vector_store.add_documents(documents=chunks)
print("Documents added to vector store.")

# -----------------------------
# Initialize LLM
# -----------------------------
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# -----------------------------
# Testing Interactive Q&A
# -----------------------------
# ⚠️ Make sure `rag_chain_with_history` is defined in your environment
session_id = "user123"

while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    response = rag_chain_with_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    print(response)