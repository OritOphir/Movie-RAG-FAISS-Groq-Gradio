# 🎬 Movie RAG Chatbot

A clean and minimal Retrieval-Augmented Generation (RAG) project built on a movie dataset.

## What this project demonstrates
- Document ingestion and chunking
- Semantic search with FAISS
- HuggingFace sentence embeddings
- LLM-based answering using Groq (LLaMA 3)
- Conversational context without legacy LangChain memory
- Simple Gradio UI

## Tech Stack
- LangChain (Runnable-style, non-legacy)
- FAISS
- Sentence-Transformers
- Groq LLMs
- Gradio

## How to Run

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
python src/app.py
