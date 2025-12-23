"""
app.py

Gradio web interface for the Movie RAG chatbot.
"""

import os
import gradio as gr
from ingest import load_documents
from index import build_retriever
from rag import MovieRAG


def main():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")

    documents = load_documents()
    retriever = build_retriever(documents)
    rag = MovieRAG(retriever, groq_api_key)

    def chat(message, history):
        return rag.ask(message)

    demo = gr.ChatInterface(
        fn=chat,
        title="🎬 Movie RAG Chatbot",
        description="Ask questions about movies using Retrieval-Augmented Generation.",
    )

    demo.launch()


if __name__ == "__main__":
    main()
