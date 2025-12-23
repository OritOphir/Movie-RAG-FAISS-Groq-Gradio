"""
index.py

Splits documents into chunks and builds a FAISS vector index.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_retriever(documents):
    """Create FAISS retriever from documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})
