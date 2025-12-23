"""
rag.py

Defines the RAG logic:
- Retriever
- Prompt
- LLM
- Conversation state
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class MovieRAG:
    def __init__(self, retriever, groq_api_key: str):
        self.retriever = retriever
        self.chat_history = []

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.2,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a factual movie assistant. "
                "Answer only using the provided context. "
                "If the answer is unknown, say so clearly."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])

    def _format_docs(self, docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def ask(self, question: str) -> str:
        """Answer a user question using RAG."""
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)

        messages = self.prompt.format_messages(
            question=question,
            context=context,
            chat_history=self.chat_history,
        )

        answer = self.llm.invoke(messages).content

        self.chat_history.append(("human", question))
        self.chat_history.append(("ai", answer))

        return answer
