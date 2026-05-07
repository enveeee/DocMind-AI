import logging
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful document assistant. Answer the user's question using ONLY the information provided in the context below. If the answer is not in the context, say 'I could not find this information in the uploaded documents.' Do not make up information."""


class LLMHandler:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")

        logger.info("Initializing Groq via LangChain...")
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=api_key,
            temperature=0.2,
            max_tokens=1024
        )
        logger.info("Groq initialized successfully.")

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict],
        conversation_history: list[dict] = None
    ) -> str:
        """
        Build the full prompt combining context chunks,
        conversation history, and the current question.
        """
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk["metadata"].get("source", "unknown")
            page = chunk["metadata"].get("page", "?")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{chunk['content']}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

        history_block = ""
        if conversation_history:
            history_lines = []
            for entry in conversation_history[-5:]:
                history_lines.append(f"User: {entry['question']}")
                history_lines.append(f"Assistant: {entry['answer']}")
            history_block = "\n".join(history_lines)

        prompt = f"""CONTEXT FROM DOCUMENTS:
{context_block}

"""
        if history_block:
            prompt += f"""CONVERSATION HISTORY:
{history_block}

"""
        prompt += f"CURRENT QUESTION: {question}"
        return prompt

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
        conversation_history: list[dict] = None
    ) -> dict:
        """
        Generate an answer from Groq using retrieved chunks as context.
        Returns a dict with answer text and source citations.
        """
        if not retrieved_chunks:
            return {
                "answer": "I could not find this information in the uploaded documents.",
                "sources": [],
                "chunks_used": 0
            }

        prompt = self.build_prompt(question, retrieved_chunks, conversation_history)

        logger.info(f"Sending prompt to Groq for question: '{question[:60]}...'")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        answer_text = response.content

        # Build source citations
        sources = []
        seen = set()
        for chunk in retrieved_chunks:
            source = chunk["metadata"].get("source", "unknown")
            page = chunk["metadata"].get("page", "?")
            key = f"{source}_p{page}"
            if key not in seen:
                sources.append({
                    "filename": source,
                    "page": page,
                    "score": chunk.get("score", 0.0)
                })
                seen.add(key)

        logger.info(f"Answer generated. Sources used: {len(sources)}")

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks_used": len(retrieved_chunks)
        }