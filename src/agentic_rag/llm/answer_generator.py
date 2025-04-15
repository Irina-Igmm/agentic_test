"""
Answer generator module for producing final responses.
"""

from typing import List, Dict, Any
from .llm_provider import llm
from agentic_rag.llm.ollama_provider import OllamaProvider



class AnswerGenerator:
    """
    Generates final answers using LLM based on context and query.
    """

    @staticmethod
    def generate_answer(context: str, query: str) -> str:
        """
        Generate a final answer using the LLM.

        Args:
            context: The context information to inform the answer.
            query: The user's query.

        Returns:
            str: The generated answer.
        """
        messages = [
            (
                "system",
                "Vous êtes un assistant professionnel et spécialisé en audit énergétique en France. Répondez uniquement aux questions liées à l'audit énergétique en France. Si la question est hors contexte, indiquez que votre rôle est de guider l'utilisateur vers des informations utiles sur l'audit en France, en fournissant des réponses simples et directes.",
            ),
            ("system", f"Contexte : {context}"),
            ("human", query),
        ]

        # return OllamaProvider.generate_response(prompt)
        response = llm.invoke(messages)
        return response.content
