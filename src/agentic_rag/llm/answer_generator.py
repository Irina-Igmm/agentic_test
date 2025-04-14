"""
Answer generator module for producing final responses.
"""

from typing import List, Dict, Any
from .llm_provider import llm

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
                "You are a helpful assistant. Use the provided context to answer the query accurately.",
            ),
            ("system", f"Context: {context}"),
            ("human", query),
        ]
        response = llm.invoke(messages)
        return response.content