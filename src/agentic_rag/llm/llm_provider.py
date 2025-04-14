"""
LLM provider module for handling language model integrations.
"""

from typing import List, Tuple, Union, Dict, Any
from langchain_groq import ChatGroq
from crewai import LLM as CrewLLM

from ..config import GROQ_API_KEY, OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_RETRIES

class LLMProvider:
    """
    Provider for language model integrations.
    """
    
    @staticmethod
    def get_groq_llm() -> ChatGroq:
        """
        Get a ChatGroq instance for generating responses.
        
        Returns:
            ChatGroq: A configured ChatGroq instance.
        """
        return ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=None,
            max_retries=LLM_MAX_RETRIES,
        )

    @staticmethod
    def get_crew_llm() -> CrewLLM:
        """
        Get a CrewLLM instance for CrewAI agents.
        
        Returns:
            CrewLLM: A configured CrewLLM instance.
        """
        return CrewLLM(
            model="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.7
        )


# Initialize default instances
llm = LLMProvider.get_groq_llm()
crew_llm = LLMProvider.get_crew_llm()