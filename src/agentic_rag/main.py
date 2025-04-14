"""
Main module for the Agentic RAG system.
"""

import os
from typing import List, Optional, Dict, Any, Union
from langchain.vectorstores.base import VectorStore

from .config import DEFAULT_DATA_DIR
from .knowledge.knowledge_router import KnowledgeRouter
from .knowledge.local_knowledge import LocalKnowledge
from .knowledge.web_knowledge import WebKnowledge
from .llm.answer_generator import AnswerGenerator


class AgenticRAG:
    """
    Main class for the Agentic RAG (Retrieval-Augmented Generation) system.
    
    This system combines local document knowledge with web search capabilities
    to intelligently answer user queries.
    """
    
    def __init__(self, file_paths: Optional[List[str]] = None):
        """
        Initialize the AgenticRAG system.
        
        Args:
            file_paths: Optional list of paths to document files.
                If None, no vector database will be initialized.
        """
        self.vector_db = None
        self.local_context = ""
        
        if file_paths:
            print("Setting up vector database...")
            self.setup_knowledge_base(file_paths)
    
    def setup_knowledge_base(self, file_paths: List[str]) -> None:
        """
        Set up the knowledge base from document files.
        
        Args:
            file_paths: List of paths to document files.
        """
        self.vector_db = LocalKnowledge.setup_vector_db(file_paths)
        
        # Get initial context for routing
        if self.vector_db:
            self.local_context = LocalKnowledge.get_content(self.vector_db, "")
            print("Knowledge base successfully initialized.")
    
    def process_query(self, 
                      query: str, 
                      specific_urls: Optional[List[str]] = None) -> str:
        """
        Process a user query and generate an answer.
        
        Args:
            query: The user's query.
            specific_urls: Optional list of specific URLs to scrape instead of searching.
            
        Returns:
            str: The generated answer to the query.
            
        Raises:
            ValueError: If no vector database has been initialized.
        """
        if self.vector_db is None:
            raise ValueError("No vector database has been initialized. Please call setup_knowledge_base() first.")
        
        print(f"Processing query: {query}")
        
        # Step 1: Check if we can answer from local knowledge
        can_answer_locally = KnowledgeRouter.check_local_knowledge(query, self.local_context)
        print(f"Can answer locally: {can_answer_locally}")
        
        # Step 2: Get context either from local DB or web
        if can_answer_locally:
            context = LocalKnowledge.get_content(self.vector_db, query)
            print("Retrieved context from local documents")
        else:
            context = WebKnowledge.get_content(query, specific_urls)
            print("Retrieved context from web")
        
        # Step 3: Generate final answer
        answer = AnswerGenerator.generate_answer(context, query)
        return answer


def initialize_from_data_dir(data_dir: str = DEFAULT_DATA_DIR) -> AgenticRAG:
    """
    Helper function to initialize the AgenticRAG system from a data directory.
    
    Args:
        data_dir: Directory path containing the data files.
        
    Returns:
        AgenticRAG: The initialized AgenticRAG instance.
    """
    # Get all supported file paths from the data directory
    supported_extensions = ['.pdf', '.csv', '.txt']
    file_paths = []
    
    for file in os.listdir(data_dir):
        if any(file.endswith(ext) for ext in supported_extensions):
            file_paths.append(os.path.join(data_dir, file))
    
    # Initialize the RAG system with the found files
    if file_paths:
        print(f"Found {len(file_paths)} files for knowledge base initialization:")
        for path in file_paths:
            print(f"  - {os.path.basename(path)}")
        
        return AgenticRAG(file_paths)
    else:
        print(f"No supported files found in {data_dir}")
        return AgenticRAG()