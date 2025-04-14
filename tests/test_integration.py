"""
Integration tests for the AgenticRAG system.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to the path to make imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agentic_rag.main import AgenticRAG
from src.agentic_rag.knowledge.knowledge_router import KnowledgeRouter
from src.agentic_rag.knowledge.local_knowledge import LocalKnowledge
from src.agentic_rag.knowledge.web_knowledge import WebKnowledge
from src.agentic_rag.llm.answer_generator import AnswerGenerator


class TestAgenticRAGIntegration(unittest.TestCase):
    """Integration tests for the AgenticRAG class."""
    
    @patch("src.agentic_rag.knowledge.knowledge_router.KnowledgeRouter.check_local_knowledge")
    @patch("src.agentic_rag.knowledge.local_knowledge.LocalKnowledge.setup_vector_db")
    @patch("src.agentic_rag.knowledge.local_knowledge.LocalKnowledge.get_content")
    @patch("src.agentic_rag.knowledge.web_knowledge.WebKnowledge.get_content")
    @patch("src.agentic_rag.llm.answer_generator.AnswerGenerator.generate_answer")
    def test_full_rag_process_local(self, 
                                   mock_generate_answer,
                                   mock_web_content, 
                                   mock_local_content,
                                   mock_setup_db, 
                                   mock_check_local):
        """Test the full RAG process when using local knowledge."""
        # Configure mocks
        mock_db = MagicMock()
        mock_setup_db.return_value = mock_db
        mock_check_local.return_value = True  # Use local knowledge
        mock_local_content.return_value = "Local document content about the query topic."
        mock_generate_answer.return_value = "This is the answer based on local knowledge."
        
        # Create temp files for testing
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            # Initialize the system with the temp file
            rag = AgenticRAG([f.name])
            
            # Process a query
            result = rag.process_query("What is RAG?")
            
            # Verify the flow
            mock_check_local.assert_called_once()
            mock_local_content.assert_called_once()
            mock_web_content.assert_not_called()  # Shouldn't use web content
            mock_generate_answer.assert_called_once()
            self.assertEqual(result, "This is the answer based on local knowledge.")
    
    @patch("src.agentic_rag.knowledge.knowledge_router.KnowledgeRouter.check_local_knowledge")
    @patch("src.agentic_rag.knowledge.local_knowledge.LocalKnowledge.setup_vector_db")
    @patch("src.agentic_rag.knowledge.local_knowledge.LocalKnowledge.get_content")
    @patch("src.agentic_rag.knowledge.web_knowledge.WebKnowledge.get_content")
    @patch("src.agentic_rag.llm.answer_generator.AnswerGenerator.generate_answer")
    def test_full_rag_process_web(self, 
                                 mock_generate_answer,
                                 mock_web_content, 
                                 mock_local_content,
                                 mock_setup_db, 
                                 mock_check_local):
        """Test the full RAG process when using web knowledge."""
        # Configure mocks
        mock_db = MagicMock()
        mock_setup_db.return_value = mock_db
        mock_check_local.return_value = False  # Use web knowledge
        mock_web_content.return_value = "Web content about the query topic."
        mock_generate_answer.return_value = "This is the answer based on web search."
        
        # Create temp files for testing
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            # Initialize the system with the temp file
            rag = AgenticRAG([f.name])
            
            # Process a query with specific URLs
            urls = ["https://example.com/page1", "https://example.com/page2"]
            result = rag.process_query("What is the latest in AI?", specific_urls=urls)
            
            # Verify the flow
            mock_check_local.assert_called_once()
            mock_local_content.assert_called_once()  # Called for initial context
            mock_web_content.assert_called_once_with("What is the latest in AI?", urls)
            mock_generate_answer.assert_called_once()
            self.assertEqual(result, "This is the answer based on web search.")
    
    @patch("src.agentic_rag.knowledge.local_knowledge.LocalKnowledge.setup_vector_db")
    def test_no_knowledge_base_error(self, mock_setup_db):
        """Test that an error is raised when trying to query without initializing the knowledge base."""
        # Configure mock to return None (no vector db)
        mock_setup_db.return_value = None
        
        # Create the RAG system with an empty list (no files)
        rag = AgenticRAG([])
        
        # Try to process a query without a knowledge base
        with self.assertRaises(ValueError):
            rag.process_query("What is RAG?")


if __name__ == "__main__":
    unittest.main()