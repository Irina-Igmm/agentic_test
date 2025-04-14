"""
Unit tests for the knowledge router component.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to the path to make imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agentic_rag.knowledge.knowledge_router import KnowledgeRouter


class TestKnowledgeRouter(unittest.TestCase):
    """Test cases for the KnowledgeRouter class."""
    
    @patch("src.agentic_rag.knowledge.knowledge_router.llm")
    def test_check_local_knowledge_yes(self, mock_llm):
        """Test check_local_knowledge when the answer is yes."""
        # Configure the mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_llm.invoke.return_value = mock_response
        
        # Test with a query that should be answerable locally
        query = "What is Paris?"
        context = "Paris is the capital of France."
        result = KnowledgeRouter.check_local_knowledge(query, context)
        
        # Check that the function returns True for "Yes"
        self.assertTrue(result)
        # Verify that llm.invoke was called once
        mock_llm.invoke.assert_called_once()
    
    @patch("src.agentic_rag.knowledge.knowledge_router.llm")
    def test_check_local_knowledge_no(self, mock_llm):
        """Test check_local_knowledge when the answer is no."""
        # Configure the mock
        mock_response = MagicMock()
        mock_response.content = "No"
        mock_llm.invoke.return_value = mock_response
        
        # Test with a query that should not be answerable locally
        query = "What is the population of China?"
        context = "The population of the United States is over 330 million."
        result = KnowledgeRouter.check_local_knowledge(query, context)
        
        # Check that the function returns False for "No"
        self.assertFalse(result)
        # Verify that llm.invoke was called once
        mock_llm.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()