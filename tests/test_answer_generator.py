"""
Unit tests for the answer generator component.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to the path to make imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agentic_rag.llm.answer_generator import AnswerGenerator


class TestAnswerGenerator(unittest.TestCase):
    """Test cases for the AnswerGenerator class."""
    
    @patch("src.agentic_rag.llm.answer_generator.llm")
    def test_generate_answer(self, mock_llm):
        """Test generate_answer function."""
        # Configure the mock
        mock_response = MagicMock()
        mock_response.content = "This is a test answer."
        mock_llm.invoke.return_value = mock_response
        
        # Test with sample context and query
        context = "Paris is the capital of France."
        query = "What is the capital of France?"
        
        # Call the function
        result = AnswerGenerator.generate_answer(context, query)
        
        # Check the result
        self.assertEqual(result, "This is a test answer.")
        
        # Verify that llm.invoke was called with the expected arguments
        mock_llm.invoke.assert_called_once()
        # Get the arguments passed to invoke
        call_args = mock_llm.invoke.call_args[0][0]
        
        # Verify the system messages
        self.assertEqual(call_args[0][0], "system")
        self.assertEqual(call_args[1][0], "system")
        self.assertEqual(call_args[1][1], f"Context: {context}")
        
        # Verify the human message contains the query
        self.assertEqual(call_args[2][0], "human")
        self.assertEqual(call_args[2][1], query)


if __name__ == "__main__":
    unittest.main()