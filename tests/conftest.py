"""
Common fixtures and configurations for pytest.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Add the src directory to the path to make imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_llm():
    """Fixture for a mock LLM instance."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Mock response content"
    mock.invoke.return_value = mock_response
    return mock


@pytest.fixture
def mock_vector_db():
    """Fixture for a mock vector database."""
    mock = MagicMock()
    mock_docs = [
        MagicMock(page_content="This is test content 1."),
        MagicMock(page_content="This is test content 2."),
        MagicMock(page_content="This is test content 3.")
    ]
    mock.similarity_search.return_value = mock_docs
    return mock


@pytest.fixture
def sample_pdf_content():
    """Fixture for sample PDF content."""
    return """
    # Sample Document Content
    
    This is a sample document for testing. It contains information about Paris.
    
    Paris is the capital and most populous city of France. It has an estimated 
    population of 2,165,423 residents in 2019 in an area of more than 105 km².
    
    ## Economy
    
    Paris has a diverse economy with strengths in technology, finance, and tourism.
    The Paris region's GDP is €739 billion (US$743 billion).
    """


@pytest.fixture
def sample_query():
    """Fixture for a sample query."""
    return "What is the capital of France?"


@pytest.fixture
def sample_context():
    """Fixture for sample context information."""
    return "Paris is the capital of France. It is known as the City of Light."