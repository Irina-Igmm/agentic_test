"""
Unit tests for the local knowledge component.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile

# Add the src directory to the path to make imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agentic_rag.knowledge.local_knowledge import LocalKnowledge


class TestLocalKnowledge(unittest.TestCase):
    """Test cases for the LocalKnowledge class."""
    
    def test_get_document_loader(self):
        """Test get_document_loader for different file types."""
        # Test PDF loader
        pdf_path = "test.pdf"
        loader = LocalKnowledge.get_document_loader(pdf_path)
        self.assertEqual(loader.file_path, pdf_path)
        
        # Test CSV loader
        csv_path = "test.csv"
        loader = LocalKnowledge.get_document_loader(csv_path)
        self.assertEqual(loader.file_path, csv_path)
        
        # Test text loader
        txt_path = "test.txt"
        loader = LocalKnowledge.get_document_loader(txt_path)
        self.assertEqual(loader.file_path, txt_path)
        
        # Test unsupported file type
        with self.assertRaises(ValueError):
            LocalKnowledge.get_document_loader("test.docx")
    
    @patch("src.agentic_rag.knowledge.local_knowledge.HuggingFaceEmbeddings")
    @patch("src.agentic_rag.knowledge.local_knowledge.FAISS")
    @patch("src.agentic_rag.knowledge.local_knowledge.RecursiveCharacterTextSplitter")
    @patch("src.agentic_rag.knowledge.local_knowledge.PyPDFLoader")
    def test_setup_vector_db(self, mock_loader, mock_splitter, mock_faiss, mock_embeddings):
        """Test setup_vector_db function."""
        # Configure mocks
        mock_docs = [MagicMock()]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader.return_value = mock_loader_instance
        
        mock_chunks = [MagicMock()]
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter.return_value = mock_splitter_instance
        
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_vectorstore = MagicMock()
        mock_faiss.from_documents.return_value = mock_vectorstore
        
        # Call the function
        result = LocalKnowledge.setup_vector_db(["test.pdf"])
        
        # Verify the result
        self.assertEqual(result, mock_vectorstore)
        mock_loader.assert_called_once()
        mock_splitter.assert_called_once()
        mock_embeddings.assert_called_once()
        mock_faiss.from_documents.assert_called_once_with(mock_chunks, mock_embedding_instance)
    
    @patch("src.agentic_rag.knowledge.local_knowledge.HuggingFaceEmbeddings")
    @patch("src.agentic_rag.knowledge.local_knowledge.FAISS")
    def test_get_content(self, mock_faiss, mock_embeddings):
        """Test get_content function."""
        # Create mock documents with page content
        mock_docs = [
            MagicMock(page_content="This is content 1."),
            MagicMock(page_content="This is content 2.")
        ]
        
        # Configure mock vector store
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = mock_docs
        
        # Call the function
        query = "test query"
        result = LocalKnowledge.get_content(mock_vectorstore, query)
        
        # Verify the result
        expected_result = "This is content 1. This is content 2."
        self.assertEqual(result, expected_result)
        mock_vectorstore.similarity_search.assert_called_once_with(query, k=5)


if __name__ == "__main__":
    unittest.main()