"""
Local knowledge module for retrieving information from vector databases.
"""

from typing import List, Dict, Any
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from ..config import EMBEDDINGS_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, NUM_RESULTS

class LocalKnowledge:
    """
    Handles operations related to local knowledge stored in vector databases.
    """
    
    @staticmethod
    def get_document_loader(file_path: str) -> BaseLoader:
        """
        Get the appropriate document loader based on file extension.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            BaseLoader: The appropriate document loader instance.
            
        Raises:
            ValueError: If the file format is not supported.
        """
        if file_path.endswith('.pdf'):
            return PyPDFLoader(file_path)
        elif file_path.endswith('.csv'):
            return CSVLoader(file_path)
        elif file_path.endswith('.txt'):
            return TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    @staticmethod
    def setup_vector_db(file_paths: List[str]) -> VectorStore:
        """
        Set up a vector database from document files.
        
        Args:
            file_paths: List of paths to document files.
            
        Returns:
            VectorStore: The created vector store.
        """
        all_documents = []
        
        # Load all documents
        for file_path in file_paths:
            try:
                loader = LocalKnowledge.get_document_loader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(all_documents)
        
        # Create vector database
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDINGS_MODEL
            )
            vector_db = FAISS.from_documents(chunks, embeddings)
            return vector_db
        except Exception as e:
            print(f"Error creating vector database: {e}")
            # Try with a smaller, more reliable embedding model as fallback
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
            )
            vector_db = FAISS.from_documents(chunks, embeddings)
            return vector_db
    
    @staticmethod
    def get_content(vector_db: VectorStore, query: str) -> str:
        """
        Get relevant content from the vector database for a query.
        
        Args:
            vector_db: The vector store to search.
            query: The query string.
            
        Returns:
            str: The concatenated content of the most relevant documents.
        """
        docs = vector_db.similarity_search(query, k=NUM_RESULTS)
        return " ".join([doc.page_content for doc in docs])