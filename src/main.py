"""
Main module for the Agentic RAG system.
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

from agentic_rag.config import DEFAULT_DATA_DIR, DEFAULT_URLS
from agentic_rag.knowledge.knowledge_router import KnowledgeRouter
from agentic_rag.knowledge.local_knowledge import LocalKnowledge
from agentic_rag.knowledge.web_knowledge import WebKnowledge
from agentic_rag.llm.answer_generator import AnswerGenerator

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import ChatMessageHistory
import uuid
import time
import shutil


# Définition des modèles Pydantic pour les requêtes et réponses
class AgentQuery(BaseModel):
    query: str
    # session_id: Optional[str] = None
    # urls: Optional[List[str]] = None


class AgentResponse(BaseModel):
    # session_id: str
    answer: str
    # history: List[Dict[str, Any]]


app = FastAPI()

# Allow CORS for all origins (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: {session_id: {"history": ChatMessageHistory, "last_active": timestamp}}
session_store: Dict[str, Dict] = {}
SESSION_TIMEOUT = 15 * 60  # 15 minutes in seconds


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
        # print("query user: ", query)

        # # Get initial context for routing
        if self.vector_db:
            self.local_context = LocalKnowledge.get_content(self.vector_db, "")
            print("Knowledge base successfully initialized.")

    def process_query(
        self, query: str, specific_urls: Optional[List[str]] = None
    ) -> str:
        if not query.strip():
            return "La requête est vide. Veuillez formuler une demande valide."

        if self.vector_db is None:
            raise ValueError(
                "No vector database has been initialized. Please call setup_knowledge_base() first."
            )

        print(f"Processing query: {query}")

        # Step 1: Check if we can answer from local knowledge
        can_answer_locally = KnowledgeRouter.check_local_knowledge(
            query, self.local_context
        )
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
    supported_extensions = [".pdf", ".csv", ".txt"]
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


# Initialize RAG system from data dir on startup
rag = initialize_from_data_dir()


# Background task to clean up expired sessions
def cleanup_sessions():
    now = time.time()
    expired = [
        sid
        for sid, v in session_store.items()
        if now - v["last_active"] > SESSION_TIMEOUT
    ]
    for sid in expired:
        del session_store[sid]


@app.post("/agent", response_model=AgentResponse)
async def chat_with_agent(
    agent_query: AgentQuery,
    background_tasks: BackgroundTasks,
):
    """
    Chat with the agent. Each request creates a new session with a unique UUID.
    """
    background_tasks.add_task(cleanup_sessions)
    sid = str(uuid.uuid4())
    # Create a new session
    session = {"history": ChatMessageHistory(), "last_active": time.time()}
    session_store[sid] = session

    # Add user message to history
    session["history"].add_user_message(agent_query.query)

    # Process the query and generate a response
    answer = rag.process_query(agent_query.query, specific_urls=DEFAULT_URLS)

    # Add AI response to history
    session["history"].add_ai_message(answer)

    # Format history for the response
    history = [
        {
            "query": msg.content if msg.type == "user" else None,
            "answer": msg.content if msg.type == "ai" else None,
            "timestamp": datetime.now().isoformat(),
        }
        for msg in session["history"].messages
    ]

    return AgentResponse(session_id=sid, answer=answer, history=history)


@app.post("/add-embedding")
async def add_embedding(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    """
    Add a new file to the FAISS index and update the knowledge base.
    """
    background_tasks.add_task(cleanup_sessions)
    # Save uploaded file to data dir
    data_dir = DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Update RAG system
    rag.setup_knowledge_base([file_path])
    return {"status": "success", "filename": file.filename}


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    # Format history for the response
    history = [
        {
            "query": msg.content if msg.type == "user" else None,
            "answer": msg.content if msg.type == "ai" else None,
            "timestamp": datetime.now().isoformat(),
        }
        for msg in session["history"].messages
    ]

    return {"session_id": session_id, "history": history}


@app.post("/query")
async def query_agent(query: str):
    """
    Endpoint to query the agent using RetrievalQA.
    """
    try:
        # Charger la base de données vectorielle
        # vector_db = LocalKnowledge.setup_vector_db(["path/to/document1.pdf", "path/to/document2.txt"])
        vector_db = rag.vector_db

        # Utiliser RetrievalQA pour répondre à la requête
        answer = LocalKnowledge.get_content_with_retrievalqa(vector_db, query)

        return {"query": query, "answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "Agentic RAG FastAPI is running."}
