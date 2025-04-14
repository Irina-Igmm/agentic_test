"""
Configuration settings for the agentic datacamp project.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Model settings
# For langchain_groq integration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mixtral-8x7b-32768")

# For direct Groq API integration
GROQ_DIRECT_MODEL = os.getenv("GROQ_DIRECT_MODEL", "llama-3.3-70b-versatile")

# Logging settings
VERBOSE = True

# Application settings
DEFAULT_TOPIC = "data science trends in 2025"