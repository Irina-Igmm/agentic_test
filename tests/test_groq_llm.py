import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from src
from src.main_old import setup_groq_direct_llm

# Initialize the LLM
llm = setup_groq_direct_llm()

# Test with a simple query
result = llm.invoke("Tell me about data science trends")
print(result.content)