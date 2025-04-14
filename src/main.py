"""
Main entry point for the agentic datacamp project.
"""
# Import local modules
from agents import create_researcher_agent, create_analyst_agent, create_report_writer_agent
from tasks import create_research_task, create_analysis_task, create_report_task
from config import GROQ_API_KEY, DEFAULT_MODEL, DEFAULT_TOPIC, VERBOSE, GROQ_DIRECT_MODEL

# Import required libraries
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai import Crew, Task
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Dict, Any, Optional
from groq import Groq

class GroqDirectLLM(BaseChatModel):
    """Custom LLM class that uses the official Groq client."""
    
    def __init__(self, api_key: str, model_name: str):
        """Initialize with API key and model name."""
        super().__init__()
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        
    def _generate(self, 
                  messages: List[Dict[str, Any]], 
                  stop: Optional[List[str]] = None,
                  **kwargs) -> Dict[str, Any]:
        """Generate a response from the Groq API."""
        # Convert LangChain message format to Groq format
        groq_messages = []
        for message in messages:
            groq_messages.append({
                "role": message["role"],
                "content": message["content"],
            })
        
        # Call the Groq API
        response = self.client.chat.completions.create(
            messages=groq_messages,
            model=self.model_name,
            stop=stop,
            **kwargs
        )
        
        # Convert Groq response to LangChain format
        return {
            "generations": [{
                "text": response.choices[0].message.content,
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                }
            }]
        }

def setup_langchain_llm():
    """Setup and return the LLM using langchain_groq."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=DEFAULT_MODEL
    )

def setup_groq_direct_llm():
    """Setup and return the LLM using the official Groq client."""
    return GroqDirectLLM(
        api_key=GROQ_API_KEY,
        model_name=GROQ_DIRECT_MODEL
    )

def main(topic=DEFAULT_TOPIC):
    """
    Main function to run the agentic workflow.
    
    Args:
        topic: The topic to research, analyze, and report on
    """
    # Initialize the LLM using the direct Groq client
    # This should help avoid the provider prefix issues
    llm = setup_groq_direct_llm()
    
    # Create agents
    researcher = create_researcher_agent(llm)
    analyst = create_analyst_agent(llm)
    report_writer = create_report_writer_agent(llm)
    
    # Create tasks using functions from tasks.py
    research_task = create_research_task(researcher, topic)
    analysis_task = create_analysis_task(analyst, research_task)
    report_task = create_report_task(report_writer, analysis_task)
    
    # Create and run the crew
    crew = Crew(
        agents=[researcher, analyst, report_writer],
        tasks=[research_task, analysis_task, report_task],
        verbose=VERBOSE
    )
    
    result = crew.kickoff()
    print(f"\n\n==== Final Report ====\n\n{result}")
    return result

if __name__ == "__main__":
    main()
