"""
Definitions of agents for the CrewAI workflow.
"""
from crewai import Agent
from crewai_tools import SerperDevTool
from config import SERPER_API_KEY, VERBOSE

def create_researcher_agent(llm):
    """
    Creates a researcher agent who gathers information.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        An Agent instance configured for research tasks
    """
    search_tool = SerperDevTool(api_key=SERPER_API_KEY)
    
    return Agent(
        role="Data Researcher",
        goal="Research and gather comprehensive information on the given topic",
        backstory="""You are an expert researcher with a keen eye for detail and 
        the ability to find information from various sources. You excel at 
        collecting and organizing data for analysis.""",
        tools=[search_tool],
        llm=llm,
        verbose=VERBOSE
    )

def create_analyst_agent(llm):
    """
    Creates an analyst agent who processes and analyzes information.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        An Agent instance configured for analysis tasks
    """
    return Agent(
        role="Data Analyst",
        goal="Analyze data and extract meaningful insights",
        backstory="""You are a skilled data analyst with expertise in identifying patterns
        and drawing conclusions from complex datasets. You transform raw data into
        actionable insights.""",
        llm=llm,
        verbose=VERBOSE
    )

def create_report_writer_agent(llm):
    """
    Creates a report writer agent who produces clear and concise reports.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        An Agent instance configured for report writing tasks
    """
    return Agent(
        role="Report Writer",
        goal="Create clear, concise, and informative reports",
        backstory="""You are an expert communicator who can distill complex information
        into clear, engaging reports. You know how to structure information to maximize
        impact and understanding.""",
        llm=llm,
        verbose=VERBOSE
    )