"""
Definitions of tasks for the CrewAI workflow.
"""
from crewai import Task

def create_research_task(agent, topic):
    """
    Creates a research task for gathering information on a specific topic.
    
    Args:
        agent: The agent to assign this task to
        topic: The topic to research
        
    Returns:
        A Task instance for research
    """
    return Task(
        description=f"Research and gather comprehensive information about {topic}. "
                   f"Find the latest trends, key statistics, and notable developments. "
                   f"Focus on factual information from reliable sources.",
        agent=agent,
        expected_output="A detailed research document with facts, figures, and key insights about the topic."
    )

def create_analysis_task(agent, research_output):
    """
    Creates an analysis task for processing research findings.
    
    Args:
        agent: The agent to assign this task to
        research_output: The context from the research task
        
    Returns:
        A Task instance for analysis
    """
    return Task(
        description=f"Analyze the following research findings and extract key insights: \n{research_output}\n"
                   f"Identify patterns, trends, and significant points. Highlight any surprising "
                   f"or counterintuitive information.",
        agent=agent,
        context=[research_output],
        expected_output="An analytical report with key insights, patterns identified, and significant conclusions."
    )

def create_report_task(agent, analysis_output):
    """
    Creates a report writing task for producing a final document.
    
    Args:
        agent: The agent to assign this task to
        analysis_output: The context from the analysis task
        
    Returns:
        A Task instance for report generation
    """
    return Task(
        description=f"Create a clear, concise report based on the following analysis: \n{analysis_output}\n"
                   f"Structure the report with an executive summary, key findings, supporting evidence, "
                   f"and recommendations. Use a professional tone and ensure the report is accessible "
                   f"to non-technical readers.",
        agent=agent,
        context=[analysis_output],
        expected_output="A well-structured, professional report ready for presentation to stakeholders."
    )