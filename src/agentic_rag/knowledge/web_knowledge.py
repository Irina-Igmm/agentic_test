"""
Web knowledge module for retrieving information from the web.
"""

from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

from ..config import SERPER_API_KEY
from ..llm.llm_provider import crew_llm

from urllib.parse import urlparse

def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.

    Args:
        url: The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    if not url:
        return False
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within the token limit.

    Args:
        text: The text to truncate.
        max_tokens: The maximum number of tokens allowed.

    Returns:
        str: The truncated text.
    """
    return text[:max_tokens]

class WebKnowledge:
    """
    Handles operations related to web knowledge retrieval using CrewAI.
    """
    
    @staticmethod
    def setup_web_scraping_agent():
        """
        Set up the web scraping agent and related components.
        
        Returns:
            Crew: The configured CrewAI crew for web operations.
        """
        search_tool = SerperDevTool()  # Tool for performing web searches
        scrape_website = ScrapeWebsiteTool()  # Tool for extracting data from websites
        
        # Define the web search agent
        web_search_agent = Agent(
            role="Expert Web Search Agent",
            goal="Identify and retrieve relevant web data for user queries",
            backstory="An expert in identifying valuable web sources for the user's needs",
            allow_delegation=False,
            verbose=True,
            llm=crew_llm
        )
        
        # Define the web scraping agent
        web_scraper_agent = Agent(
            role="Expert Web Scraper Agent",
            goal="Extract and analyze content from specific web pages identified by the search agent",
            backstory="A highly skilled web scraper, capable of analyzing and summarizing website content accurately",
            allow_delegation=False,
            verbose=True,
            llm=crew_llm
        )
        
        # Define the web search task
        search_task = Task(
            description=(
                "Identify the most relevant web page or article for the topic: '{topic}'. "
                "Use all available tools to search for and provide a link to a web page "
                "that contains valuable information about the topic. Keep your response concise."
            ),
            expected_output=(
                "A concise summary of the most relevant web page or article for '{topic}', "
                "including the link to the source and key points from the content."
            ),
            tools=[search_tool],
            agent=web_search_agent,
        )
        
        # Define the web scraping task
        scraping_task = Task(
            description=(
                "Extract and analyze data from the given web page or website. Focus on the key sections "
                "that provide insights into the topic: '{topic}'. Use all available tools to retrieve the content, "
                "and summarize the key findings in a concise manner."
            ),
            expected_output=(
                "A detailed summary of the content from the given web page or website, highlighting the key insights "
                "and explaining their relevance to the topic: '{topic}'. Ensure clarity and conciseness."
            ),
            tools=[scrape_website],
            agent=web_scraper_agent,
        )
        
        # Define the crew to manage agents and tasks
        crew = Crew(
            agents=[web_search_agent, web_scraper_agent],
            tasks=[search_task, scraping_task],
            verbose=1,
            memory=False,
        )
        return crew

    @staticmethod
    def get_content(query: str, specific_urls: Optional[List[str]] = None) -> str:
        """
        Get content from web scraping.
        
        Args:
            query: The query to search for.
            specific_urls: Optional list of specific URLs to scrape instead of searching.
            
        Returns:
            str: The scraped and summarized web content.
        """
        # If specific URLs are provided, scrape them directly
        if specific_urls:
            print("urls: ", specific_urls)
            # scrape_tool = ScrapeWebsiteTool(api_key=SERPER_API_KEY)
            combined_content = []
            
            for url in specific_urls:
                if not url or not url.startswith("http"):
                    print(f"Invalid URL: {url}. Skipping...")
                    continue
                try:
                    # Appel de la méthode correcte
                    tool = ScrapeWebsiteTool(website_url=url)
                    content = tool.run()  # Utilisation de l'argument nommé 'url'
                    truncated_content = truncate_text(content, max_tokens=6000)
                    combined_content.append(f"Source: {url}\n{truncated_content}")
                except Exception as e:
                    print(f"Error scraping {url}: {e}")
            
            return "\n\n".join(combined_content) if combined_content else ""
        
        # Otherwise, use the CrewAI crew for web search and scraping
        try:
            crew = WebKnowledge.setup_web_scraping_agent()
            result = crew.kickoff(inputs={"topic": query})
            return result.raw
        except Exception as e:
            print(f"Error in web scraping: {e}")
            return ""