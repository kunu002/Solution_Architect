# tools.py
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from llm_config import TAVILY_API_KEY

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a short summary."""
    import wikipedia
    try:
        return wikipedia.summary(query, sentences=3)
    except wikipedia.exceptions.PageError:
        return "No Wikipedia results found."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation error. Try a more specific query from this list: {e.options}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def tavily_search(query: str) -> str:
    """Perform a Tavily web search and return a brief summary of top results."""
    tavily = TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)
    try:
        return tavily.run(query)
    except Exception as e:
        return f"An error occurred with Tavily search: {str(e)}"

@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for related papers and return a brief summary."""
    from langchain_community.tools import ArxivQueryRun
    from langchain_community.utilities import ArxivAPIWrapper
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    try:
        return arxiv.run(query)
    except Exception as e:
        return f"An error occurred with Arxiv search: {str(e)}"

# Exported tool lists for robust imports
TOOLS = [wikipedia_search, tavily_search, arxiv_search]

def get_tools():
    """Preferred accessor for tools (used by agents)."""
    return TOOLS
