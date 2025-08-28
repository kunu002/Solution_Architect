from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

llm = AzureChatOpenAI(azure_deployment="gpt-4o", temperature=float(0.2), api_key=os.getenv("AZURE_API_KEY"), azure_endpoint=os.getenv("AZURE_ENDPOINT"), openai_api_version="2024-08-01-preview")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")