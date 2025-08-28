# state.py
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ChatState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    phase: Optional[str]        # "start" | "solution" | "architect" | "analysis" | "done"
    awaiting_confirm: bool      # supervisor asks user to confirm next step
    route: Optional[str]        # "solution_agent" | "architect_agent" | "analysis_agent" | "END"
    # for CLI printing control (optional)
    last_printed_ai_messages: List[BaseMessage]
