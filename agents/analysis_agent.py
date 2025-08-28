# agents/analysis_agent.py

from typing import List, Optional
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.types import Command
from llm_config import llm
from tools.tools import get_tools 

# Optional tool-agent support
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    AGENT_IMPORTS_OK = True
except Exception:
    AGENT_IMPORTS_OK = False

def _load_tools():
    try:
        return get_tools()
    except Exception:
        pass
    try:
        from tools import TOOLS  # type: ignore
        return TOOLS
    except Exception:
        pass
    # 🚫 Do NOT import `tools` module directly
    return []


CONTROL_TOKENS = {"yes", "no", "new", "end"}

SUP_CONFIRM_MARKERS = [
    "Do you want to proceed to the **Architecture** phase?",
    "Do you want to proceed to the **Analysis/Research** phase?",
    "The analysis phase is complete with research insights.",
    "Please reply with 'yes' to proceed, 'no' to revise",
    "Let's start fresh. What new solution or requirement do you have?",
]

def _extract_real_user_query(msgs: List[BaseMessage]) -> Optional[str]:
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if isinstance(m, HumanMessage):
            txt = (m.content or "").strip()
            low = txt.lower()
            if low in CONTROL_TOKENS:
                continue
            if i > 0 and isinstance(msgs[i - 1], AIMessage):
                prev_c = str(msgs[i - 1].content)
                if any(marker in prev_c for marker in SUP_CONFIRM_MARKERS) and len(txt) <= 8:
                    continue
            return txt
    return None

def _collect_context_from_ai(msgs: List[BaseMessage], max_chars: int = 2000) -> str:
    chunks: List[str] = []
    for m in msgs:
        if isinstance(m, AIMessage):
            c = str(m.content)
            if any(marker in c for marker in SUP_CONFIRM_MARKERS):
                continue
            chunks.append(c)

    # take last 3 AI chunks
    trimmed = "\n\n".join(chunks[-3:])
    # hard trim to avoid token bloat
    return trimmed[-max_chars:]

def _run_tool_agent(user_query: str, context_text: str) -> str:
    tools = _load_tools()
    if not (AGENT_IMPORTS_OK and tools):
        # Fallback: direct LLM with structured markdown
        prompt = (
            "You are an analysis & research agent.\n"
            "Respond **concisely** and **structurally** in Markdown using headings and bullet points.\n"
            "Only include relevant, high-signal information.\n\n"
            f"# Context\n{context_text}\n\n"
            f"# User Query\n{user_query}\n\n"
            "Provide a short, structured analysis with key findings, options, and risks."
        )
        return llm.invoke(prompt).content  # type: ignore

    system_msg = (
        "You are a senior analysis & research agent. "
        "Use tools to verify facts and gather evidence. "
        "Reply concisely, in Markdown, with headings, bullet points, and short paragraphs."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("system", "Context:\n{context_text}"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    result = executor.invoke({"input": user_query, "context_text": context_text})

    # Optional: print intermediate steps to make CLI non-empty
    inter = result.get("intermediate_steps", []) if isinstance(result, dict) else []
    if inter:
        print("\n--- DEBUG: Intermediate tool steps ---")
        for i, step in enumerate(inter, 1):
            print(f"[{i}] {step}")
        print("--------------------------------------\n")

    if isinstance(result, dict) and "output" in result:
        return result["output"]
    return str(result)

def analysis_agent(state):
    print("\n--- DEBUG: Entering analysis_agent ---")
    msgs: List[BaseMessage] = state.get("messages", [])
    phase: str = state.get("phase") or "analysis"
    awaiting: bool = state.get("awaiting_confirm", False)
    print(f"  Phase: {phase}, Awaiting Confirm: {awaiting}, Route: analysis_agent")
    print(f"  Total messages in state: {len(msgs)}")

    user_query = _extract_real_user_query(msgs) or \
        next((m.content for m in msgs if isinstance(m, HumanMessage)), "Please continue the analysis.")
    context_text = _collect_context_from_ai(msgs)

    print("\n> Entering AgentExecutor chain...\n")
    final_output = _run_tool_agent(user_query, context_text)
    print("\n> Finished chain.\n")

    analysis_msg = AIMessage(content=final_output)
    new_msgs = msgs + [analysis_msg]

    return Command(update={"messages": new_msgs})
