# agents/solution_agent.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from llm_config import llm

def solution_agent(state):
    print("\n--- DEBUG: Entering solution_agent ---")
    print(f"  Phase: {state.get('phase')}, Awaiting Confirm: {state.get('awaiting_confirm')}, Route: {state.get('route')}")
    print(f"  Total messages in state: {len(state['messages'])}")

    messages = state["messages"]

    # Find core requirement
    core_query = ""
    for msg in messages:
        if isinstance(msg, HumanMessage) and not (msg.content.lower().strip() in ["yes", "no", "new", "end", ""] or len(msg.content.split()) < 5):
            core_query = msg.content
            break
    if not core_query:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                core_query = msg.content
                break

    if not core_query:
        return {"messages": messages + [AIMessage(content="I need a clear requirement to provide a solution.")]}

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a Solution Architect Agent.\n"
             "Respond concisely in **Markdown**.\n"
             "Structure as: 1) Summary, 2) Step-by-step plan, 3) Trade-offs, 4) Risks/assumptions."),
            ("user", "Requirement: {requirement}")
        ]
    )

    try:
        response_content = llm.invoke(
            prompt_template.format_messages(requirement=core_query)
        ).content
        final_message = AIMessage(content=response_content)
        return {"messages": messages + [final_message]}
    except Exception as e:
        error_message = f"Solution Agent Error: {str(e)}"
        return {"messages": messages + [AIMessage(content=error_message)]}
