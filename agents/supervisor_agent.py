# agents/supervisor_agent.py

from typing import Optional, List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm_config import llm
from langgraph.types import Command

# --- Confirmation texts (keep these exact; used to detect confirmation flow) ---
CONFIRM_SOL_TEXT = (
    "I've outlined a step-by-step solution. Do you want to proceed to the **Architecture** phase? "
    "Reply `yes` to proceed, `no` to revise this solution, 'new' to start a new request, or 'end' to finish."
)
CONFIRM_ARCH_TEXT = (
    "Here's the high-level technical architecture. Do you want to proceed to the **Analysis/Research** phase? "
    "Reply `yes` to proceed, `no` to revise this architecture, 'new' to start a new request, or 'end' to finish."
)
CONFIRM_ANALYSIS_TEXT = (
    "The analysis phase is complete with research insights. Do you have any other questions, "
    "or would you like to start a new request? Reply with your query, 'new' to start a new solution, or 'end' to finish."
)
CLARIFICATION_TEXT = (
    "I'm sorry, I didn't quite understand your last response. "
    "Please reply with 'yes' to proceed, 'no' to revise, 'new' to start a new solution, or 'end' to finish. "
    "If you have a different question, please state it clearly."
)
ASK_NEW_QUERY_TEXT = "Okay, let's start fresh. What new solution or requirement do you have?"

CONTROL_TOKENS = {"yes", "no", "new", "end"}

class RouteDecision(BaseModel):
    decision: str = Field(
        ...,
        description="One of: 'proceed_to_next_phase', 'revise_current_phase', 'start_new_query', 'end_session', 'clarify'",
    )

def _latest_human_after_confirm(msgs: List[BaseMessage]) -> Optional[HumanMessage]:
    for i in range(len(msgs) - 1, 0, -1):
        m = msgs[i]
        if isinstance(m, HumanMessage):
            prev = msgs[i - 1]
            if isinstance(prev, AIMessage):
                prev_c = str(prev.content)
                if (
                    CONFIRM_SOL_TEXT in prev_c
                    or CONFIRM_ARCH_TEXT in prev_c
                    or CONFIRM_ANALYSIS_TEXT in prev_c
                    or CLARIFICATION_TEXT in prev_c
                ):
                    return m
    return None

def supervisor_agent(state):
    print("\n--- DEBUG SUPERVISOR ENTERED ---")

    msgs: List[BaseMessage] = state.get("messages", [])
    phase: str = state.get("phase") or "start"
    awaiting: bool = state.get("awaiting_confirm", False)

    # ---------- INITIAL ROUTE ----------
    if phase == "start":
        last_h = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        if last_h:
            txt = (last_h.content or "").lower().strip()
            if txt not in CONTROL_TOKENS:
                return Command(update={
                    "phase": "solution",
                    "route": "solution_agent",
                    "awaiting_confirm": False,
                })
        return Command(update={
            "messages": msgs + [AIMessage(content=ASK_NEW_QUERY_TEXT)],
            "awaiting_confirm": False,
            "route": "END",
        })

    # ---------- AFTER ANY WORKER: ASK CONFIRMATION ----------
    if phase in ["solution", "architect", "analysis"] and not awaiting:
        if phase == "solution":
            confirm_text = CONFIRM_SOL_TEXT
        elif phase == "architect":
            confirm_text = CONFIRM_ARCH_TEXT
        else:
            confirm_text = CONFIRM_ANALYSIS_TEXT

        print(f"  Supervisor: Worker finished phase '{phase}'. Asking for confirmation.")
        return Command(update={
            "messages": msgs + [AIMessage(content=confirm_text)],
            "awaiting_confirm": True,
            "route": "END",
        })

    # ---------- HANDLE CONFIRMATION REPLY ----------
    if awaiting:
        reply_msg = _latest_human_after_confirm(msgs)
        if not reply_msg:
            print("  Supervisor: Awaiting confirmation, no new human input yet. Ending run.")
            return Command(update={"route": "END"})

        user_text = (reply_msg.content or "").lower().strip()
        decision_chain = (
            ChatPromptTemplate.from_template(
                "Classify message: {message}\n"
                "Options: proceed_to_next_phase, revise_current_phase, start_new_query, end_session, clarify"
            )
            | llm.with_structured_output(RouteDecision)
        )
        decision = decision_chain.invoke({"message": user_text}).decision
        print(f"  Supervisor: User replied '{user_text}' → interpreted as '{decision}'.")

        if decision == "proceed_to_next_phase":
            if phase == "solution":
                return Command(update={
                    "phase": "architect",
                    "route": "architect_agent",
                    "awaiting_confirm": False,
                })
            elif phase == "architect":
                return Command(update={
                    "phase": "analysis",
                    "route": "analysis_agent",
                    "awaiting_confirm": False,
                })
            else:
                return Command(update={
                    "phase": "done",
                    "route": "END",
                    "awaiting_confirm": False,
                })

        elif decision == "revise_current_phase":
            return Command(update={
                "phase": phase,
                "route": f"{phase}_agent",
                "awaiting_confirm": False,
            })

        elif decision == "start_new_query":
            return Command(update={
                "phase": "start",
                "route": "END",
                "awaiting_confirm": False,
                "messages": msgs + [AIMessage(content=ASK_NEW_QUERY_TEXT)],
            })

        elif decision == "end_session":
            return Command(update={
                "phase": "done",
                "route": "END",
                "awaiting_confirm": False,
            })

        else:  # clarify
            return Command(update={
                "messages": msgs + [AIMessage(content=CLARIFICATION_TEXT)],
                "awaiting_confirm": True,
                "route": "END",
            })

    return Command(update={"route": "END"})
