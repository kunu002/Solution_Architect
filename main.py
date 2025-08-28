# main.py
from graph_builder import build_graph
from state import ChatState
from langchain_core.messages import HumanMessage, AIMessage

def run_chatbot():
    graph = build_graph()

    state: ChatState = {
        "messages": [],
        "phase": "start",
        "awaiting_confirm": False,
        "route": None,
        "last_printed_ai_messages": []
    }

    print("\n--- Chatbot Started ---")
    print("Hello! I'm here to help you with step by step guide for everything. What's your requirement?")

    while True:
        # If awaiting confirmation, the supervisor already printed a question.
        # We just need to capture the user's response.
        prompt = "User (reply to confirmation): " if state.get("awaiting_confirm") else "User: "
        user_input = input(prompt)

        # Append user's message to the state
        state["messages"].append(HumanMessage(content=user_input))

        # Debugging: Print state before invocation
        print(f"\n--- State before invoke ---")
        print(f"Phase: {state['phase']}, Awaiting Confirm: {state['awaiting_confirm']}, Route: {state['route']}")
        print(f"Last User Message: {user_input}")
        print("--------------------------")

        # Stream the graph to see tool calls / steps in real time
        latest_state = None
        for event in graph.stream(state):
            latest_state = event
            # Show incremental AI messages
            if "messages" in event:
                for msg in reversed(event["messages"]):
                    if msg.type == "ai" and msg not in state.get("last_printed_ai_messages", []):
                        print("🤖", msg.content)
                        break

        # Merge updates rather than replacing entire dict
        if latest_state:
            state.update(latest_state)

        # Track printed AI messages to avoid duplicates
        state["last_printed_ai_messages"] = [m for m in state["messages"] if m.type == "ai"]

        # Debugging: Print state after invocation
        print(f"\n--- State after invoke ---")
        print(f"Phase: {state['phase']}, Awaiting Confirm: {state['awaiting_confirm']}, Route: {state['route']}")
        print("--------------------------")

        if state.get("phase") == "done":
            print("✅ Flow complete. Thank you for using the chatbot!")
            break

if __name__ == "__main__":
    run_chatbot()
