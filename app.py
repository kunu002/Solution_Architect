# app.py
import streamlit as st
from graph_builder import build_graph
from state import ChatState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.errors import GraphRecursionError

st.set_page_config(page_title="Solution Architect", layout="centered", page_icon="💡")

# --- Init ---
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "chat_state" not in st.session_state:
    st.session_state.chat_state = ChatState(messages=[], phase="start", awaiting_confirm=False, route=None)

st.markdown(
    "<h2 style='text-align:center;'>💡 Solution Architect</h2>",
    unsafe_allow_html=True
)

# --- Chat history ---
for m in st.session_state.chat_state["messages"]:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# --- Run Graph Helper ---
def run_graph():
    try:
        result = st.session_state.graph.invoke(st.session_state.chat_state)
        st.session_state.chat_state.update(result)
    except GraphRecursionError:
        st.error("⚠️ Paused: waiting for confirmation.")

# --- User input ---
if user_input := st.chat_input("Type here..."):
    st.session_state.chat_state["messages"].append(HumanMessage(content=user_input))
    run_graph()
    st.rerun()
