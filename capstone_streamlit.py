from __future__ import annotations

import uuid

import streamlit as st

from agent import DOCUMENTS, StudyBuddyAgent


st.set_page_config(page_title="Study Buddy", page_icon="SB", layout="wide")


@st.cache_resource
def load_agent(groq_api_key: str) -> StudyBuddyAgent:
    return StudyBuddyAgent(groq_api_key=groq_api_key)


def reset_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


if "thread_id" not in st.session_state:
    reset_conversation()

with st.sidebar:
    st.title("Study Buddy")
    st.caption("B.Tech Physics assistant using Groq, LangGraph, ChromaDB, tools, and memory.")
    groq_api_key = st.text_input(
        "Groq API key",
        type="password",
        help="Used only for this running session. It is not saved in the project files.",
    )
    st.button("New conversation", use_container_width=True, on_click=reset_conversation)
    st.divider()
    st.subheader("Knowledge base")
    st.write(f"{len(DOCUMENTS)} physics topics")
    for doc in DOCUMENTS:
        st.markdown(f"- {doc['topic']}")
    st.divider()
    st.subheader("Tools")
    st.markdown("- Local date/time")
    st.markdown("- Safe calculator")

st.title("Study Buddy")
st.caption("Ask about mechanics, optics, thermodynamics, electricity, magnetism, waves, or modern physics.")

if not groq_api_key:
    st.info("Enter your Groq API key in the sidebar to start. Do not hardcode API keys before submitting.")
    st.stop()

try:
    agent = load_agent(groq_api_key)
except Exception as exc:
    st.error(f"Could not initialize Study Buddy: {exc}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

prompt = st.chat_input("Ask a physics question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.ask(prompt, st.session_state.thread_id)
        st.markdown(result["answer"])
        meta = (
            f"Route: `{result.get('route', 'unknown')}` | "
            f"Sources: {', '.join(result.get('sources', [])) or 'none'} | "
            f"Faithfulness: `{result.get('faithfulness', 0):.2f}`"
        )
        st.caption(meta)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "meta": meta})
