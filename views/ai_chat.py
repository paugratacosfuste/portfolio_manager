"""
AI Chat view -- Agentic chatbot with tool-use transparency.
"""
import streamlit as st
import json
from utils.chatbot_engine import run_chatbot_turn


EXAMPLE_PROMPTS = [
    "How is my portfolio doing?",
    "What if I add 10 shares of TSLA?",
    "How risky is my portfolio?",
    "What's the market outlook?",
    "Get me news on AAPL",
    "What if I remove all my crypto?",
]


def render_ai_chat():
    st.markdown("<h1>AI Portfolio Chat</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>Ask me anything about your portfolio. I can look up prices, calculate risk, "
        "simulate changes, fetch news, and check the macro outlook — all in real time.</p>",
        unsafe_allow_html=True,
    )

    holdings = st.session_state.holdings
    profile = st.session_state.profile
    eli10_mode = st.session_state.get("eli10_mode", False)

    # Initialize chat state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # API-format messages
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []  # UI display messages

    # Clear button
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.chat_display = []
            st.rerun()

    # Show example prompts when chat is empty
    if not st.session_state.chat_display:
        st.markdown("#### Try asking:")
        cols = st.columns(3)
        for i, prompt in enumerate(EXAMPLE_PROMPTS):
            with cols[i % 3]:
                if st.button(prompt, key=f"example_{i}"):
                    st.session_state.pending_prompt = prompt
                    st.rerun()

    # Display chat history
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.status("Tools Called", state="complete"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**{tc['tool_name']}**({json.dumps(tc['tool_input'])})")
                        try:
                            parsed = json.loads(tc["tool_result"])
                            st.json(parsed)
                        except (json.JSONDecodeError, TypeError):
                            st.code(tc["tool_result"])

    # Handle pending prompt from example buttons
    pending = st.session_state.pop("pending_prompt", None)

    # Chat input
    user_input = st.chat_input("Ask about your portfolio...")

    prompt_to_process = pending or user_input

    if prompt_to_process:
        # Display user message
        st.session_state.chat_display.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Run agentic loop
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text, tool_calls_log = run_chatbot_turn(
                    user_message=prompt_to_process,
                    conversation_history=st.session_state.chat_history,
                    holdings=holdings,
                    profile=profile,
                    eli10_mode=eli10_mode,
                )

            st.markdown(response_text)

            # Show tool calls transparency
            if tool_calls_log:
                with st.status(f"Tools Called ({len(tool_calls_log)})", state="complete"):
                    for tc in tool_calls_log:
                        st.markdown(f"**{tc['tool_name']}**({json.dumps(tc['tool_input'])})")
                        try:
                            parsed = json.loads(tc["tool_result"])
                            st.json(parsed)
                        except (json.JSONDecodeError, TypeError):
                            st.code(tc["tool_result"])

        # Save to display history
        st.session_state.chat_display.append({
            "role": "assistant",
            "content": response_text,
            "tool_calls": tool_calls_log,
        })
