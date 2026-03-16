"""
Agentic chatbot engine with multi-turn tool use.
Uses Claude to autonomously decide which tools to call, executes them,
and feeds results back until Claude produces a final text response.
"""
from typing import Dict, List, Tuple, Any
from utils.ai_advisor import client
from utils.chatbot_tools import CHATBOT_TOOLS, execute_tool

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 5


def _build_system_prompt(profile: Dict[str, Any], eli10_mode: bool) -> str:
    risk = profile.get("risk_tolerance", "Moderate")
    horizon = profile.get("horizon", "Medium-term")
    name = profile.get("name", "Investor")

    if eli10_mode:
        return (
            f"You are a super-friendly portfolio assistant talking to {name}. "
            f"Explain everything as if talking to a 10-year-old. Use simple words, "
            f"fun analogies (pizza slices, piggy banks, roller coasters), and keep it encouraging. "
            f"The user's risk tolerance is {risk} and their horizon is {horizon}. "
            f"You have tools to look up real portfolio data, prices, news, and risk metrics. "
            f"Always use the tools to get real data before answering - never guess numbers."
        )
    else:
        return (
            f"You are a professional AI portfolio assistant for {name}. "
            f"Their risk tolerance is {risk} and investment horizon is {horizon}. "
            f"Provide data-driven, precise analysis using your available tools. "
            f"Always call tools to retrieve real data before answering - never fabricate numbers. "
            f"Use markdown formatting for clarity. Be concise but thorough."
        )


def run_chatbot_turn(
    user_message: str,
    conversation_history: List[Dict],
    holdings: Dict[str, float],
    profile: Dict[str, Any],
    eli10_mode: bool = False,
) -> Tuple[str, List[Dict]]:
    """
    Runs one user turn through the agentic loop.

    Returns:
        (final_text, tool_calls_log) where tool_calls_log is a list of
        dicts with {tool_name, tool_input, tool_result} for the transparency panel.
    """
    if not client:
        return "Anthropic API key is missing or invalid. Please check your .env file.", []

    system_prompt = _build_system_prompt(profile, eli10_mode)

    # Append user message to conversation
    conversation_history.append({"role": "user", "content": user_message})

    tool_calls_log = []

    for _ in range(MAX_ITERATIONS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=system_prompt,
                tools=CHATBOT_TOOLS,
                messages=conversation_history,
            )
        except Exception as e:
            return f"Error communicating with Claude: {e}", tool_calls_log

        # Process response content blocks
        assistant_content = response.content
        conversation_history.append({"role": "assistant", "content": assistant_content})

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    # Execute the tool
                    result = execute_tool(tool_name, tool_input, holdings)

                    tool_calls_log.append({
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_result": result,
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                    })

            # Feed tool results back to Claude
            conversation_history.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # Extract the final text response
            final_text = ""
            for block in assistant_content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text, tool_calls_log

    # Safety: if we hit max iterations, return whatever we have
    final_text = "I've gathered a lot of data but hit my analysis limit. Here's what I found so far."
    for block in conversation_history[-1].get("content", []):
        if hasattr(block, "text"):
            final_text = block.text
            break
    return final_text, tool_calls_log
