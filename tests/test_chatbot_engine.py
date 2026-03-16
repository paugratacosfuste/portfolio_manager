import pytest
from unittest.mock import patch, MagicMock
from utils.chatbot_engine import run_chatbot_turn


def _make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(name, input_dict, tool_id="tool_123"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_dict
    block.id = tool_id
    return block


@patch('utils.chatbot_engine.client')
def test_simple_text_response(mock_client):
    """Test a turn where Claude responds with text only (no tool use)."""
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [_make_text_block("Your portfolio looks great!")]
    mock_client.messages.create.return_value = mock_response

    text, tool_log = run_chatbot_turn(
        "How is my portfolio?",
        [],
        {'AAPL': 10},
        {'name': 'Test', 'risk_tolerance': 'Moderate', 'horizon': 'Mid-term'},
    )

    assert "portfolio looks great" in text
    assert len(tool_log) == 0
    mock_client.messages.create.assert_called_once()


@patch('utils.chatbot_engine.execute_tool')
@patch('utils.chatbot_engine.client')
def test_tool_use_then_text(mock_client, mock_execute):
    """Test a turn where Claude calls a tool, then responds with text."""
    # First call: Claude wants to use a tool
    tool_response = MagicMock()
    tool_response.stop_reason = "tool_use"
    tool_response.content = [
        _make_tool_use_block("get_portfolio_summary", {})
    ]

    # Second call: Claude responds with text after receiving tool result
    text_response = MagicMock()
    text_response.stop_reason = "end_turn"
    text_response.content = [_make_text_block("Based on the data, your portfolio is worth $50,000.")]

    mock_client.messages.create.side_effect = [tool_response, text_response]
    mock_execute.return_value = '{"total_value": 50000, "holdings": []}'

    text, tool_log = run_chatbot_turn(
        "Show me my portfolio",
        [],
        {'AAPL': 10},
        {'name': 'Test', 'risk_tolerance': 'Moderate', 'horizon': 'Mid-term'},
    )

    assert "$50,000" in text
    assert len(tool_log) == 1
    assert tool_log[0]['tool_name'] == 'get_portfolio_summary'
    assert mock_client.messages.create.call_count == 2


@patch('utils.chatbot_engine.client', None)
def test_no_client():
    """Test that missing API key returns error message."""
    text, tool_log = run_chatbot_turn(
        "Hello",
        [],
        {'AAPL': 10},
        {'name': 'Test', 'risk_tolerance': 'Moderate', 'horizon': 'Mid-term'},
    )

    assert "API key" in text
    assert len(tool_log) == 0
