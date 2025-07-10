"""
Tests for communication outboxes.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import discord
import asyncio

from jentic_agents.communication.outbox.cli_outbox import CLIOutbox
from jentic_agents.communication.outbox.discord_outbox import DiscordOutbox
from jentic_agents.reasoners.base_reasoner import ReasoningResult

# --- Force `anyio` to use the asyncio backend for this module ---
@pytest.fixture
def anyio_backend():
    return 'asyncio'
    
# --- Tests for CLIOutbox ---

@patch('jentic_agents.communication.outbox.cli_outbox.console')
def test_cli_outbox_displays(mock_console):
    """Test that CLIOutbox calls the console to display messages."""
    outbox = CLIOutbox(verbose=True)

    outbox.display_welcome()
    mock_console.print.assert_called()

    outbox.display_goal_start("goal")
    mock_console.print.assert_called_with("[green]Processing goal:[/green] goal  [dim](working...)[/dim]")

    success_result = ReasoningResult(final_answer="Success!", success=True, iterations=1, tool_calls=[])
    outbox.display_reasoning_result(success_result)
    # The result is a Panel object, its content is in the `renderable` attribute
    assert "✅" in str(mock_console.print.call_args[0][0].renderable)

    fail_result = ReasoningResult(final_answer="Failure!", success=False, error_message="It failed", iterations=1, tool_calls=[])
    outbox.display_reasoning_result(fail_result)
    assert "❌" in str(mock_console.print.call_args[0][0].renderable)

# --- Tests for DiscordOutbox ---

@pytest.fixture
def mock_discord_bot():
    """Provides a mock discord bot object with a running event loop."""
    bot = Mock()
    bot.loop = Mock() # The loop is passed to a patched function, so it doesn't need to be real.
    # Mock the async methods needed
    channel = Mock()
    channel.send = AsyncMock(return_value=None)
    bot.get_channel.return_value = channel
    return bot

def consuming_side_effect(coro):
    """'Consumes' a coroutine by closing it, preventing a RuntimeWarning."""
    coro.close()
    # We still need to return a mock task object if the code uses it
    return Mock()

@patch('jentic_agents.communication.outbox.discord_outbox.asyncio.create_task')
def test_discord_outbox_schedules_messages(mock_create_task, mock_discord_bot):
    """Test that DiscordOutbox schedules messages to be sent."""
    mock_create_task.side_effect = consuming_side_effect
    outbox = DiscordOutbox(bot=mock_discord_bot, default_channel_id=123, use_embeds=True)
    
    outbox.display_welcome()
    
    mock_create_task.assert_called_once()
    # Check that the coroutine passed to it is the one we expect
    coro = mock_create_task.call_args[0][0]
    assert coro.__name__ == '_send_message'
    