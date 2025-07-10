import pytest
from unittest.mock import MagicMock, patch

from jentic_agents.communication.hitl.cli_intervention_hub import CLIInterventionHub
from jentic_agents.communication.hitl.discord_intervention_hub import DiscordInterventionHub

# --- Tests for CLIInterventionHub ---

def test_cli_hub_is_always_available():
    """The CLI hub should always report as available."""
    hub = CLIInterventionHub()
    assert hub.is_available() is True

@patch('rich.console.Console.input', return_value="Here is the answer.")
def test_cli_hub_ask_human(mock_input):
    """Test that the CLI hub correctly formats the prompt and returns the user's input."""
    hub = CLIInterventionHub()
    # Mock the print function to avoid console output during tests
    hub._console.print = MagicMock()
    
    question = "What is the meaning of life?"
    context = "The agent is pondering existence."
    
    response = hub.ask_human(question, context)
    
    assert response == "Here is the answer."
    
    # Check that the prompt was printed correctly
    print_calls = hub._console.print.call_args_list
    assert len(print_calls) > 4 # Banners, context, question
    assert any(f"Context:[/bold] {context}" in str(call) for call in print_calls)
    assert any(f"Question:[/bold] {question}" in str(call) for call in print_calls)
    mock_input.assert_called_once_with("Your response: ")


# --- Tests for DiscordInterventionHub ---

@pytest.fixture
def mock_discord_bot():
    """Fixture to create a mock Discord bot."""
    bot = MagicMock()
    bot.is_ready.return_value = True
    bot.get_user.return_value = MagicMock()
    bot.fetch_user.return_value = MagicMock()
    bot._connection.loop.is_running.return_value = True
    return bot

def test_discord_hub_availability(mock_discord_bot):
    """Test the is_available method for the Discord hub."""
    # Available when ready and user ID is set
    hub_available = DiscordInterventionHub(bot=mock_discord_bot, target_user_id=12345)
    assert hub_available.is_available() is True
    
    # Not available if bot is not ready
    mock_discord_bot.is_ready.return_value = False
    hub_not_ready = DiscordInterventionHub(bot=mock_discord_bot, target_user_id=12345)
    assert hub_not_ready.is_available() is False
    
    # Not available if user ID is missing
    mock_discord_bot.is_ready.return_value = True
    hub_no_user = DiscordInterventionHub(bot=mock_discord_bot, target_user_id=None)
    assert hub_no_user.is_available() is False

@patch('asyncio.run_coroutine_threadsafe')
def test_discord_hub_ask_human(mock_run_threadsafe, mock_discord_bot):
    """Test the main ask_human entrypoint for the Discord hub."""
    hub = DiscordInterventionHub(bot=mock_discord_bot, target_user_id=12345)
    
    # Mock the future that run_coroutine_threadsafe returns
    mock_future = MagicMock()
    mock_future.result.return_value = "Human says yes."
    mock_run_threadsafe.return_value = mock_future
    
    response = hub.ask_human("Should I proceed?")
    
    assert response == "Human says yes."
    # Check that the async method was called in the bot's event loop
    mock_run_threadsafe.assert_called_once()
    assert mock_run_threadsafe.call_args[0][1] == mock_discord_bot._connection.loop 