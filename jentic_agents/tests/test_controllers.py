"""
Tests for communication controllers.
"""

import pytest
from unittest.mock import Mock, patch
from jentic_agents.communication.controllers.base_controller import BaseController
from jentic_agents.communication.controllers.cli_controller import CLIController
from jentic_agents.communication.controllers.discord_controller import DiscordController
from jentic_agents.communication.inbox.cli_inbox import CLIInbox
from jentic_agents.communication.outbox.cli_outbox import CLIOutbox
from jentic_agents.communication.hitl.cli_intervention_hub import CLIInterventionHub

# --- CLIController Tests ---

def test_cli_controller_initialization():
    """Test that CLIController initializes its components and closes them."""
    # 1. Create mocks for the components
    mock_inbox = Mock(spec=CLIInbox)
    mock_outbox = Mock(spec=CLIOutbox)
    mock_hub = Mock(spec=CLIInterventionHub)
    
    # 2. Initialize the controller with the mocks
    controller = CLIController(
        inbox=mock_inbox,
        outbox=mock_outbox,
        intervention_hub=mock_hub
    )
    
    # 3. Assert that the components are set correctly
    assert controller.inbox is mock_inbox
    assert controller.outbox is mock_outbox
    assert controller.intervention_hub is mock_hub

    # 4. Test that close() calls close() on components
    controller.close()
    mock_inbox.close.assert_called_once()
    mock_outbox.close.assert_called_once()
    mock_hub.close.assert_called_once()

def test_controller_factory():
    """Test that the factory method returns a CLIController."""
    controller, _, _ = CLIController.create_controller("cli")

# --- Tests for DiscordController ---

@pytest.fixture
def mock_discord_bot():
    """Provides a mock discord bot object."""
    bot = Mock()
    bot.event = lambda func: func  # Decorator passthrough
    bot.is_ready.return_value = True
    return bot

@patch('os.getenv', return_value="fake_token")
@patch('jentic_agents.utils.config.get_config_value', return_value=12345)
@patch('discord.Client')
def test_discord_controller_creation(mock_client, mock_get_config, mock_getenv, mock_discord_bot):
    """Test the factory method for DiscordController."""
    mock_client.return_value = mock_discord_bot
    
    controller, bot, token = DiscordController.create_controller(mode='discord')
    
    assert isinstance(controller, DiscordController)
    assert token == "fake_token"
    mock_get_config.assert_called()
    mock_getenv.assert_called_with("DISCORD_BOT_AGENT_TOKEN")

def test_discord_controller_initialization(mock_discord_bot):
    """Test that DiscordController initializes its components."""
    controller = DiscordController(bot=mock_discord_bot, target_user_id=123)
    assert controller.inbox is not None
    assert controller.outbox is not None
    assert controller.intervention_hub is not None
    assert controller.bot is mock_discord_bot

def test_discord_controller_delegates_display(mock_discord_bot):
    """Test that display methods are delegated to the outbox."""
    controller = DiscordController(bot=mock_discord_bot, target_user_id=123)
    controller.outbox = Mock()
    
    controller.display_welcome()
    controller.outbox.display_welcome.assert_called_once()
    
    controller.display_goal_start("goal")
    controller.outbox.display_goal_start.assert_called_once_with("goal", None) 