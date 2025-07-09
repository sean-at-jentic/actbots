"""
Discord implementation of BaseController that aggregates Discord-specific communication channels.
"""
from typing import Optional, Union, List
import discord
from discord.ext import commands
import logging

from .base_controller import BaseController
from .inbox.discord_inbox import DiscordInbox
from .hitl.discord_intervention_hub import DiscordInterventionHub
from .outbox.discord_outbox import DiscordOutbox


class DiscordController(BaseController):
    """
    Discord-specific communication controller.
    
    Aggregates Discord implementations of inbox, intervention hub, and outbox
    for Discord-based agent interactions. Handles Discord bot setup and
    provides a unified interface for Discord-based agent communication.
    """
    
    def __init__(
        self,
        bot: Union[discord.Client, commands.Bot],
        target_user_id: int,
        monitored_channels: Optional[List[int]] = None,
        default_channel_id: Optional[int] = None,
        escalation_channel_id: Optional[int] = None,
        notification_user_ids: Optional[List[int]] = None,
        inbox: Optional[DiscordInbox] = None,
        intervention_hub: Optional[DiscordInterventionHub] = None,
        outbox: Optional[DiscordOutbox] = None,
        command_prefix: str = "!",
        auto_react: bool = True,
        use_embeds: bool = True,
        verbose: bool = True,
        escalation_timeout: int = 300
    ):
        """
        Initialize Discord controller with optional custom implementations.
        
        Args:
            bot: Discord bot instance to use for all communications
            target_user_id: Discord user ID to send escalation messages to
            monitored_channels: List of channel IDs to monitor for goals (None = all channels)
            default_channel_id: Default channel ID for outbox messages
            escalation_channel_id: Channel ID for escalation messages (None = use DM)
            notification_user_ids: List of user IDs to notify for important messages
            inbox: Custom Discord inbox (defaults to DiscordInbox)
            intervention_hub: Custom Discord intervention hub (defaults to DiscordInterventionHub)
            outbox: Custom Discord outbox (defaults to DiscordOutbox)
            command_prefix: Prefix for Discord bot commands
            auto_react: Whether to auto-react to messages with emoji
            use_embeds: Whether to use Discord embeds for formatting
            verbose: Whether to include detailed information in messages
            escalation_timeout: Timeout in seconds for escalation responses
        """
        # Set up notification user IDs - include target user if not already in list
        if notification_user_ids is None:
            notification_user_ids = [target_user_id]
        elif target_user_id not in notification_user_ids:
            notification_user_ids.append(target_user_id)
        
        # Initialize components with defaults if not provided
        inbox = inbox or DiscordInbox(
            bot=bot,
            monitored_channels=monitored_channels,
            command_prefix=command_prefix,
            auto_react=auto_react
        )
        
        intervention_hub = intervention_hub or DiscordInterventionHub(
            bot=bot,
            target_user_id=target_user_id,
            timeout=escalation_timeout,
            escalation_channel_id=escalation_channel_id
        )
        
        outbox = outbox or DiscordOutbox(
            bot=bot,
            default_channel_id=default_channel_id,
            notification_user_ids=notification_user_ids,
            verbose=verbose,
            use_embeds=use_embeds
        )
        
        super().__init__(
            inbox=inbox,
            intervention_hub=intervention_hub,
            outbox=outbox
        )
        
        # Store bot reference for event registration
        self.bot = bot

        # Register a unified on_message event to ensure both Inbox and InterventionHub
        # receive every message without overwriting each other. discord.Client only
        # supports a single attribute-based `on_message` handler (no add_listener in
        # the fork we are using – discord.py 2.5.x). We therefore define one
        # coroutine that delegates to the two sub-components.

        bot = self.bot  # local alias for the closure

        # Remove any previously set `on_message` to avoid reference cycles
        if hasattr(bot, "on_message"):
            delattr(bot, "on_message")

        @bot.event  # type: ignore
        async def on_message(message):  # noqa: N802  (discord.py expects this exact name)
            """Fan-out incoming messages to inbox + HITL hub."""

            # First, let the inbox process the message (queue goals, commands, …)
            try:
                await inbox._process_message(message)  # pyright: ignore [private-member]
            except Exception:  # pragma: no cover – keep bot alive on handler error
                logging.getLogger(__name__).exception("Inbox on_message handler failed")

            # Then, pass it to the intervention hub to pick up potential replies
            try:
                await intervention_hub._handle_escalation_response(message)  # pyright: ignore [private-member]
            except Exception:
                logging.getLogger(__name__).exception("Intervention hub handler failed")
        
        # Store additional Discord-specific properties
        self.bot = bot
        self.target_user_id = target_user_id
        self.monitored_channels = monitored_channels
        self.default_channel_id = default_channel_id
        self.escalation_channel_id = escalation_channel_id
        self.notification_user_ids = notification_user_ids
    
    def display_welcome(self, channel_id: Optional[int] = None) -> None:
        """Display welcome message to Discord channel."""
        target_channel = channel_id or self.default_channel_id
        if hasattr(self.outbox, 'display_welcome'):
            self.outbox.display_welcome(target_channel)
    
    def display_goal_start(self, goal: str, channel_id: Optional[int] = None) -> None:
        """Display that goal processing has started."""
        target_channel = channel_id or self.default_channel_id
        if hasattr(self.outbox, 'display_goal_start'):
            self.outbox.display_goal_start(goal, target_channel)
    
    def display_reasoning_result(self, result, channel_id: Optional[int] = None) -> None:
        """Display a reasoning result with Discord formatting."""
        target_channel = channel_id or self.default_channel_id
        if hasattr(self.outbox, 'display_reasoning_result'):
            self.outbox.display_reasoning_result(result, target_channel)
    
    def display_goal_error(self, goal: str, error_msg: str, channel_id: Optional[int] = None) -> None:
        """Display a goal processing error."""
        target_channel = channel_id or self.default_channel_id
        if hasattr(self.outbox, 'display_goal_error'):
            self.outbox.display_goal_error(goal, error_msg, target_channel)
    
    def is_ready(self) -> bool:
        """Check if Discord bot is ready for operation."""
        return self.bot.is_ready()
    
    def add_monitored_channel(self, channel_id: int) -> None:
        """Add a channel to the monitored channels list."""
        if hasattr(self.inbox, 'monitored_channels'):
            if self.inbox.monitored_channels is None:
                self.inbox.monitored_channels = []
            if channel_id not in self.inbox.monitored_channels:
                self.inbox.monitored_channels.append(channel_id)
    
    def remove_monitored_channel(self, channel_id: int) -> None:
        """Remove a channel from the monitored channels list."""
        if hasattr(self.inbox, 'monitored_channels') and self.inbox.monitored_channels:
            if channel_id in self.inbox.monitored_channels:
                self.inbox.monitored_channels.remove(channel_id)
    
    def set_default_channel(self, channel_id: int) -> None:
        """Set the default channel for outbox messages."""
        self.default_channel_id = channel_id
        if hasattr(self.outbox, 'default_channel_id'):
            self.outbox.default_channel_id = channel_id
    
    def add_notification_user(self, user_id: int) -> None:
        """Add a user to the notification list."""
        if user_id not in self.notification_user_ids:
            self.notification_user_ids.append(user_id)
            if hasattr(self.outbox, 'notification_user_ids'):
                self.outbox.notification_user_ids = self.notification_user_ids
    
    def remove_notification_user(self, user_id: int) -> None:
        """Remove a user from the notification list."""
        if user_id in self.notification_user_ids:
            self.notification_user_ids.remove(user_id)
            if hasattr(self.outbox, 'notification_user_ids'):
                self.outbox.notification_user_ids = self.notification_user_ids
    
    def close(self) -> None:
        """Clean up all Discord communication channel resources."""
        super().close()
        # Additional Discord-specific cleanup could go here if needed