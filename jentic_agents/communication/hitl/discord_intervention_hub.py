"""
Discord implementation of BaseInterventionHub for escalation via Discord DM.
"""
import asyncio
import logging
from typing import Optional, Union
import discord
from discord.ext import commands

from .base_intervention_hub import BaseInterventionHub

logger = logging.getLogger(__name__)


class DiscordInterventionHub(BaseInterventionHub):
    """
    Discord-based escalation that sends DMs to a predefined user when help is needed.
    
    When the agent needs help, it will:
    1. Send a DM to the predefined user with the question and context
    2. Wait for a response from that user
    3. Return the response to the agent
    """
    
    def __init__(
        self,
        bot: Union[discord.Client, commands.Bot],
        target_user_id: int,
        timeout: int = 300,  # 5 minutes default
        escalation_channel_id: Optional[int] = None
    ):
        """
        Initialize Discord intervention hub.
        
        Args:
            bot: Discord bot instance
            target_user_id: Discord user ID to send escalation DMs to
            timeout: Timeout in seconds to wait for human response
            escalation_channel_id: Optional channel ID to send escalations to instead of DM
        """
        super().__init__()
        self.bot = bot
        self.target_user_id = target_user_id
        self.timeout = timeout
        self.escalation_channel_id = escalation_channel_id
        self._pending_escalations = {}  # escalation_id -> Future
        self._escalation_counter = 0
        
        # Set up Discord event handlers
        self._setup_discord_events()
    
    def _setup_discord_events(self) -> None:
        """Set up Discord event handlers for escalation responses."""
        async def _hitl_on_message(message: discord.Message) -> None:
            """Listener that captures potential escalation replies."""
            # Only care about messages from the configured human helper
            if message.author.id != self.target_user_id:
                return

            await self._handle_escalation_response(message)

        # Register listener without overriding others
        try:
            self.bot.add_listener(_hitl_on_message, name="on_message")  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback – allow controller to overwrite later
            setattr(self.bot, "on_message", _hitl_on_message)
    
    async def _handle_escalation_response(self, message: discord.Message) -> None:
        """Handle potential escalation response from the target user."""
        # Ignore messages that are not from the configured human helper. This prevents
        # the bot's own escalation messages (or messages from other users) from being
        # misinterpreted as human responses and prematurely resolving pending
        # escalations.
        if message.author.id != self.target_user_id:
            return

        # Check if this is a DM or in the escalation channel
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_escalation_channel = (
            self.escalation_channel_id and 
            hasattr(message.channel, 'id') and 
            message.channel.id == self.escalation_channel_id
        )
        
        if not (is_dm or is_escalation_channel):
            return
        
        # Look for pending escalations and resolve them
        content = message.content.strip()
        if content and self._pending_escalations:
            # For simplicity, resolve the oldest pending escalation
            # In a more complex system, you might use message IDs or other identifiers
            escalation_id = min(self._pending_escalations.keys())
            future = self._pending_escalations.pop(escalation_id)
            
            if not future.done():
                future.set_result(content)
                
                # React to acknowledge we received the response
                try:
                    await message.add_reaction("✅")
                except discord.DiscordException:
                    pass
    
    def ask_human(self, question: str, context: Optional[str] = None) -> str:
        """
        Ask a human for help via Discord DM.
        
        Args:
            question: The question to ask the human
            context: Optional context to help the human understand the situation
            
        Returns:
            Human's response as a string
        """
        try:
            # Check if bot is ready first
            if not self.bot.is_ready():
                logger.error("Discord bot is not ready")
                return "Error: Discord bot is not ready"
            
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # We have a running event loop, use run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(
                    self._ask_human_async(question, context), 
                    loop
                )
                return future.result()  # Wait indefinitely
                
            except RuntimeError:
                # No running event loop in this thread, but Discord bot should have one
                # Try to find the bot's event loop through the _loop attribute
                if hasattr(self.bot, '_connection') and hasattr(self.bot._connection, 'loop'):
                    bot_loop = self.bot._connection.loop
                    if bot_loop and bot_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            self._ask_human_async(question, context), 
                            bot_loop
                        )
                        return future.result()  # Wait indefinitely
                
                # Fallback: no way to access Discord's async context
                logger.error("No running event loop found and cannot access bot's event loop")
                return "Error: Cannot access Discord's async context for escalation"
                
        except asyncio.TimeoutError:
            # This should not happen anymore since we removed timeouts
            logger.error("Unexpected timeout in Discord escalation")
            return "Error: Unexpected timeout in Discord escalation"
        except Exception as e:
            logger.error(f"Unexpected error during Discord escalation: {e}")
            return f"Error: Unexpected error during escalation: {e}"
    
    async def _ask_human_async(self, question: str, context: Optional[str] = None) -> str:
        """
        Async implementation of ask_human.
        
        Args:
            question: The question to ask the human
            context: Optional context to help the human understand the situation
            
        Returns:
            Human's response as a string
        """
        try:
            # Get the target user
            user = self.bot.get_user(self.target_user_id)
            if not user:
                user = await self.bot.fetch_user(self.target_user_id)
            
            if not user:
                logger.error(f"Could not find Discord user with ID {self.target_user_id}")
                return "Error: Could not reach human for help"
            
            # Create escalation message
            escalation_msg = "🤖➡️👤 **AGENT REQUESTING HELP**\n"
            escalation_msg += "=" * 50 + "\n\n"
            
            if context:
                # Truncate context if too long
                max_context_length = 1000
                if len(context) > max_context_length:
                    context = context[:max_context_length] + "... [truncated]"
                escalation_msg += f"**Context:** {context}\n\n"
            
            # Truncate question if too long
            max_question_length = 800
            if len(question) > max_question_length:
                question = question[:max_question_length] + "... [truncated]"
            
            escalation_msg += f"**Question:** {question}\n\n"
            escalation_msg += "Please respond with your answer. I'll wait for your response."
            
            # Final safety check - Discord has 2000 char limit
            if len(escalation_msg) > 1950:  # Leave some margin
                escalation_msg = escalation_msg[:1950] + "... [truncated]"
            
            # Create future for response
            self._escalation_counter += 1
            escalation_id = self._escalation_counter
            response_future = asyncio.Future()
            self._pending_escalations[escalation_id] = response_future
            
            # Send the escalation
            try:
                if self.escalation_channel_id:
                    # Send to specific channel
                    channel = self.bot.get_channel(self.escalation_channel_id)
                    if channel:
                        await channel.send(f"<@{self.target_user_id}> {escalation_msg}")
                    else:
                        logger.error(f"Could not find escalation channel {self.escalation_channel_id}")
                        return "Error: Could not reach human for help"
                else:
                    # Send DM
                    await user.send(escalation_msg)
                
                # Wait for response indefinitely (no timeout)
                try:
                    response = await response_future
                    return response
                except asyncio.CancelledError:
                    # Clean up pending escalation if cancelled
                    self._pending_escalations.pop(escalation_id, None)
                    return "Escalation was cancelled"
                
            except discord.DiscordException as e:
                logger.error(f"Failed to send escalation message: {e}")
                return f"Error: Failed to send escalation message: {e}"
            
        except Exception as e:
            logger.error(f"Unexpected error in escalation: {e}")
            return f"Error: Unexpected error during escalation: {e}"
    
    def is_available(self) -> bool:
        """
        Check if human escalation is available.
        
        Returns:
            True if the bot is ready and target user is configured
        """
        return (
            self.bot.is_ready() and 
            self.target_user_id is not None and
            self.target_user_id > 0
        )


# Backward compatibility alias
DiscordEscalation = DiscordInterventionHub