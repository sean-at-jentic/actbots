"""
Discord-based inbox that reads goals from Discord messages.
"""
import asyncio
import logging
from typing import Optional, Dict, List, Callable, Any, Union
from queue import Queue, Empty
import discord
from discord.ext import commands

from .base_inbox import BaseInbox

logger = logging.getLogger(__name__)


class DiscordInbox(BaseInbox):
    """
    Inbox that reads goals from Discord messages.
    
    Monitors specified Discord channels for messages and treats them as goals.
    Supports command handling and goal acknowledgment through Discord reactions.
    """
    
    def __init__(
        self,
        bot: Union[discord.Client, commands.Bot],
        monitored_channels: Optional[List[int]] = None,
        command_prefix: str = "!",
        auto_react: bool = True
    ):
        """
        Initialize Discord inbox.
        
        Args:
            bot: Discord bot instance
            monitored_channels: List of channel IDs to monitor (None = all channels)
            command_prefix: Prefix for bot commands
            auto_react: Whether to auto-react to messages with emoji
        """
        self.bot = bot
        self.monitored_channels = monitored_channels or []
        self.command_prefix = command_prefix
        self.auto_react = auto_react
        self._closed = False
        self._goal_queue: Queue = Queue()
        self._history: List[str] = []
        self._message_goal_map: Dict[int, str] = {}  # message_id -> goal
        self._commands = self._setup_commands()
        
        # Set up Discord event handlers
        self._setup_discord_events()
    
    def _setup_commands(self) -> Dict[str, Callable[[discord.Message, str], bool]]:
        """Setup available Discord commands. Returns True if command was handled."""
        return {
            "help": self._handle_help_command,
            "history": self._handle_history_command,
            "status": self._handle_status_command,
        }
    
    def _setup_discord_events(self) -> None:
        """Set up Discord event handlers."""
        async def _inbox_on_message(message: discord.Message) -> None:
            """Listener that forwards messages to the inbox queue."""
            logger.info(f"[Inbox] Received message from {message.author}: {message.content}")

            # Ignore the bot's own messages
            if message.author == self.bot.user:
                logger.debug("[Inbox] Ignoring bot's own message")
                return

            # Respect monitored channel filter
            if self.monitored_channels and message.channel.id not in self.monitored_channels:
                logger.debug("[Inbox] Ignoring message from unmonitored channel %s", message.channel.id)
                return

            await self._process_message(message)

        # Register the listener without clobbering other on_message handlers
        try:
            # discord.py <2.0 may not expose add_listener on bare Client
            self.bot.add_listener(_inbox_on_message, name="on_message")  # type: ignore[attr-defined]
        except AttributeError:
            # As a fallback, simply assign. This may be overwritten by the controller
            setattr(self.bot, "on_message", _inbox_on_message)
    
    async def _process_message(self, message: discord.Message) -> None:
        """Process incoming Discord message."""
        # Ignore the bot's own messages
        if message.author == self.bot.user:
            logger.debug("Ignoring bot's own message")
            return

        # Respect monitored channel filter
        if self.monitored_channels and message.channel.id not in self.monitored_channels:
            logger.debug("Ignoring message from unmonitored channel %s", message.channel.id)
            return
        
        content = message.content.strip()
        logger.info(f"Processing message content: '{content}'")
        
        # Only process messages that mention the bot
        if not (self.bot.user.mentioned_in(message) or content.startswith(self.command_prefix)):
            logger.debug("Message doesn't mention bot and isn't a command, ignoring")
            return
        
        # Empty message
        if not content:
            logger.info("Empty message content, skipping")
            return
        
        # Check if it's a command
        if content.startswith(self.command_prefix):
            logger.info(f"Processing as command: {content}")
            await self._handle_command(message, content[len(self.command_prefix):])
            return
        
        # Remove bot mention from content to get the actual goal
        if self.bot.user.mentioned_in(message):
            # Remove the bot mention from the content
            bot_mention = f"<@{self.bot.user.id}>"
            bot_mention_nickname = f"<@!{self.bot.user.id}>"
            content = content.replace(bot_mention, "").replace(bot_mention_nickname, "").strip()
            
            if not content:
                logger.info("Message only contained bot mention with no goal, ignoring")
                return
        
        # It's a goal - add to queue
        logger.info(f"Adding goal to queue: '{content}'")
        self._goal_queue.put(content)
        self._history.append(content)
        self._message_goal_map[message.id] = content
        
        # Auto-react to acknowledge receipt
        if self.auto_react:
            try:
                logger.info(f"Adding reaction to message {message.id}")
                await message.add_reaction("👀")  # Eyes emoji to show we saw it
            except discord.DiscordException:
                logger.warning(f"Failed to react to message {message.id}")
    
    async def _handle_command(self, message: discord.Message, command_text: str) -> None:
        """Handle Discord command."""
        logger.info(f"Handling command: '{command_text}'")
        parts = command_text.split(None, 1)
        command = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        logger.info(f"Command: '{command}', Args: '{args}'")
        logger.info(f"Available commands: {list(self._commands.keys())}")
        
        if command in self._commands:
            try:
                logger.info(f"Executing command: {command}")
                await self._commands[command](message, args)
            except Exception as e:
                logger.error(f"Error handling command {command}: {e}")
                await message.channel.send(f"Error handling command: {e}")
        else:
            logger.info(f"Unknown command: {command}")
    
    async def _handle_help_command(self, message: discord.Message, args: str) -> None:
        """Handle help command."""
        help_text = f"""**Available Commands:**

**{self.command_prefix}help** - Show this help message
**{self.command_prefix}history** - Show recent goals
**{self.command_prefix}status** - Show bot status

**Usage:**
- Send any message to create a goal for the bot to process
- The bot will react with 👀 when it sees your message
- It will react with ✅ when the goal is completed
- It will react with ❌ if the goal fails

**Examples:**
`Find information about machine learning`
`Summarize the latest news`
`Create a plan for my project`
"""
        await message.channel.send(help_text)
    
    async def _handle_history_command(self, message: discord.Message, args: str) -> None:
        """Handle history command."""
        if not self._history:
            await message.channel.send("No goal history available")
            return
        
        # Show last 10 goals
        recent_history = self._history[-10:]
        history_text = "**Recent Goals:**\n"
        for i, goal in enumerate(recent_history, 1):
            history_text += f"{i}. {goal}\n"
        
        await message.channel.send(history_text)
    
    async def _handle_status_command(self, message: discord.Message, args: str) -> None:
        """Handle status command."""
        queue_size = self._goal_queue.qsize()
        total_goals = len(self._history)
        
        status_text = f"""**Bot Status:**
- Queue size: {queue_size}
- Total goals processed: {total_goals}
- Monitoring channels: {len(self.monitored_channels) if self.monitored_channels else 'all'}
- Status: {'Running' if not self._closed else 'Closed'}
"""
        await message.channel.send(status_text)
    
    def get_next_goal(self) -> Optional[str]:
        """
        Get the next goal from the Discord message queue.
        
        Returns:
            Next goal string, or None if no goals available
        """
        if self._closed:
            return None
        
        try:
            # Non-blocking get with short timeout
            goal = self._goal_queue.get(timeout=0.1)
            return goal
        except Empty:
            return None
    
    def acknowledge_goal(self, goal: str) -> None:
        """
        Acknowledge that a goal has been processed.
        
        Args:
            goal: The goal that was successfully processed
        """
        # Find the message that corresponds to this goal and react with checkmark
        asyncio.create_task(self._react_to_goal(goal, "✅"))
    
    def reject_goal(self, goal: str, reason: str) -> None:
        """
        Reject a goal that couldn't be processed.
        
        Args:
            goal: The goal that failed to process
            reason: Reason for rejection
        """
        # Find the message that corresponds to this goal and react with X
        asyncio.create_task(self._react_to_goal(goal, "❌"))
        # Also send the reason
        asyncio.create_task(self._send_rejection_message(goal, reason))
    
    async def _react_to_goal(self, goal: str, emoji: str) -> None:
        """React to the Discord message that corresponds to a goal."""
        message_id = None
        for msg_id, msg_goal in self._message_goal_map.items():
            if msg_goal == goal:
                message_id = msg_id
                break
        
        if message_id:
            try:
                # Find the message and react
                for channel in self.bot.get_all_channels():
                    if isinstance(channel, discord.TextChannel):
                        try:
                            message = await channel.fetch_message(message_id)
                            await message.add_reaction(emoji)
                            break
                        except discord.NotFound:
                            continue
                        except discord.DiscordException:
                            continue
            except Exception as e:
                logger.error(f"Failed to react to goal message: {e}")
    
    async def _send_rejection_message(self, goal: str, reason: str) -> None:
        """Send rejection message to the channel where the goal was posted."""
        message_id = None
        for msg_id, msg_goal in self._message_goal_map.items():
            if msg_goal == goal:
                message_id = msg_id
                break
        
        if message_id:
            try:
                # Find the message and send rejection in same channel
                for channel in self.bot.get_all_channels():
                    if isinstance(channel, discord.TextChannel):
                        try:
                            message = await channel.fetch_message(message_id)
                            await message.channel.send(f"❌ **Goal rejected:** {reason}")
                            break
                        except discord.NotFound:
                            continue
                        except discord.DiscordException:
                            continue
            except Exception as e:
                logger.error(f"Failed to send rejection message: {e}")
    
    def has_goals(self) -> bool:
        """
        Check if there are pending goals.
        
        Returns:
            True if goals are available, False otherwise
        """
        return not self._closed and not self._goal_queue.empty()
    
    def close(self) -> None:
        """
        Clean up inbox resources.
        """
        self._closed = True
        # Clear the queue
        while not self._goal_queue.empty():
            try:
                self._goal_queue.get_nowait()
            except Empty:
                break