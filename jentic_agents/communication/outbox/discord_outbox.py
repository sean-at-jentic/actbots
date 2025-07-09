"""
Discord implementation of BaseOutbox for sending results to Discord channels.
"""
import asyncio
import json
import logging
from typing import Any, Dict, Optional, Union, List
import discord
from discord.ext import commands

from .base_outbox import BaseOutbox

logger = logging.getLogger(__name__)


class DiscordOutbox(BaseOutbox):
    """
    Discord implementation that sends results to Discord channels.
    
    Sends progress updates, results, and error messages to specified Discord channels.
    Can also send DMs to specific users for important notifications.
    """
    
    def __init__(
        self,
        bot: Union[discord.Client, commands.Bot],
        default_channel_id: Optional[int] = None,
        notification_user_ids: Optional[List[int]] = None,
        verbose: bool = True,
        use_embeds: bool = True
    ):
        """
        Initialize Discord outbox.
        
        Args:
            bot: Discord bot instance
            default_channel_id: Default channel ID to send messages to
            notification_user_ids: List of user IDs to notify for important messages
            verbose: If True, includes detailed formatting and metadata
            use_embeds: If True, uses Discord embeds for better formatting
        """
        self.bot = bot
        self.default_channel_id = default_channel_id
        self.notification_user_ids = notification_user_ids or []
        self.verbose = verbose
        self.use_embeds = use_embeds
        self._channel_cache = {}  # channel_id -> channel object
    
    async def _get_channel(self, channel_id: Optional[int] = None) -> Optional[discord.TextChannel]:
        """Get Discord channel by ID with caching."""
        target_id = channel_id or self.default_channel_id
        if not target_id:
            return None
        
        if target_id in self._channel_cache:
            return self._channel_cache[target_id]
        
        try:
            channel = self.bot.get_channel(target_id)
            if not channel:
                channel = await self.bot.fetch_channel(target_id)
            
            if isinstance(channel, discord.TextChannel):
                self._channel_cache[target_id] = channel
                return channel
        except discord.DiscordException as e:
            logger.error(f"Failed to get channel {target_id}: {e}")
        
        return None
    
    async def _send_message(
        self,
        content: str = "",
        embed: Optional[discord.Embed] = None,
        channel_id: Optional[int] = None,
        mention_users: bool = False
    ) -> None:
        """Send message to Discord channel."""
        channel = await self._get_channel(channel_id)
        if not channel:
            logger.error(f"Could not send message - channel not found: {channel_id}")
            return
        
        try:
            # Add user mentions if requested
            if mention_users and self.notification_user_ids:
                mentions = " ".join(f"<@{user_id}>" for user_id in self.notification_user_ids)
                content = f"{mentions} {content}".strip()
            
            if self.use_embeds and embed:
                await channel.send(content, embed=embed)
            else:
                await channel.send(content)
        except discord.DiscordException as e:
            logger.error(f"Failed to send message to channel {channel_id}: {e}")
    
    def _create_embed(self, title: str, description: str, color: int) -> discord.Embed:
        """Create a Discord embed with consistent styling."""
        embed = discord.Embed(
            title=title,
            description=description,
            color=color
        )
        embed.set_footer(text="ActBots Framework")
        return embed
    
    def send_progress(self, goal_id: str, message: str, step: Optional[str] = None) -> None:
        """Send progress update to Discord channel."""
        async def _send():
            if self.use_embeds:
                title = f"🔄 Progress Update"
                description = f"**Goal:** {goal_id}\n**Status:** {message}"
                if step:
                    description += f"\n**Step:** {step}"
                
                embed = self._create_embed(title, description, 0x3498db)  # Blue
                await self._send_message(embed=embed)
            else:
                content = f"🔄 **Progress Update**\nGoal: {goal_id}\nStatus: {message}"
                if step:
                    content += f"\nStep: {step}"
                await self._send_message(content)
        
        asyncio.create_task(_send())
    
    def send_result(self, goal_id: str, result: Any, success: bool = True) -> None:
        """Send final result to Discord channel."""
        status_emoji = "✅" if success else "❌"
        status_text = "SUCCESS" if success else "FAILED"
        
        if self.use_embeds:
            title = f"{status_emoji} Goal {status_text}"
            description = f"**Goal:** {goal_id}\n**Result:**\n"
            
            # Format result based on type
            if isinstance(result, (dict, list)):
                result_text = json.dumps(result, indent=2)
                if len(result_text) > 1000:
                    result_text = result_text[:1000] + "..."
                description += f"```json\n{result_text}\n```"
            else:
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "..."
                description += result_str
            
            color = 0x27ae60 if success else 0xe74c3c  # Green or Red
            embed = self._create_embed(title, description, color)
            asyncio.create_task(self._send_message(embed=embed, mention_users=True))
        else:
            content = f"{status_emoji} **Goal {status_text}**\nGoal: {goal_id}\nResult: {result}"
            asyncio.create_task(self._send_message(content))
    
    def send_error(self, goal_id: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Send error notification to Discord channel."""
        if self.use_embeds:
            title = "❌ Error"
            description = f"**Goal:** {goal_id}\n**Error:** {error_message}"
            
            if context and self.verbose:
                context_text = json.dumps(context, indent=2)
                if len(context_text) > 500:
                    context_text = context_text[:500] + "..."
                description += f"\n**Context:**\n```json\n{context_text}\n```"
            
            embed = self._create_embed(title, description, 0xe74c3c)  # Red
            asyncio.create_task(self._send_message(embed=embed, mention_users=True))
        else:
            content = f"❌ **Error**\nGoal: {goal_id}\nError: {error_message}"
            if context and self.verbose:
                content += f"\nContext: {context}"
            asyncio.create_task(self._send_message(content))
    
    def send_step_complete(self, goal_id: str, step: str, result: Any) -> None:
        """Send step completion notification to Discord channel."""
        if self.use_embeds:
            title = "✓ Step Complete"
            description = f"**Goal:** {goal_id}\n**Step:** {step}"
            
            if result and self.verbose:
                result_str = str(result)
                if len(result_str) > 300:
                    result_str = result_str[:300] + "..."
                description += f"\n**Result:** {result_str}"
            
            embed = self._create_embed(title, description, 0xf39c12)  # Orange
            asyncio.create_task(self._send_message(embed=embed))
        else:
            content = f"✓ **Step Complete**\nGoal: {goal_id}\nStep: {step}"
            if result and self.verbose:
                content += f"\nResult: {result}"
            asyncio.create_task(self._send_message(content))
    
    def send_status_change(self, goal_id: str, old_status: str, new_status: str) -> None:
        """Send status change notification to Discord channel."""
        if self.use_embeds:
            title = "🔄 Status Change"
            description = f"**Goal:** {goal_id}\n**Status:** {old_status} → {new_status}"
            
            embed = self._create_embed(title, description, 0x9b59b6)  # Purple
            asyncio.create_task(self._send_message(embed=embed))
        else:
            content = f"🔄 **Status Change**\nGoal: {goal_id}\nStatus: {old_status} → {new_status}"
            asyncio.create_task(self._send_message(content))
    
    def display_welcome(self, channel_id: Optional[int] = None) -> None:
        """Display welcome message to Discord channel."""
        if self.use_embeds:
            title = "🤖 ActBots Framework"
            description = """Welcome to the ActBots Discord interface!

**How to use:**
• Send any message to create a goal for the bot
• I'll react with 👀 when I see your message
• I'll react with ✅ when the goal is completed
• I'll react with ❌ if the goal fails

**Commands:**
• `!help` - Show available commands
• `!history` - Show recent goals
• `!status` - Show bot status

Ready to help! 🚀"""
            
            embed = self._create_embed(title, description, 0x3498db)
            asyncio.create_task(self._send_message(embed=embed, channel_id=channel_id))
        else:
            content = """🤖 **ActBots Framework**

Welcome to the ActBots Discord interface!
Send any message to create a goal for the bot.
Type `!help` for available commands.

Ready to help! 🚀"""
            asyncio.create_task(self._send_message(content, channel_id=channel_id))
    
    def display_goal_start(self, goal: str, channel_id: Optional[int] = None) -> None:
        """Display that goal processing has started."""
        if self.use_embeds:
            title = "🎯 Goal Started"
            description = f"**Processing:** {goal}\n*Working on it...*"
            
            embed = self._create_embed(title, description, 0xf39c12)
            asyncio.create_task(self._send_message(embed=embed, channel_id=channel_id))
        else:
            content = f"🎯 **Goal Started**\nProcessing: {goal}\n*Working on it...*"
            asyncio.create_task(self._send_message(content, channel_id=channel_id))
    
    def display_reasoning_result(self, result: Any, channel_id: Optional[int] = None) -> None:
        """Display a reasoning result with Discord formatting."""
        success = getattr(result, 'success', False)
        final_answer = getattr(result, 'final_answer', result)
        
        if success:
            if self.use_embeds:
                title = "✅ Goal Completed Successfully"
                description = str(final_answer)
                
                # Add tool usage info if available
                tool_calls = getattr(result, 'tool_calls', [])
                iterations = getattr(result, 'iterations', 0)
                if tool_calls:
                    description += f"\n\n**Tools Used:** {len(tool_calls)} tool(s) in {iterations} iteration(s)"
                    tools_list = []
                    for call in tool_calls:
                        tool_name = call.get('tool_name', call.get('tool_id', 'Unknown'))
                        tools_list.append(tool_name)
                    description += f"\n**Tools:** {', '.join(set(tools_list))}"
                
                embed = self._create_embed(title, description, 0x27ae60)
                asyncio.create_task(self._send_message(embed=embed, channel_id=channel_id, mention_users=True))
            else:
                content = f"✅ **Goal Completed Successfully**\n{final_answer}"
                asyncio.create_task(self._send_message(content, channel_id=channel_id))
        else:
            if self.use_embeds:
                title = "❌ Goal Failed"
                description = str(final_answer)
                
                error_message = getattr(result, 'error_message', None)
                if error_message:
                    description += f"\n\n**Error:** {error_message}"
                
                embed = self._create_embed(title, description, 0xe74c3c)
                asyncio.create_task(self._send_message(embed=embed, channel_id=channel_id, mention_users=True))
            else:
                content = f"❌ **Goal Failed**\n{final_answer}"
                asyncio.create_task(self._send_message(content, channel_id=channel_id))
    
    def display_goal_error(self, goal: str, error_msg: str, channel_id: Optional[int] = None) -> None:
        """Display a goal processing error."""
        if self.use_embeds:
            title = "❌ Goal Error"
            description = f"**Goal:** {goal}\n**Error:** {error_msg}"
            
            embed = self._create_embed(title, description, 0xe74c3c)
            asyncio.create_task(self._send_message(embed=embed, channel_id=channel_id, mention_users=True))
        else:
            content = f"❌ **Goal Error**\nGoal: {goal}\nError: {error_msg}"
            asyncio.create_task(self._send_message(content, channel_id=channel_id))
    
    def close(self) -> None:
        """Clean up Discord outbox resources."""
        if self.use_embeds:
            title = "📤 Outbox Closed"
            description = "Discord outbox has been closed."
            embed = self._create_embed(title, description, 0x95a5a6)
            asyncio.create_task(self._send_message(embed=embed))
        else:
            content = "📤 **Outbox Closed**\nDiscord outbox has been closed."
            asyncio.create_task(self._send_message(content))
        
        # Clear cache
        self._channel_cache.clear()