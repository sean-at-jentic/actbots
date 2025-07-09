#!/usr/bin/env python3
"""
SETUP INSTRUCTIONS:

1. Create a `.env` file in this directory by copying `.env.template`.

2. Add your API keys to the `.env` file as needed:
   - JENTIC_API_KEY: Your API key for the Jentic platform (required)
   - OPENAI_API_KEY: If using OpenAI as LLM provider
   - GEMINI_API_KEY: If using Gemini as LLM provider
   - ANTHROPIC_API_KEY: If using Anthropic as LLM provider
   - DISCORD_BOT_TOKEN: Your Discord bot token (for Discord mode)

3. Edit `pyproject.toml` to set your desired LLM provider and model, e.g.:

   [tool.actbots.llm]
   provider = "openai"  # or "gemini", or "anthropic"
   model = "gpt-4o"     # or your preferred model

   [tool.actbots.discord]
   enabled = true
   token = ""
   target_user_id = 123456789         # Your Discord user ID for escalations
   monitored_channels = [987654321]   # Channel IDs to monitor
   default_channel_id = 987654321     # Default channel for responses

4. Install dependencies:
   pip install -r requirements.txt
   For Discord mode, also install: pip install discord.py

5. Run the demo:
   - CLI mode: python main.py or python main.py --mode cli
   - UI mode:  python main.py --mode ui
   - Discord mode: python main.py --mode discord
-----------------------------
"""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv

from jentic_agents.utils.logger import get_logger
from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.agents.simple_ui_agent import SimpleUIAgent
from jentic_agents.communication import CLIController
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.communication.inbox.cli_inbox import CLIInbox
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.reasoners.bullet_list_reasoner import BulletPlanReasoner
from jentic_agents.utils.llm import LiteLLMChatLLM
from jentic_agents.utils.config import get_config_value

# Discord imports (optional)
try:
    import discord
    from discord.ext import commands
    from jentic_agents.communication.discord_controller import DiscordController
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

get_logger(__name__)


def validate_api_keys():
    """Validate required API keys are present."""
    if not os.getenv("JENTIC_API_KEY"):
        print("ERROR: Missing JENTIC_API_KEY in your .env file.")
        sys.exit(1)

    provider = get_config_value("llm", "provider", default="openai")
    api_key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY", 
        "anthropic": "ANTHROPIC_API_KEY"
    }
    
    if provider in api_key_map and not os.getenv(api_key_map[provider]):
        print(f"ERROR: LLM provider is {provider} but {api_key_map[provider]} is not set in .env.")
        sys.exit(1)


def create_controller(mode):
    """Create the appropriate controller based on mode."""
    if mode == "cli":
        return CLIController(), None, None
    
    elif mode == "discord":
        if not DISCORD_AVAILABLE:
            raise ImportError("Discord mode requires 'discord.py'. Install it with: pip install discord.py")

        # Get Discord configuration
        discord_token = os.getenv("DISCORD_BOT_TOKEN") or get_config_value("discord", "token")
        discord_user_id = get_config_value("discord", "target_user_id")
        
        if not discord_token:
            raise ValueError("Discord mode requires DISCORD_BOT_TOKEN in .env or discord.token in config.json")
        if not discord_user_id:
            raise ValueError("Discord mode requires discord.target_user_id in config.json")

        monitored_channels = get_config_value("discord", "monitored_channels", default=[])
        default_channel_id = get_config_value("discord", "default_channel_id", default=None)
        escalation_channel_id = get_config_value("discord", "escalation_channel_id", default=None)
        command_prefix = get_config_value("discord", "command_prefix", default="!")
        use_embeds = get_config_value("discord", "use_embeds", default=True)
        auto_react = get_config_value("discord", "auto_react", default=True)
        verbose = get_config_value("discord", "verbose", default=True)
        escalation_timeout = get_config_value("discord", "escalation_timeout", default=300)

        # Create Discord bot
        intents = discord.Intents.default()
        intents.message_content = True
        bot = discord.Client(intents=intents)
        
        # Create Discord controller
        controller = DiscordController(
            bot=bot,
            target_user_id=discord_user_id,
            monitored_channels=monitored_channels if monitored_channels else None,
            default_channel_id=default_channel_id if default_channel_id else None,
            escalation_channel_id=escalation_channel_id if escalation_channel_id else None,
            command_prefix=command_prefix,
            auto_react=auto_react,
            use_embeds=use_embeds,
            verbose=verbose,
            escalation_timeout=escalation_timeout
        )
        
        return controller, bot, discord_token
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def setup_agent(mode):
    """Initialize and return the agent based on the specified mode."""
    # Initialize core components (common to all modes)
    jentic_client = JenticClient()
    model_name = get_config_value("llm", "model", default="gpt-4o")
    llm_wrapper = LiteLLMChatLLM(model=model_name)
    memory = ScratchPadMemory()
    
    if mode == "ui":
        # UI mode uses SimpleUIAgent with CLI inbox
        reasoner = BulletPlanReasoner(
            jentic=jentic_client,
            memory=memory,
            llm=llm_wrapper,
        )
        
        inbox = CLIInbox(prompt="Enter your goal: ")
        agent = SimpleUIAgent(
            reasoner=reasoner,
            memory=memory,
            inbox=inbox,
            jentic_client=jentic_client,
        )
        
        return agent, None, None, None
    
    # For CLI and Discord modes - create controller first
    controller, bot, discord_token = create_controller(mode)
    
    # Create reasoner with intervention hub for CLI/Discord modes
    reasoner = BulletPlanReasoner(
        jentic=jentic_client,
        memory=memory,
        llm=llm_wrapper,
        intervention_hub=controller.intervention_hub,
    )
    
    # Use InteractiveCLIAgent for both CLI and Discord modes
    agent = InteractiveCLIAgent(
        reasoner=reasoner,
        memory=memory,
        controller=controller,
        jentic_client=jentic_client,
    )
    
    return agent, bot, discord_token, controller


def run_agent(mode, agent, bot, discord_token, controller):
    """Run the agent based on the specified mode."""
    if mode == "discord":
        discord_user_id = get_config_value("discord", "target_user_id")
        monitored_channels = get_config_value("discord", "monitored_channels", default=[])
        default_channel_id = get_config_value("discord", "default_channel_id", default=None)
        
        @bot.event
        async def on_ready():
            print(f"Discord bot logged in as {bot.user}")
            print(f"Monitoring user: {discord_user_id}")
            if monitored_channels:
                print(f"Monitoring channels: {monitored_channels}")
            else:
                print("Monitoring all channels")
            
            # Display welcome message if default channel is set
            if default_channel_id:
                controller.display_welcome(default_channel_id)
            
            # Start the agent in the background
            import asyncio
            asyncio.create_task(agent.spin_async())
        
        bot.run(discord_token)
    else:
        # CLI and UI modes use synchronous spin
        agent.spin()


def main():
    parser = argparse.ArgumentParser(description="ActBots Live Demo")
    parser.add_argument(
        "--mode", 
        choices=["cli", "ui", "discord"], 
        default="cli",
        help="Interface mode: 'cli' for command line, 'ui' for graphical interface, 'discord' for Discord bot (default: cli)"
    )
    args = parser.parse_args()

    load_dotenv()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    validate_api_keys()
    
    mode_name = {"cli": "CLI", "ui": "UI", "discord": "Discord"}[args.mode]
    print(f"Starting ActBots ({mode_name} Mode)")
    print("=" * 50)

    if args.mode == "cli":
        print("Type your goal below, or 'quit' to exit.")
    elif args.mode == "discord":
        print("Starting Discord bot...")
    print("-" * 50)

    try:
        # Setup agent based on mode
        agent, bot, discord_token, controller = setup_agent(args.mode)

        # Run the agent
        run_agent(args.mode, agent, bot, discord_token, controller)

    except (ImportError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    print("-" * 50)
    print("👋 Demo finished. Goodbye!")


if __name__ == "__main__":
    main()