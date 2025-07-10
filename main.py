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

4. Run command `uv venv && source .venv/bin/activate && uv pip install -e .`

5. Run the demo:
   - CLI mode: python main.py or python main.py --mode cli
   - UI mode:  python main.py --mode ui
   - Discord mode: python main.py --mode discord
-----------------------------
"""

import argparse
import os
import sys
from dotenv import load_dotenv

from jentic_agents.utils.logger import get_logger
from jentic_agents.utils.config import validate_api_keys, get_config_value
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.utils.llm import LiteLLMChatLLM
from jentic_agents.reasoners.hybrid_reasoner import HybridReasoner
from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.agents.simple_ui_agent import SimpleUIAgent
from jentic_agents.communication.controllers.cli_controller import CLIController
from jentic_agents.communication.inbox.cli_inbox import CLIInbox
from jentic_agents.communication.controllers.discord_controller import DiscordController

logger = get_logger(__name__)

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
    validate_api_keys()

    # Initialize core components (common to all modes)
    jentic_client = JenticClient()
    model_name = get_config_value("llm", "model", default="gpt-4o")
    llm_wrapper = LiteLLMChatLLM(model=model_name)
    memory = ScratchPadMemory()

    try:
        if args.mode == "ui":
            reasoner = HybridReasoner(
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
            agent.spin()
        elif args.mode == "cli":
            controller = CLIController()
            reasoner = HybridReasoner(
                jentic=jentic_client,
                memory=memory,
                llm=llm_wrapper,
                intervention_hub=controller.intervention_hub,
            )
            agent = InteractiveCLIAgent(
                reasoner=reasoner,
                memory=memory,
                controller=controller,
                jentic_client=jentic_client,
            )
            agent.spin()
        elif args.mode == "discord":
            controller, bot, discord_token = DiscordController.create_controller("discord")
            reasoner = HybridReasoner(
                jentic=jentic_client,
                memory=memory,
                llm=llm_wrapper,
                intervention_hub=controller.intervention_hub,
            )
            agent = InteractiveCLIAgent(
                reasoner=reasoner,
                memory=memory,
                controller=controller,
                jentic_client=jentic_client,
            )
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
                if default_channel_id:
                    controller.display_welcome(default_channel_id)
                import asyncio
                asyncio.create_task(agent.spin_async())
            bot.run(discord_token)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except (ImportError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()