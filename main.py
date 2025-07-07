#!/usr/bin/env python3
"""
SETUP INSTRUCTIONS:

1. Create a `.env` file in this directory by copying `.env.template`.

2. Add your API keys to the `.env` file as needed:
   - JENTIC_API_KEY: Your API key for the Jentic platform (required)
   - OPENAI_API_KEY: If using OpenAI as LLM provider
   - GEMINI_API_KEY: If using Gemini as LLM provider
   - ANTHROPIC_API_KEY: If using Anthropic as LLM provider

3. Edit `config.json` to set your desired LLM provider and model, e.g.:
   {
     "llm": {
       "provider": "openai",    // or "gemini", or "anthropic"
       "model": "gpt-4o"        // or your preferred model 
     }
   }

4. Install dependencies:
   pip install -r requirements.txt

5. Run the demo:
   - CLI mode: python main.py or python main.py --mode cli
   - UI mode:  python main.py --mode ui
-----------------------------
"""

import argparse
import logging
import os, json
import sys

from dotenv import load_dotenv

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.agents.simple_ui_agent import SimpleUIAgent
from jentic_agents.communication import CLIController
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.communication.inbox.cli_inbox import CLIInbox
from jentic_agents.memory.agent_memory import create_agent_memory
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.reasoners.bullet_list_reasoner import BulletPlanReasoner
from jentic_agents.reasoners.freeform_reasoner import FreeformReasoner
from jentic_agents.reasoners.standard_reasoner import StandardReasoner

from jentic_agents.utils.llm import LiteLLMChatLLM
from jentic_agents.utils.config import get_config, get_config_value

logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

def main():

    parser = argparse.ArgumentParser(description="ActBots Live Demo")
    parser.add_argument(
        "--mode", 
        choices=["cli", "ui"], 
        default="cli",
        help="Interface mode: 'cli' for command line, 'ui' for graphical interface (default: cli)"
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    mode_name = "CLI" if args.mode == "cli" else "UI"
    print(f"Starting ActBots ({mode_name} Mode)")
    print("=" * 50)

    if args.mode == "cli":
        print("Type your goal below, or 'quit' to exit.")
    print("-" * 50)

    provider = get_config_value("llm", "provider", default="openai")
    model_name = get_config_value("llm", "model", default="gpt-4o")

    if not os.getenv("JENTIC_API_KEY"):
        print("ERROR: Missing JENTIC_API_KEY in your .env file.")
        sys.exit(1)

    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print("ERROR: LLM provider is Gemini but GEMINI_API_KEY is not set in .env.")
        sys.exit(1)

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: LLM provider is OpenAI but OPENAI_API_KEY is not set in .env.")
        sys.exit(1)

    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: LLM provider is Anthropic but ANTHROPIC_API_KEY is not set in .env.")
        sys.exit(1)

    try:
        # 1. Initialise the JenticClient
        jentic_client = JenticClient()

        # 2. Initialise lite LLM wrapper and memory
        llm_wrapper = LiteLLMChatLLM(model=model_name)
        memory = ScratchPadMemory()

        # 3. Initialize the Reasoner
        reasoner = BulletPlanReasoner(
            jentic=jentic_client,
            memory=memory,
            llm=llm_wrapper,
        )

        # 4. Initialize Agent based on mode
        if args.mode == "cli":
            # CLI mode: Use controller pattern (preferred)
            controller = CLIController()
            agent = InteractiveCLIAgent(
                reasoner=reasoner,
                memory=memory,
                controller=controller,
                jentic_client=jentic_client,
            )

        else:  # ui mode
            # For UI mode, we still use individual components for now
            # (SimpleUIAgent might need its own controller in the future)
            inbox = CLIInbox(prompt="Enter your goal: ")
            agent = SimpleUIAgent(
                reasoner=reasoner,
                memory=memory,
                inbox=inbox,
                jentic_client=jentic_client,
            )

        # 5. Run the Agent
        agent.spin()

    except ImportError as e:
        print(f"ERROR: A required package is not installed. {e}")
        print("Please make sure you have run 'pip install -r requirements.txt'.")
        sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error("An unexpected error occurred during the demo.", exc_info=True)
        sys.exit(1)

    print("-" * 50)
    print("👋 Demo finished. Goodbye!")


if __name__ == "__main__":
    main() 