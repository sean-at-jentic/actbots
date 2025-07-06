"""
Example demonstrating the simple escalation system.

This shows how the agent can choose to escalate to a human when it decides it needs help,
rather than having automatic triggers.

Run with: python -m jentic_agents.examples.escalation_example
"""

import os
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jentic_agents.communication.hitl.cli_intervention_hub import CLIInterventionHub
from jentic_agents.communication.hitl.base_intervention_hub import NoEscalation
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.reasoners.bullet_list_reasoner import BulletPlanReasoner
from jentic_agents.utils.llm import LiteLLMChatLLM


def main():
    """Demonstrate the escalation system."""
    print("🤖 Escalation System Demo")
    print("=" * 50)
    print("This demo shows how the agent can choose to escalate to humans.")
    print("The agent has full autonomy and can ask for help when it decides to.")
    print("-" * 50)

    # Check for required environment variables
    if not os.getenv("JENTIC_API_KEY"):
        print("❌ ERROR: Missing JENTIC_API_KEY in your .env file.")
        print("Set JENTIC_API_KEY=dummy for this demo")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: Missing OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    try:
        # Initialize components
        jentic_client = JenticClient()
        memory = ScratchPadMemory()
        llm = LiteLLMChatLLM(model="gpt-4o-mini")  # Use cheaper model for demo
        
        # Create escalation system
        escalation = CLIInterventionHub()
        
        # Create reasoner with escalation
        reasoner = BulletPlanReasoner(
            jentic=jentic_client,
            memory=memory,
            llm=llm,
            escalation=escalation,
            max_iters=5  # Limit iterations for demo
        )
        
        print("\n🎯 Dynamic Escalation Demo:")
        print("This demo shows how the agent can request help at ANY point during execution:")
        print("1. Planning - 'help me with something unclear'")
        print("2. Tool selection - 'find a very specific rare tool'") 
        print("3. Parameter generation - 'send Discord message'")
        print("4. Error handling - goals with authentication issues")
        print("5. Proactive guidance - complex multi-step goals")
        print("6. Reasoning - tasks requiring domain knowledge")
        print()
        
        # Enhanced demo scenarios showcasing different escalation points
        demo_goals = [
            "help me with something unclear",
            "send a Discord message to the development team",
            "find weather information for a specific location"
        ]
        
        for i, goal in enumerate(demo_goals, 1):
            print(f"\n🧪 Demo {i}/3: '{goal}'")
            print("-" * 30)
            
            try:
                result = reasoner.run(goal, max_iterations=3)
                print(f"✅ Result: {result}")
            except Exception as e:
                print(f"❌ Failed: {e}")
            
            if i < len(demo_goals):
                input("\nPress Enter to continue to next demo...")
        
        print("\n🎮 Interactive Mode:")
        print("Now you can try your own goals. Type 'quit' to exit.")
        
        while True:
            goal = input("\nEnter your goal: ").strip()
            if goal.lower() in ['quit', 'exit', 'q']:
                break
            
            if not goal:
                continue
                
            try:
                result = reasoner.run(goal, max_iterations=5)
                print(f"✅ Result: {result}")
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Failed: {e}")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n👋 Demo finished!")


if __name__ == "__main__":
    main() 