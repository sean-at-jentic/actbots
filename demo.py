#!/usr/bin/env python3
"""
Demo script to test the InteractiveCLIAgent with mock OpenAI.

This simulates the full system without requiring actual API keys.
"""
import os
import sys
from io import StringIO
from unittest.mock import Mock

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.reasoners.standard_reasoner import StandardReasoner
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.inbox.cli_inbox import CLIInbox
from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent


def create_mock_openai_client():
    """Create a mock OpenAI client that returns helpful responses."""
    mock_client = Mock()
    call_count = 0
    
    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        
        # Simple response generator based on the messages
        messages = kwargs.get('messages', [])
        user_content = ""
        
        for msg in messages:
            if msg.get('role') == 'user':
                user_content = msg.get('content', '').lower()
                break
        
        # Generate appropriate responses based on content and call sequence
        if 'what should be the next step' in user_content:
            response_text = "Use echo tool to calculate 2+2"
        elif 'which tool id should be used' in user_content:
            response_text = "echo_tool_001"
        elif 'generate parameters' in user_content:
            response_text = '{"message": "2+2"}'
        elif 'has the goal been achieved' in user_content:
            # On first evaluation call, say no to allow tool execution
            # On second call, say yes to complete
            if call_count <= 4:
                response_text = "NO"
            else:
                response_text = "YES"
        elif 'provide a final answer' in user_content:
            response_text = "The answer to 2+2 is 4, as confirmed by the echo tool result: Echo: 2+2."
        else:
            response_text = "I understand your request and will proceed accordingly."
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = response_text
        return mock_response
    
    mock_client.chat.completions.create = mock_create
    return mock_client


def main():
    """Run the demo."""
    # Enable debug logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting ActBots Demo")
    print("=" * 50)
    
    # Create components
    jentic_client = JenticClient()
    mock_openai = create_mock_openai_client()
    
    reasoner = StandardReasoner(
        jentic_client=jentic_client,
        openai_client=mock_openai,
        model="gpt-4-demo"  # Mock model
    )
    
    memory = ScratchPadMemory()
    
    # Create test input
    test_input = StringIO("What's 2+2?\nquit\n")
    inbox = CLIInbox(input_stream=test_input, prompt="Demo > ")
    
    # Create and run agent
    agent = InteractiveCLIAgent(
        reasoner=reasoner,
        memory=memory,
        inbox=inbox,
        jentic_client=jentic_client
    )
    
    print("Demo agent created successfully!")
    print("Processing demo goal: 'What's 2+2?'")
    print("-" * 30)
    
    try:
        agent.spin()
    except Exception as e:
        print(f"Error during demo: {e}")
    
    print("-" * 30)
    print("Demo completed!")
    
    # Show what's in memory
    print(f"\nMemory contents: {memory.keys()}")
    if "current_goal" in memory:
        print(f"Last goal: {memory.retrieve('current_goal')}")


if __name__ == "__main__":
    main()
