#!/usr/bin/env python3
"""
Test script to verify that both reasoners properly extract and use api_name from Jentic search results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jentic_agents'))

from unittest.mock import Mock, patch
from jentic_agents.reasoners.bullet_list_reasoner import BulletPlanReasoner
from jentic_agents.reasoners.standard_reasoner import StandardReasoner
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.utils.llm import LiteLLMChatLLM

def test_bullet_reasoner_api_name_extraction():
    """Test that BulletPlanReasoner extracts and uses api_name in tool selection."""
    print("Testing BulletPlanReasoner api_name extraction...")
    
    # Mock search results with api_name
    mock_search_results = [
        {
            'id': 'tool1',
            'name': 'GitHub API Tool',
            'description': 'Tool for GitHub operations',
            'api_name': 'github_api'
        },
        {
            'id': 'tool2', 
            'name': 'Slack API Tool',
            'description': 'Tool for Slack operations',
            'api_name': 'slack_api'
        }
    ]
    
    # Create mocks
    mock_jentic = Mock(spec=JenticClient)
    mock_jentic.search.return_value = mock_search_results
    
    mock_memory = Mock(spec=ScratchPadMemory)
    
    mock_llm = Mock(spec=LiteLLMChatLLM)
    mock_llm.chat.return_value = "2"  # Select second tool
    
    # Create reasoner
    reasoner = BulletPlanReasoner(
        jentic=mock_jentic,
        memory=mock_memory,
        llm=mock_llm
    )
    
    # Create a test step
    from jentic_agents.reasoners.bullet_list_reasoner import Step, ReasonerState
    step = Step(text="Use Slack to send a message", goal_context="Slack integration")
    state = ReasonerState(goal="Send notification via Slack")
    
    # Call select_tool
    try:
        tool_id = reasoner.select_tool(step, state)
        
        # Verify the LLM was called with api_name information
        llm_call_args = mock_llm.chat.call_args
        prompt_content = llm_call_args[1]['messages'][0]['content']
        
        print(f"✓ Tool selected: {tool_id}")
        print(f"✓ LLM prompt contains api_name: {'API:' in prompt_content}")
        print(f"✓ GitHub API mentioned: {'github_api' in prompt_content}")
        print(f"✓ Slack API mentioned: {'slack_api' in prompt_content}")
        
        assert 'API:' in prompt_content, "api_name not included in LLM prompt"
        assert 'github_api' in prompt_content, "GitHub api_name not found"
        assert 'slack_api' in prompt_content, "Slack api_name not found"
        
        print("✓ BulletPlanReasoner api_name extraction test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ BulletPlanReasoner test FAILED: {e}\n")
        return False

def test_standard_reasoner_api_name_extraction():
    """Test that StandardReasoner extracts and uses api_name in tool selection."""
    print("Testing StandardReasoner api_name extraction...")
    
    # Mock search results with api_name
    mock_tools = [
        {
            'id': 'tool1',
            'name': 'GitHub API Tool',
            'description': 'Tool for GitHub operations',
            'api_name': 'github_api'
        },
        {
            'id': 'tool2',
            'name': 'Slack API Tool', 
            'description': 'Tool for Slack operations',
            'api_name': 'slack_api'
        }
    ]
    
    # Create mocks
    mock_jentic = Mock(spec=JenticClient)
    mock_llm = Mock(spec=LiteLLMChatLLM)
    mock_llm.chat.return_value = "tool2"  # Select Slack tool
    
    # Create reasoner
    reasoner = StandardReasoner(
        jentic_client=mock_jentic,
        llm=mock_llm
    )
    
    # Call select_tool
    try:
        selected_tool = reasoner.select_tool("Send message to Slack", mock_tools)
        
        # Verify the LLM was called with api_name information
        llm_call_args = mock_llm.chat.call_args
        prompt_content = llm_call_args[1]['messages'][1]['content']  # User message
        
        print(f"✓ Tool selected: {selected_tool['id'] if selected_tool else None}")
        print(f"✓ LLM prompt contains api_name: {'API:' in prompt_content}")
        print(f"✓ GitHub API mentioned: {'github_api' in prompt_content}")
        print(f"✓ Slack API mentioned: {'slack_api' in prompt_content}")
        
        assert 'API:' in prompt_content, "api_name not included in LLM prompt"
        assert 'github_api' in prompt_content, "GitHub api_name not found"
        assert 'slack_api' in prompt_content, "Slack api_name not found"
        
        print("✓ StandardReasoner api_name extraction test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ StandardReasoner test FAILED: {e}\n")
        return False

if __name__ == "__main__":
    print("=== Testing api_name extraction in both reasoners ===\n")
    
    bullet_success = test_bullet_reasoner_api_name_extraction()
    standard_success = test_standard_reasoner_api_name_extraction()
    
    if bullet_success and standard_success:
        print("🎉 All tests PASSED! Both reasoners now extract and use api_name information.")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED!")
        sys.exit(1)
