import pytest
from unittest.mock import MagicMock, patch
import sys
import os

from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.reasoners.standard_reasoner import StandardReasoner
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.communication.controllers.cli_controller import CLIController
from jentic_agents.tests.mocks.jentic_mock import JenticMock
from jentic_agents.utils.llm import BaseLLM
from jentic_agents.reasoners.base_reasoner import ReasoningResult

# Fixture for a fully assembled agent 

@pytest.fixture
def fully_integrated_agent(mocker):
    """
    Sets up a complete InteractiveCLIAgent with all its components,
    using a mock Jentic service for testing.
    """
    # 1. Mock the Jentic SDK to use our mock service
    mock_jentic_instance = JenticMock()
    sys.modules["jentic"] = MagicMock(Jentic=lambda **kwargs: mock_jentic_instance)
    class DummyModels:
        class ApiCapabilitySearchRequest:
            def __init__(self, capability_description, max_results):
                self.capability_description = capability_description
                self.max_results = max_results
    sys.modules["jentic.models"] = DummyModels
    os.environ["JENTIC_API_KEY"] = "mock_key"
    
    # 2. Initialize real components
    client = JenticClient()
    reasoner = StandardReasoner(jentic_client=client)
    memory = ScratchPadMemory()

    # Mock the LLM to prevent network calls and control its output
    reasoner.llm = MagicMock(spec=BaseLLM)
    # Mock the entire run method to isolate the agent's logic
    reasoner.run = MagicMock()

    # 3. Use a mock controller to capture output
    mock_controller = MagicMock(spec=CLIController)
    mock_controller.inbox = MagicMock()
    mock_controller.outbox = MagicMock()
    mock_controller.intervention_hub = MagicMock()
    # The agent gets goals from the inbox, not the controller directly.
    mock_controller.inbox.get_goal.return_value = ("ping", {}) # Simulate user typing 'ping'
    
    # 4. Assemble the agent
    agent = InteractiveCLIAgent(
        controller=mock_controller,
        reasoner=reasoner,
        jentic_client=client,
        memory=memory
    )
    
    return agent, mock_controller

# Integration Tests 

def test_ping_pong_loop(fully_integrated_agent):
    """
    Tests a full agent loop from user input to tool execution and final output.
    The user says "ping", the agent should find the "ping" tool and execute it,
    resulting in the output "pong".
    """
    agent, mock_controller = fully_integrated_agent
    
    # Configure the mock for this test
    agent.reasoner.run.return_value = ReasoningResult(
        final_answer="pong", success=True, iterations=1, tool_calls=[]
    )

    # Run the agent for one cycle
    agent._handle_goal("ping")

    # Assertions
    # 1. Check that the reasoner was called with the goal
    agent.reasoner.run.assert_called_once_with("ping")

    # 2. Check that the outbox displayed the final result.
    mock_controller.outbox.display_reasoning_result.assert_called_once()
    result_arg = mock_controller.outbox.display_reasoning_result.call_args[0][0]
    assert "pong" in result_arg.final_answer


def test_calculator_loop(fully_integrated_agent):
    """
    Tests a more complex loop where the agent needs to find a tool,
    generate parameters for it, and execute it.
    """
    agent, mock_controller = fully_integrated_agent

    # Configure the mock reasoner to return a successful result
    agent.reasoner.run.return_value = ReasoningResult(
        final_answer="The result is 8", success=True, iterations=1, tool_calls=[]
    )

    # Change the mocked user input for this test
    # The agent and controller are already created, so we modify the mock on the agent instance
    agent.controller.inbox.get_goal.return_value = ("What is 5 plus 3?", {})

    # Run the agent for one cycle
    agent._handle_goal("What is 5 plus 3?")

    # Assert that the final output contains the correct answer '8'
    agent.reasoner.run.assert_called_once_with("What is 5 plus 3?")
    mock_controller.outbox.display_reasoning_result.assert_called_once()
    result_arg = mock_controller.outbox.display_reasoning_result.call_args[0][0]
    assert "8" in result_arg.final_answer 