"""
Abstract base class for AI agents that compose reasoner, memory, inbox, and Jentic client.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..reasoners.base_reasoner import BaseReasoner, ReasoningResult
from ..memory.base_memory import BaseMemory
from ..communication.inbox.base_inbox import BaseInbox
from ..communication.hitl.base_intervention_hub import BaseInterventionHub
from ..communication.outbox.base_outbox import BaseOutbox
from ..communication.base_controller import BaseController
from ..platform.jentic_client import JenticClient


class BaseAgent(ABC):
    """
    Base class for AI agents that wire together core components.

    Composes:
    - Reasoner: Implements the reasoning loop logic
    - Memory: Stores and retrieves information across sessions
    - Communication: Inbox/outbox/intervention hub for user interaction
    - JenticClient: Interface to Jentic workflows and tools

    Concrete agents override I/O methods while inheriting core behavior.
    """

    def __init__(
        self,
        reasoner: BaseReasoner,
        memory: BaseMemory,
        jentic_client: Optional[JenticClient] = None,
        *,
        # Option 1: Single controller (preferred)
        controller: Optional[BaseController] = None,
        # Option 2: Individual components (backward compatible)
        inbox: Optional[BaseInbox] = None,
        intervention_hub: Optional[BaseInterventionHub] = None,
        outbox: Optional[BaseOutbox] = None,
    ):
        """
        Initialize agent with core components.

        Args:
            reasoner: Reasoning loop implementation
            memory: Memory backend for storing information
            jentic_client: Interface to Jentic platform
            controller: Communication controller (preferred - aggregates inbox/outbox/intervention_hub)
            inbox: Source of goals/tasks (if not using controller)
            intervention_hub: Human-in-the-loop interface (if not using controller)
            outbox: Result/progress delivery system (if not using controller)
        """
        self.reasoner = reasoner
        self.memory = memory
        self.jentic_client = jentic_client

        # Set up communication channels
        if controller:
            self.controller = controller
            self.inbox = controller.inbox
            self.intervention_hub = controller.intervention_hub
            self.outbox = controller.outbox
        else:
            self.controller = None
            self.inbox = inbox
            self.intervention_hub = intervention_hub
            self.outbox = outbox

        # Sync the intervention hub with the reasoner (if supported)
        if self.intervention_hub is not None and hasattr(self.reasoner, "escalation"):
            try:
                self.reasoner.escalation = self.intervention_hub
            except Exception:
                # Reasoner may not allow assignment; ignore silently
                pass

    @abstractmethod
    def spin(self) -> None:
        """
        Main agent loop that processes goals from inbox.

        This is the primary entry point for running the agent.
        Concrete implementations define how the agent interacts with users.
        """
        pass

    def process_goal(self, goal: str) -> ReasoningResult:
        """
        Process a single goal using the reasoning loop.

        Args:
            goal: The objective or question to process

        Returns:
            ReasoningResult with the agent's response
        """
        # Store the goal in memory
        self.memory.store("current_goal", goal)

        # Execute reasoning loop
        result = self.reasoner.run(goal)

        # Store the result in memory
        self.memory.store("last_result", result.model_dump())

        return result

    @abstractmethod
    def handle_input(self, input_data: Any) -> str:
        """
        Handle input from the user/environment.

        Args:
            input_data: Raw input data from the interface

        Returns:
            Processed goal string
        """
        pass

    @abstractmethod
    def handle_output(self, result: ReasoningResult) -> None:
        """
        Handle output to the user/environment.

        Args:
            result: Reasoning result to present
        """
        pass

    @abstractmethod
    def should_continue(self) -> bool:
        """
        Determine if the agent should continue processing.

        Returns:
            True if agent should keep running, False to stop
        """
        pass

    def close(self) -> None:
        """
        Clean up agent resources.

        Closes communication channels and other resources.
        """
        if self.controller:
            self.controller.close()
        else:
            if self.inbox:
                self.inbox.close()
            if self.outbox:
                self.outbox.close()
