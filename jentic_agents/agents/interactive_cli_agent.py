"""
Interactive CLI agent that reads goals from stdin and outputs to stdout.
Provides command-line interface for goal input and execution.
"""

import logging
from typing import Any

from .base_agent import BaseAgent
from ..reasoners.base_reasoner import ReasoningResult

logger = logging.getLogger(__name__)


class InteractiveCLIAgent(BaseAgent):
    """
    Interactive command-line agent.

    Reads goals from the CLI inbox, processes them using the reasoner,
    and outputs results via the CLI outbox. All UI concerns are delegated
    to the controller components.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the interactive CLI agent."""
        super().__init__(*args, **kwargs)
        self._running = False

    def spin(self) -> None:
        """
        Main agent loop that processes goals from inbox.

        Continues until the inbox is closed or an exit command is received.
        """
        logger.info("Starting InteractiveCLIAgent")

        # Show welcome message via outbox
        if hasattr(self.outbox, "display_welcome"):
            self.outbox.display_welcome()

        self._running = True

        try:
            while self.should_continue():
                # Get next goal from inbox (handles commands internally)
                goal = self.inbox.get_next_goal()

                if goal is None:
                    break

                # Process the goal
                self._handle_goal(goal)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self._running = False
            self.close()
            logger.info("InteractiveCLIAgent stopped")

    def _handle_goal(self, goal: str) -> None:
        """Handle goal execution."""
        try:
            # Notify outbox that goal processing has started
            if hasattr(self.outbox, "display_goal_start"):
                self.outbox.display_goal_start(goal)

            # Process the goal
            result = self.process_goal(goal)

            # Display result via outbox
            if hasattr(self.outbox, "display_reasoning_result"):
                self.outbox.display_reasoning_result(result)
            else:
                # Fallback to standard outbox method
                self.handle_output(result)

            # Acknowledge successful processing
            self.inbox.acknowledge_goal(goal)

        except Exception as e:
            error_msg = f"Error processing goal: {str(e)}"
            logger.error(error_msg)

            # Display error via outbox
            if hasattr(self.outbox, "display_goal_error"):
                self.outbox.display_goal_error(goal, error_msg)

            # Reject the goal
            self.inbox.reject_goal(goal, error_msg)

    def handle_input(self, input_data: Any) -> str:
        """
        Handle input from the user/environment.

        For CLI agent, this just returns the input as-is since
        the inbox already handles input parsing.

        Args:
            input_data: Raw input data from the interface

        Returns:
            Processed goal string
        """
        return str(input_data).strip()

    def handle_output(self, result: ReasoningResult) -> None:
        """
        Handle output to the user/environment.

        Delegates to outbox for display. This is a fallback for outboxes
        that don't have display_reasoning_result method.

        Args:
            result: Reasoning result to present
        """
        # Use standard outbox method as fallback
        self.outbox.send_result("latest", result.final_answer, result.success)

    def should_continue(self) -> bool:
        """
        Determine if the agent should continue processing.

        Returns:
            True if agent should keep running, False to stop
        """
        return self._running and self.inbox.has_goals()

    def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False
