"""
Abstract base class for reasoning loops that implement plan → select_tool → act → observe → evaluate → reflect.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import enum


class StepType(enum.Enum):
    """Category of a plan step used by reasoners to decide execution path."""

    TOOL_USING = "tool-using"
    REASONING = "reasoning"


class ReasoningResult(BaseModel):
    """Result object returned by reasoner.run()"""

    final_answer: str
    iterations: int
    tool_calls: List[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None


class BaseReasoner(ABC):
    """
    Abstract base class defining the reasoning loop contract.

    Implements the ReAct pattern: plan → select_tool → act → observe → evaluate → reflect.
    Subclasses implement the specific reasoning logic while maintaining a consistent interface.
    """

    @abstractmethod
    def run(self, goal: str, max_iterations: int = 10) -> ReasoningResult:
        """
        Execute the reasoning loop for a given goal.

        Args:
            goal: The objective or question to reason about
            max_iterations: Maximum number of reasoning iterations

        Returns:
            ReasoningResult with final answer and execution metadata
        """
        pass

    @abstractmethod
    def plan(self, goal: str, context: Dict[str, Any]) -> str:
        """
        Generate a plan for achieving the goal.

        Args:
            goal: The objective to plan for
            context: Current reasoning context and history

        Returns:
            A plan description string
        """
        pass

    @abstractmethod
    def select_tool(
        self, plan: str, available_tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Select the most appropriate tool for executing the current plan.

        Args:
            plan: The current plan description
            available_tools: List of available tools/workflows

        Returns:
            Selected tool metadata, or None if no tool is needed
        """
        pass

    @abstractmethod
    def act(self, tool: Dict[str, Any], plan: str) -> Dict[str, Any]:
        """
        Execute an action using the selected tool.

        Args:
            tool: Tool metadata and definition
            plan: Current plan description

        Returns:
            Action parameters to pass to the tool
        """
        pass

    @abstractmethod
    def observe(self, action_result: Dict[str, Any]) -> str:
        """
        Process and interpret the result of an action.

        Args:
            action_result: Result returned from tool execution

        Returns:
            Observation summary string
        """
        pass

    @abstractmethod
    def evaluate(self, goal: str, observations: List[str]) -> bool:
        """
        Evaluate whether the goal has been achieved based on observations.

        Args:
            goal: The original objective
            observations: List of observation summaries from actions

        Returns:
            True if goal is achieved, False otherwise
        """
        pass

    @abstractmethod
    def reflect(
        self, goal: str, observations: List[str], failed_attempts: List[str]
    ) -> str:
        """
        Reflect on failures and generate improved strategies.

        Args:
            goal: The original objective
            observations: List of observation summaries
            failed_attempts: List of previous failed attempt descriptions

        Returns:
            Reflection insights for improving the approach
        """
        pass
