"""
Abstract base class for outbox systems that deliver results from agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from enum import Enum


class MessageType(Enum):
    """Types of messages that can be sent through the outbox."""

    PROGRESS_UPDATE = "progress_update"
    STEP_COMPLETE = "step_complete"
    GOAL_COMPLETE = "goal_complete"
    ERROR = "error"
    RESULT = "result"
    STATUS_CHANGE = "status_change"


class BaseOutbox(ABC):
    """
    Abstract base class for result/progress delivery systems.

    Provides a way to send results, progress updates, and status changes
    back to users/systems that requested the goals.
    """

    @abstractmethod
    def send_progress(
        self, goal_id: str, message: str, step: Optional[str] = None
    ) -> None:
        """
        Send a progress update for an ongoing goal.

        Args:
            goal_id: Identifier for the goal being processed
            message: Progress message to send
            step: Optional current step description
        """
        pass

    @abstractmethod
    def send_result(self, goal_id: str, result: Any, success: bool = True) -> None:
        """
        Send the final result of goal processing.

        Args:
            goal_id: Identifier for the completed goal
            result: The result data (could be text, JSON, etc.)
            success: Whether the goal completed successfully
        """
        pass

    @abstractmethod
    def send_error(
        self, goal_id: str, error_message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send an error notification.

        Args:
            goal_id: Identifier for the failed goal
            error_message: Description of the error
            context: Optional additional context about the error
        """
        pass

    @abstractmethod
    def send_step_complete(self, goal_id: str, step: str, result: Any) -> None:
        """
        Send notification that a plan step has completed.

        Args:
            goal_id: Identifier for the goal
            step: Description of the completed step
            result: Result of the step
        """
        pass

    @abstractmethod
    def send_status_change(
        self, goal_id: str, old_status: str, new_status: str
    ) -> None:
        """
        Send notification of status change.

        Args:
            goal_id: Identifier for the goal
            old_status: Previous status
            new_status: New status
        """
        pass

    def send_message(
        self, goal_id: str, message_type: MessageType, content: Any, **kwargs
    ) -> None:
        """
        Generic message sending interface.

        Args:
            goal_id: Identifier for the goal
            message_type: Type of message being sent
            content: Message content
            **kwargs: Additional message-specific parameters
        """
        if message_type == MessageType.PROGRESS_UPDATE:
            self.send_progress(goal_id, content, kwargs.get("step"))
        elif message_type == MessageType.RESULT:
            self.send_result(goal_id, content, kwargs.get("success", True))
        elif message_type == MessageType.ERROR:
            self.send_error(goal_id, content, kwargs.get("context"))
        elif message_type == MessageType.STEP_COMPLETE:
            self.send_step_complete(goal_id, kwargs.get("step", ""), content)
        elif message_type == MessageType.STATUS_CHANGE:
            self.send_status_change(
                goal_id, kwargs.get("old_status", ""), kwargs.get("new_status", "")
            )

    @abstractmethod
    def close(self) -> None:
        """
        Clean up outbox resources.
        """
        pass
