"""
Simple escalation system that allows reasoners to request human help when they choose to.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseInterventionHub(ABC):
    """
    Simple interface for reasoners to escalate to humans when they decide they need help.
    No automatic triggers - the reasoner has full control over when to escalate.
    """

    @abstractmethod
    def ask_human(self, question: str, context: Optional[str] = None) -> str:
        """
        Ask a human for help with a question.

        Args:
            question: The question to ask the human
            context: Optional context to help the human understand the situation

        Returns:
            Human's response as a string
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if human escalation is available.

        Returns:
            True if a human can be reached, False otherwise
        """
        pass


# Keep the old names for backward compatibility
BaseEscalation = BaseInterventionHub


class NoEscalation(BaseInterventionHub):
    """
    Null escalation that doesn't provide human help.
    Used when the agent should work completely autonomously.
    """

    def ask_human(self, question: str, context: Optional[str] = None) -> str:
        """No human available - return empty response."""
        return ""

    def is_available(self) -> bool:
        """No escalation available."""
        return False
