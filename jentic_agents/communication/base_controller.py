"""
Abstract base class for communication controllers that aggregate inbox, intervention hub, and outbox.
"""

from abc import ABC
from .inbox.base_inbox import BaseInbox
from .hitl.base_intervention_hub import BaseInterventionHub
from .outbox.base_outbox import BaseOutbox


class BaseController(ABC):
    """
    Base class for communication controllers.

    Aggregates the three core communication channels:
    - Inbox: Receives goals/tasks from users
    - Intervention Hub: Handles human-in-the-loop requests
    - Outbox: Sends progress updates and results back to users
    """

    def __init__(
        self,
        inbox: BaseInbox,
        intervention_hub: BaseInterventionHub,
        outbox: BaseOutbox,
    ):
        """
        Initialize controller with communication channels.

        Args:
            inbox: Source of goals/tasks
            intervention_hub: Human-in-the-loop interface
            outbox: Result/progress delivery system
        """
        self.inbox = inbox
        self.intervention_hub = intervention_hub
        self.outbox = outbox

    def close(self) -> None:
        """
        Clean up all communication channel resources.
        """
        self.inbox.close()
        self.outbox.close()
        # intervention_hub doesn't have a close method in the current interface
