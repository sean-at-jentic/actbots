"""
CLI implementation of BaseController that aggregates CLI-specific communication channels.
"""

from typing import Optional
from .base_controller import BaseController
from .inbox.cli_inbox import CLIInbox
from .hitl.cli_intervention_hub import CLIInterventionHub
from .outbox.cli_outbox import CLIOutbox


class CLIController(BaseController):
    """
    CLI-specific communication controller.

    Aggregates CLI implementations of inbox, intervention hub, and outbox
    for command-line based agent interactions.
    """

    def __init__(
        self,
        inbox: Optional[CLIInbox] = None,
        intervention_hub: Optional[CLIInterventionHub] = None,
        outbox: Optional[CLIOutbox] = None,
    ):
        """
        Initialize CLI controller with optional custom implementations.

        Args:
            inbox: CLI inbox (defaults to CLIInbox with standard prompt)
            intervention_hub: CLI intervention hub (defaults to CLIInterventionHub)
            outbox: CLI outbox (defaults to CLIOutbox with verbose mode)
        """
        super().__init__(
            inbox=inbox or CLIInbox(prompt="Enter goal: "),
            intervention_hub=intervention_hub or CLIInterventionHub(),
            outbox=outbox or CLIOutbox(verbose=True),
        )
