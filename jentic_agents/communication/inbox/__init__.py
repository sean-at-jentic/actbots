"""
Inbox communication components for receiving goals/tasks from humans.
"""

from .base_inbox import BaseInbox
from .cli_inbox import CLIInbox

__all__ = ["BaseInbox", "CLIInbox"]
