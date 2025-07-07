"""
Outbox communication components for sending progress updates and results to humans.
"""

from .base_outbox import BaseOutbox, MessageType
from .cli_outbox import CLIOutbox

__all__ = [
    "BaseOutbox",
    "MessageType", 
    "CLIOutbox"
] 