from .inbox.base_inbox import BaseInbox
from .inbox.cli_inbox import CLIInbox

# Simple escalation system
from .hitl.base_intervention_hub import BaseInterventionHub, NoEscalation

# Legacy HITL system components (mostly deprecated)
from .outbox.base_outbox import BaseOutbox, MessageType
from .outbox.cli_outbox import CLIOutbox
from .hitl.cli_intervention_hub import CLIInterventionHub

# Controllers
from .base_controller import BaseController
from .cli_controller import CLIController

# Backward compatibility aliases
BaseEscalation = BaseInterventionHub
CLIEscalation = CLIInterventionHub

__all__ = [
    # Core communication
    "BaseInbox",
    "CLIInbox",
    # Controllers
    "BaseController",
    "CLIController",
    # Simple escalation system
    "BaseInterventionHub",
    "CLIInterventionHub",
    "NoEscalation",
    # Backward compatibility aliases
    "BaseEscalation",
    "CLIEscalation",
    # Legacy HITL system (deprecated)
    "BaseOutbox",
    "MessageType",
    "CLIOutbox",
]
