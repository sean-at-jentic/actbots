"""
CLI implementation of BaseInterventionHub for simple command-line escalation.
"""

from typing import Optional

# Use the shared console instance across components.
from ...utils.shared_console import console as shared_console

from .base_intervention_hub import BaseInterventionHub


class CLIInterventionHub(BaseInterventionHub):
    """
    Simple CLI-based escalation that prompts the human via console.

    Uses Rich's ``Console`` for both printing and reading user input. Rich will
    automatically pause any active *live* or *status* renders (such as the
    spinner used by ``InteractiveCLIAgent``) while waiting for input, which
    prevents the user's typed text from being erased or "blanked out".
    """

    def __init__(self) -> None:
        super().__init__()
        # Use shared console for consistent behaviour.
        self._console = shared_console

    def ask_human(self, question: str, context: Optional[str] = None) -> str:
        """Prompt the human user for help via the command-line interface."""

        self._console.print("\n" + "=" * 60)
        self._console.print("🤖➡️👤 [bold]AGENT REQUESTING HELP[/bold]")
        self._console.print("=" * 60)

        if context:
            self._console.print(f"[bold]Context:[/bold] {context}")
            self._console.print("-" * 40)

        self._console.print(f"[bold]Question:[/bold] {question}\n")

        # Rich's Console.input pauses any active Live rendering to avoid
        # flickering or loss of the user's typed characters.
        response: str = self._console.input("Your response: ").strip()

        self._console.print("=" * 60 + "\n")

        return response

    def is_available(self) -> bool:
        """CLI escalation is always available."""
        return True


# Backward compatibility aliases
CLIEscalation = CLIInterventionHub
