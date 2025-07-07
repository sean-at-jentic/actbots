"""
CLI implementation of BaseOutbox for sending results to console.
"""

import json
from typing import Any, Dict, Optional
from rich.panel import Panel
from rich.text import Text

from .base_outbox import BaseOutbox
from ...utils.shared_console import console


class CLIOutbox(BaseOutbox):
    """
    CLI implementation that sends results to console output.

    Useful for development, testing, and simple command-line agents.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize CLI outbox.

        Args:
            verbose: If True, includes detailed formatting and timestamps
        """
        self.verbose = verbose

    def display_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_text = Text()
        welcome_text.append("🤖 Jentic ActBots Framework", style="bold blue")
        welcome_text.append("\n\n")
        welcome_text.append(
            "Type 'help' for available commands or enter your goal directly."
        )
        welcome_text.append("\nType 'quit' to exit.")

        panel = Panel(
            welcome_text,
            title="Welcome to ActBots",
            border_style="blue",
            padding=(1, 2),
        )

        console.print(panel)

    def display_goal_start(self, goal: str) -> None:
        """Display that goal processing has started."""
        console.print(
            f"[green]Processing goal:[/green] {goal}  [dim](working...)[/dim]"
        )

    def display_reasoning_result(self, result: Any) -> None:
        """
        Display a reasoning result with rich formatting.

        Args:
            result: ReasoningResult to display
        """
        if getattr(result, "success", False):
            # Success panel
            success_text = Text()
            success_text.append("✅ Goal Completed Successfully\n", style="bold green")
            success_text.append(f"{getattr(result, 'final_answer', result)}\n")

            tool_calls = getattr(result, "tool_calls", [])
            iterations = getattr(result, "iterations", 0)
            if tool_calls:
                success_text.append(
                    f"\nUsed {len(tool_calls)} tool(s) in {iterations} iteration(s):\n",
                    style="dim",
                )
                for i, call in enumerate(tool_calls, 1):
                    tool_name = call.get("tool_name", call.get("tool_id", "Unknown"))
                    success_text.append(f"  {i}. {tool_name}\n", style="dim")

            panel = Panel(
                success_text, title="Success", border_style="green", padding=(1, 2)
            )
            console.print(panel)
        else:
            # Error panel
            error_text = Text()
            error_text.append("❌ Goal Failed\n", style="bold red")
            error_text.append(f"{getattr(result, 'final_answer', result)}\n")

            error_message = getattr(result, "error_message", None)
            if error_message:
                error_text.append(f"Error: {error_message}\n", style="red")

            panel = Panel(error_text, title="Error", border_style="red", padding=(1, 2))
            console.print(panel)

    def display_goal_error(self, goal: str, error_msg: str) -> None:
        """Display a goal processing error."""
        error_text = Text()
        error_text.append("❌ Goal Failed\n", style="bold red")
        error_text.append(f"Goal: {goal}\n")
        error_text.append(f"Error: {error_msg}\n", style="red")

        panel = Panel(error_text, title="Error", border_style="red", padding=(1, 2))
        console.print(panel)

    def send_progress(
        self, goal_id: str, message: str, step: Optional[str] = None
    ) -> None:
        """Send progress update to console."""
        if self.verbose:
            step_info = f" (Step: {step})" if step else ""
            console.print(f"🔄 [PROGRESS] Goal {goal_id}{step_info}: {message}")
        else:
            console.print(f"Progress: {message}")

    def send_result(self, goal_id: str, result: Any, success: bool = True) -> None:
        """Send final result to console."""
        status_icon = "✅" if success else "❌"
        status_text = "SUCCESS" if success else "FAILED"

        if self.verbose:
            console.print(f"{status_icon} [RESULT] Goal {goal_id} {status_text}")
            console.print("Result:")
            if isinstance(result, (dict, list)):
                console.print(json.dumps(result, indent=2))
            else:
                console.print(str(result))
        else:
            console.print(f"Result: {result}")

    def send_error(
        self, goal_id: str, error_message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send error notification to console."""
        if self.verbose:
            console.print(f"❌ [ERROR] Goal {goal_id}: {error_message}")
            if context:
                console.print("Context:")
                console.print(json.dumps(context, indent=2))
        else:
            console.print(f"Error: {error_message}")

    def send_step_complete(self, goal_id: str, step: str, result: Any) -> None:
        """Send step completion notification to console."""
        if self.verbose:
            console.print(f"✓ [STEP COMPLETE] Goal {goal_id}: {step}")
            if result:
                console.print(f"  Result: {result}")
        else:
            console.print(f"Step complete: {step}")

    def send_status_change(
        self, goal_id: str, old_status: str, new_status: str
    ) -> None:
        """Send status change notification to console."""
        if self.verbose:
            console.print(f"🔄 [STATUS] Goal {goal_id}: {old_status} → {new_status}")
        else:
            console.print(f"Status: {old_status} → {new_status}")

    def close(self) -> None:
        """Clean up CLI outbox resources."""
        if self.verbose:
            console.print("📤 [OUTBOX] Closed")
        # No resources to clean up for console output
