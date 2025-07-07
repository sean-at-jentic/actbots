"""
CLI-based inbox that reads goals from standard input.
"""

import sys
from typing import Optional, TextIO, Dict, List, Callable
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .base_inbox import BaseInbox
from ...utils.shared_console import console


class CLIInbox(BaseInbox):
    """
    Inbox that reads goals from command line input.

    Reads from stdin and treats each line as a separate goal.
    Handles built-in commands like help, quit, history.
    Useful for interactive CLI agents and testing.
    """

    def __init__(
        self, input_stream: Optional[TextIO] = None, prompt: str = "Enter goal: "
    ):
        """
        Initialize CLI inbox.

        Args:
            input_stream: Stream to read from (defaults to stdin)
            prompt: Prompt to display when asking for input
        """
        self.input_stream = input_stream or sys.stdin
        self.prompt = prompt
        self._closed = False
        self._current_goal: Optional[str] = None
        self._history: List[str] = []
        self._commands = self._setup_commands()

    def _setup_commands(self) -> Dict[str, Callable[[str], bool]]:
        """Setup available CLI commands. Returns True if command was handled."""
        return {
            "help": self._handle_help_command,
            "quit": self._handle_quit_command,
            "exit": self._handle_quit_command,
            "history": self._handle_history_command,
        }

    def _handle_help_command(self, args: str) -> bool:
        """Handle help command."""
        help_text = Text()
        help_text.append("Available Commands:\n", style="bold blue")
        help_text.append("\n")
        help_text.append("<goal description>", style="green")
        help_text.append(" - Execute a goal with the given description\n")
        help_text.append("history", style="green")
        help_text.append(" - Show command history\n")
        help_text.append("help", style="green")
        help_text.append(" - Show this help message\n")
        help_text.append("exit/quit", style="green")
        help_text.append(" - Exit the CLI agent\n")
        help_text.append("\n")
        help_text.append("Examples:\n", style="bold yellow")
        help_text.append("Find information about machine learning\n", style="cyan")
        help_text.append("Summarize the text in file.txt\n", style="cyan")
        help_text.append("Create a plan for my project\n", style="cyan")

        panel = Panel(help_text, title="Help", border_style="blue", padding=(1, 2))

        console.print(panel)
        return True

    def _handle_quit_command(self, args: str) -> bool:
        """Handle quit/exit command."""
        console.print("[yellow]Shutting down CLI agent...[/yellow]")
        self._closed = True
        return True

    def _handle_history_command(self, args: str) -> bool:
        """Handle history command."""
        if not self._history:
            console.print("[yellow]No command history available[/yellow]")
            return True

        table = Table(
            title="Command History", show_header=True, header_style="bold blue"
        )
        table.add_column("#", style="cyan", width=4)
        table.add_column("Command", style="white")

        # Show last 20 commands
        recent_history = self._history[-20:]
        for i, command in enumerate(recent_history, 1):
            table.add_row(str(i), command)

        console.print(table)
        return True

    def get_next_goal(self) -> Optional[str]:
        """
        Get the next goal from user input.

        Returns:
            Next goal string, or None if no more input available
        """
        if self._closed:
            return None

        try:
            # Display prompt if using stdin
            if self.input_stream == sys.stdin:
                console.print("[bold blue]ActBots[/bold blue]: ", end="")

            line = self.input_stream.readline()

            # EOF reached
            if not line:
                self._closed = True
                return None

            user_input = line.strip()

            # Empty line
            if not user_input:
                return self.get_next_goal()  # Try again

            # Add to history
            self._history.append(user_input)

            # Check if input is a command
            parts = user_input.split(None, 1)
            command = parts[0].lower() if parts else ""
            args = parts[1] if len(parts) > 1 else ""

            if command in self._commands:
                # Handle command and try again for next goal
                self._commands[command](args)
                return self.get_next_goal()

            # It's a goal, not a command
            self._current_goal = user_input
            return user_input

        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Interrupted by user. Goodbye![/yellow]")
            self._closed = True
            return None

    def acknowledge_goal(self, goal: str) -> None:
        """
        Acknowledge that a goal has been processed.

        Args:
            goal: The goal that was successfully processed
        """
        # For CLI inbox, acknowledgment is just logging
        # In more complex implementations, this might update a database
        if goal == self._current_goal:
            self._current_goal = None

    def reject_goal(self, goal: str, reason: str) -> None:
        """
        Reject a goal that couldn't be processed.

        Args:
            goal: The goal that failed to process
            reason: Reason for rejection
        """
        # For CLI inbox, just print the rejection reason
        console.print(f"[red]Goal rejected: {reason}[/red]")
        if goal == self._current_goal:
            self._current_goal = None

    def has_goals(self) -> bool:
        """
        Check if there are pending goals.

        For CLI inbox, this is true unless we've been closed.

        Returns:
            True if goals might be available, False if definitely none
        """
        return not self._closed

    def close(self) -> None:
        """
        Clean up inbox resources.
        """
        self._closed = True
        if self.input_stream != sys.stdin:
            self.input_stream.close()
