"""
Interactive CLI agent that reads goals from stdin and outputs to stdout.
Provides command-line interface for goal input and execution.
"""
import logging
import sys
from typing import Any, Dict

# Use the shared console instance across components to avoid clashing Live renders.
from ..utils.shared_console import console as shared_console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from .base_agent import BaseAgent
from ..reasoners.base_reasoner import ReasoningResult

logger = logging.getLogger(__name__)


class InteractiveCLIAgent(BaseAgent):
    """
    Interactive command-line agent.
    
    Reads goals from the CLI inbox, processes them using the reasoner,
    and outputs results to stdout. Supports commands like help and quit.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the interactive CLI agent."""
        super().__init__(*args, **kwargs)
        self._running = False
        # Use the shared console for all output and status handling.
        self.console = shared_console
        self._commands = self._setup_commands()
        self._history = []
    
    def _setup_commands(self) -> Dict[str, callable]:
        """Setup available CLI commands."""
        return {
            "help": self._handle_help_command,
            "quit": self._handle_exit_command,
            "exit": self._handle_exit_command,
            "history": self._handle_history_command,
        }
    
    def _display_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_text = Text()
        welcome_text.append("🤖 Jentic ActBots Framework", style="bold blue")
        welcome_text.append("\n\n")
        welcome_text.append("Type 'help' for available commands or enter your goal directly.")
        welcome_text.append("\nType 'quit' to exit.")
        
        panel = Panel(
            welcome_text,
            title="Welcome to ActBots",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def spin(self) -> None:
        """
        Main agent loop that processes goals from inbox.
        
        Continues until the inbox is closed or an exit command is received.
        """
        logger.info("Starting InteractiveCLIAgent")
        self._display_welcome()
        
        self._running = True
        
        try:
            while self.should_continue():
                # Get user input
                user_input = self._get_user_input()
                
                if not user_input.strip():
                    continue
                
                # Process command or goal
                self._process_input(user_input)
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user. Goodbye![/yellow]")
        
        finally:
            self._running = False
            self.close()
            logger.info("InteractiveCLIAgent stopped")
    
    def _get_user_input(self) -> str:
        """Get user input."""
        try:
            self.console.print("[bold blue]ActBots[/bold blue]: ", end="")
            return input().strip()
        except (KeyboardInterrupt, EOFError):
            return "quit"
    
    def _process_input(self, user_input: str) -> None:
        """Process user input as command or goal."""
        # Add to history
        self._history.append(user_input)
        
        # Check if input is a command
        parts = user_input.strip().split(None, 1)
        command = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        if command in self._commands:
            self._commands[command](args)
        else:
            # Process as goal
            self._handle_goal(user_input)
    
    def _handle_goal(self, goal: str) -> None:
        """Handle goal execution."""
        try:
            # Process the goal (no spinner to avoid conflicts with human input)
            self.console.print(f"[green]Processing goal:[/green] {goal}  [dim](working...)[/dim]")

            result = self.process_goal(goal)
            
            # Handle output
            self.handle_output(result)
            
            # Acknowledge successful processing
            self.inbox.acknowledge_goal(goal)
            
        except Exception as e:
            error_msg = f"Error processing goal: {str(e)}"
            logger.error(error_msg)
            
            error_text = Text()
            error_text.append("❌ Goal Failed\n", style="bold red")
            error_text.append(f"Goal: {goal}\n")
            error_text.append(f"Error: {error_msg}\n", style="red")
            
            panel = Panel(
                error_text,
                title="Error",
                border_style="red",
                padding=(1, 2)
            )
            self.console.print(panel)
            
            # Reject the goal
            self.inbox.reject_goal(goal, error_msg)
    
    def _handle_help_command(self, args: str) -> None:
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
        
        panel = Panel(
            help_text,
            title="Help",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def _handle_exit_command(self, args: str) -> None:
        """Handle exit command."""
        self.console.print("[yellow]Shutting down CLI agent...[/yellow]")
        self._running = False
    
    def _handle_history_command(self, args: str) -> None:
        """Handle history command."""
        if not self._history:
            self.console.print("[yellow]No command history available[/yellow]")
            return
        
        table = Table(title="Command History", show_header=True, header_style="bold blue")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Command", style="white")
        
        # Show last 20 commands
        recent_history = self._history[-20:]
        for i, command in enumerate(recent_history, 1):
            table.add_row(str(i), command)
        
        self.console.print(table)
    
    def handle_input(self, input_data: Any) -> str:
        """
        Handle input from the user/environment.
        
        For CLI agent, this just returns the input as-is since
        the inbox already handles input parsing.
        
        Args:
            input_data: Raw input data from the interface
            
        Returns:
            Processed goal string
        """
        return str(input_data).strip()
    
    def handle_output(self, result: ReasoningResult) -> None:
        """
        Handle output to the user/environment.
        
        Formats and prints the reasoning result to stdout using rich formatting.
        
        Args:
            result: Reasoning result to present
        """
        if result.success:
            # Success panel
            success_text = Text()
            success_text.append("✅ Goal Completed Successfully\n", style="bold green")
            success_text.append(f"{result.final_answer}\n")
            
            if result.tool_calls:
                success_text.append(f"\nUsed {len(result.tool_calls)} tool(s) in {result.iterations} iteration(s):\n", style="dim")
                for i, call in enumerate(result.tool_calls, 1):
                    tool_name = call.get('tool_name', call.get('tool_id', 'Unknown'))
                    success_text.append(f"  {i}. {tool_name}\n", style="dim")
            
            panel = Panel(
                success_text,
                title="Success",
                border_style="green",
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            # Error panel
            error_text = Text()
            error_text.append("❌ Goal Failed\n", style="bold red")
            error_text.append(f"{result.final_answer}\n")
            
            if result.error_message:
                error_text.append(f"Error: {result.error_message}\n", style="red")
            
            panel = Panel(
                error_text,
                title="Error",
                border_style="red",
                padding=(1, 2)
            )
            self.console.print(panel)
    
    def should_continue(self) -> bool:
        """
        Determine if the agent should continue processing.
        
        Returns:
            True if agent should keep running, False to stop
        """
        return self._running
    
    def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False
