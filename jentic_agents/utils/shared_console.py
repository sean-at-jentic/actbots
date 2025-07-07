# Shared Rich console instance used across CLI components.

from rich.console import Console

# Single console instance reused by CLI agents, intervention hubs, etc.
console = Console()
