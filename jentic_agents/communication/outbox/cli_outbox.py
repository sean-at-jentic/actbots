"""
CLI implementation of BaseOutbox for sending results to console.
"""
import json
from typing import Any, Dict, Optional
from .base_outbox import BaseOutbox


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
    
    def send_progress(self, goal_id: str, message: str, step: Optional[str] = None) -> None:
        """Send progress update to console."""
        if self.verbose:
            step_info = f" (Step: {step})" if step else ""
            print(f"🔄 [PROGRESS] Goal {goal_id}{step_info}: {message}")
        else:
            print(f"Progress: {message}")
    
    def send_result(self, goal_id: str, result: Any, success: bool = True) -> None:
        """Send final result to console."""
        status_icon = "✅" if success else "❌"
        status_text = "SUCCESS" if success else "FAILED"
        
        if self.verbose:
            print(f"{status_icon} [RESULT] Goal {goal_id} {status_text}")
            print("Result:")
            if isinstance(result, (dict, list)):
                print(json.dumps(result, indent=2))
            else:
                print(str(result))
        else:
            print(f"Result: {result}")
    
    def send_error(self, goal_id: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Send error notification to console."""
        if self.verbose:
            print(f"❌ [ERROR] Goal {goal_id}: {error_message}")
            if context:
                print("Context:")
                print(json.dumps(context, indent=2))
        else:
            print(f"Error: {error_message}")
    
    def send_step_complete(self, goal_id: str, step: str, result: Any) -> None:
        """Send step completion notification to console."""
        if self.verbose:
            print(f"✓ [STEP COMPLETE] Goal {goal_id}: {step}")
            if result:
                print(f"  Result: {result}")
        else:
            print(f"Step complete: {step}")
    
    def send_status_change(self, goal_id: str, old_status: str, new_status: str) -> None:
        """Send status change notification to console."""
        if self.verbose:
            print(f"🔄 [STATUS] Goal {goal_id}: {old_status} → {new_status}")
        else:
            print(f"Status: {old_status} → {new_status}")
    
    def close(self) -> None:
        """Clean up CLI outbox resources."""
        if self.verbose:
            print("📤 [OUTBOX] Closed")
        # No resources to clean up for console output 