"""
CLI implementation of BaseInterventionHub for simple command-line escalation.
"""
from typing import Optional
from .base_intervention_hub import BaseInterventionHub


class CLIInterventionHub(BaseInterventionHub):
    """
    Simple CLI-based escalation that prompts the human via console.
    """
    
    def ask_human(self, question: str, context: Optional[str] = None) -> str:
        """Ask human via CLI prompt."""
        print("\n" + "="*60)
        print("🤖➡️👤 AGENT REQUESTING HELP")
        print("="*60)
        
        if context:
            print(f"Context: {context}")
            print("-" * 40)
        
        print(f"Question: {question}")
        print()
        
        response = input("Your response: ").strip()
        print("="*60)
        print()
        
        return response
    
    def is_available(self) -> bool:
        """CLI escalation is always available."""
        return True


# Backward compatibility aliases
CLIEscalation = CLIInterventionHub 