"""
Example demonstrating human-in-the-loop agent interaction using inbox, outbox, and intervention hub.

This example shows how the three communication abstractions work together:
- BaseInbox: Receives goals/tasks from users
- BaseOutbox: Sends progress updates and results back to users  
- BaseInterventionHub: Handles mid-execution requests for human assistance

Run with: python -m jentic_agents.examples.hitl_example
"""

import uuid
from typing import Dict, Any
from ..communication import CLIInbox, CLIOutbox, CLIInterventionHub
from ..communication.hitl.base_intervention_hub import InterventionType


class MockHITLAgent:
    """
    Mock agent demonstrating human-in-the-loop capabilities.
    
    This is a simplified agent that shows how to integrate the three
    communication channels for robust human-AI collaboration.
    """
    
    def __init__(self, inbox: CLIInbox, outbox: CLIOutbox, intervention_hub: CLIInterventionHub):
        self.inbox = inbox
        self.outbox = outbox
        self.intervention_hub = intervention_hub
        self.current_goal_id = None
    
    def run(self):
        """Main agent loop with human-in-the-loop support."""
        print("🤖 Human-in-the-Loop Agent Started")
        print("Enter goals at the prompt. Type 'quit' to exit.\n")
        
        try:
            # Main goal processing loop
            for goal in self.inbox.goal_stream():
                self.current_goal_id = str(uuid.uuid4())[:8]
                
                try:
                    self.outbox.send_progress(
                        self.current_goal_id,
                        f"Starting work on: {goal}"
                    )
                    
                    # Process the goal with potential human intervention
                    result = self.process_goal_with_hitl(goal)
                    
                    # Send successful result
                    self.outbox.send_result(self.current_goal_id, result, success=True)
                    self.inbox.acknowledge_goal(goal)
                    
                except Exception as e:
                    # Send error and reject goal
                    self.outbox.send_error(self.current_goal_id, str(e))
                    self.inbox.reject_goal(goal, str(e))
                    
        except KeyboardInterrupt:
            print("\n🛑 Agent interrupted by user")
        finally:
            self.cleanup()
    
    def process_goal_with_hitl(self, goal: str) -> str:
        """Process a goal with human-in-the-loop intervention points."""
        
        # Step 1: Goal Clarification (if needed)
        if self.is_goal_ambiguous(goal):
            self.outbox.send_progress(
                self.current_goal_id,
                "Goal appears ambiguous, requesting clarification"
            )
            
            clarified_goal = self.intervention_hub.request_goal_clarification(
                goal_id=self.current_goal_id,
                original_goal=goal,
                issue="Goal is too vague or has multiple possible interpretations",
                suggested_interpretations=[
                    f"Interpret as a simple task: {goal}",
                    f"Interpret as a complex workflow: {goal}",
                    f"Interpret as a research question: {goal}"
                ]
            )
            goal = clarified_goal
            self.outbox.send_progress(
                self.current_goal_id,
                f"Goal clarified to: {goal}"
            )
        
        # Step 2: Plan Generation & Review
        self.outbox.send_progress(self.current_goal_id, "Generating plan")
        
        plan = self.generate_plan(goal)
        
        # Ask for plan review if it's complex
        if len(plan) > 3:
            self.outbox.send_progress(
                self.current_goal_id,
                "Plan is complex, requesting human review"
            )
            
            reviewed_plan = self.intervention_hub.request_plan_review(
                goal_id=self.current_goal_id,
                original_goal=goal,
                generated_plan=plan,
                concerns="Plan has many steps and may need human oversight"
            )
            plan = reviewed_plan
        
        # Step 3: Execute Plan with Intervention Support
        results = []
        for i, step in enumerate(plan, 1):
            self.outbox.send_progress(
                self.current_goal_id,
                f"Executing step {i}/{len(plan)}: {step}"
            )
            
            try:
                # Simulate step execution with potential failures
                step_result = self.execute_step_with_retry(step, max_retries=2)
                results.append(step_result)
                
                self.outbox.send_step_complete(
                    self.current_goal_id,
                    step,
                    step_result
                )
                
            except Exception as e:
                # Request human guidance for failed steps
                guidance = self.intervention_hub.request_step_guidance(
                    goal_id=self.current_goal_id,
                    failed_step=step,
                    error_history=[str(e)],
                    context={"current_results": results, "remaining_steps": plan[i:]}
                )
                
                # Apply guidance and retry
                self.outbox.send_progress(
                    self.current_goal_id,
                    f"Applying human guidance: {guidance}"
                )
                
                # Simulate applying guidance
                step_result = f"Completed with guidance: {guidance}"
                results.append(step_result)
                
                self.outbox.send_step_complete(
                    self.current_goal_id,
                    step,
                    step_result
                )
        
        # Step 4: Final Decision Point (if needed)
        if len(results) > 1:
            decision = self.intervention_hub.request_decision(
                goal_id=self.current_goal_id,
                decision_point="How should the results be combined?",
                options=[
                    "Combine all results into a summary",
                    "Return the most important result only",
                    "Return all results as a list",
                    "Let me specify a custom format"
                ],
                context={"results": results}
            )
            
            if decision == 0:
                final_result = f"Summary of {len(results)} completed steps: " + "; ".join(results)
            elif decision == 1:
                final_result = results[-1]  # Last result
            elif decision == 2:
                final_result = results
            else:
                # Custom format - could ask for more specific instructions
                final_result = {"goal": goal, "steps_completed": len(results), "results": results}
        else:
            final_result = results[0] if results else "No results produced"
        
        return final_result
    
    def is_goal_ambiguous(self, goal: str) -> bool:
        """Simple heuristic to detect ambiguous goals."""
        ambiguous_words = ["maybe", "might", "could", "possibly", "unclear", "help", "?"]
        return any(word in goal.lower() for word in ambiguous_words) or len(goal.split()) < 3
    
    def generate_plan(self, goal: str) -> list[str]:
        """Generate a simple plan for the goal."""
        # Simple plan generation logic
        words = goal.lower().split()
        
        if "send" in words or "email" in words:
            return [
                "Identify recipient",
                "Draft email content", 
                "Review email for accuracy",
                "Send email",
                "Confirm delivery"
            ]
        elif "create" in words or "make" in words:
            return [
                "Gather requirements",
                "Design structure",
                "Implement solution",
                "Test and validate",
                "Finalize deliverable"
            ]
        elif "research" in words or "find" in words:
            return [
                "Define search criteria",
                "Search relevant sources",
                "Analyze findings",
                "Summarize results"
            ]
        else:
            return [
                "Analyze the request",
                "Execute the task",
                "Verify completion"
            ]
    
    def execute_step_with_retry(self, step: str, max_retries: int = 2) -> str:
        """Execute a step with retry logic and potential failure."""
        # Simulate occasional failures to demonstrate intervention
        import random
        
        for attempt in range(max_retries + 1):
            if random.random() < 0.2:  # 20% failure rate
                if attempt == max_retries:
                    raise Exception(f"Step failed after {max_retries} retries: {step}")
                continue
            else:
                return f"Completed: {step}"
        
        return f"Completed: {step}"
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        self.inbox.close()
        self.outbox.close()
        self.intervention_hub.close()
        print("✅ Cleanup complete")


def main():
    """Run the human-in-the-loop agent example."""
    
    # Initialize communication channels
    inbox = CLIInbox()
    outbox = CLIOutbox(verbose=True)
    intervention_hub = CLIInterventionHub()
    
    # Create and run the agent
    agent = MockHITLAgent(inbox, outbox, intervention_hub)
    agent.run()


if __name__ == "__main__":
    main() 