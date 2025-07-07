# Human-in-the-Loop Agent Communication

This document describes the three communication abstractions that enable human-in-the-loop (HITL) agent interactions:

## Overview

The system uses three complementary abstractions to enable robust human-AI collaboration:

1. **📥 Inbox** - Receives goals/tasks from humans
2. **📤 Outbox** - Sends progress updates and results back to humans
3. **🤝 InterventionHub** - Handles mid-execution requests for human assistance

## Architecture

```
Human ←→ Inbox ←→ Agent ←→ InterventionHub ←→ Human
       ↖        ↗
         Outbox
```

## Communication Flow

### 1. Goal Input (Inbox)
- Human provides goals/tasks to the agent
- Supports various input sources (CLI, web, queue, etc.)
- Agent acknowledges or rejects goals based on processing results

### 2. Progress & Results (Outbox) 
- Agent sends real-time progress updates
- Reports step completions and status changes
- Delivers final results or error notifications
- Enables transparent agent operation

### 3. Human Intervention (InterventionHub)
- Agent requests help when encountering ambiguity or errors
- Supports multiple intervention types:
  - Goal clarification
  - Tool selection assistance
  - Parameter correction
  - Plan review and modification
  - Step guidance for failures
  - Decision points with multiple options

## Intervention Types

### Goal Clarification
When the agent encounters an ambiguous goal:
```python
clarified_goal = intervention_hub.request_goal_clarification(
    goal_id="task_123",
    original_goal="Help with the project",
    issue="Goal is too vague",
    suggested_interpretations=[
        "Help with project planning",
        "Help with project execution", 
        "Help with project documentation"
    ]
)
```

### Tool Selection Help
When the agent can't find appropriate tools:
```python
tool_id = intervention_hub.request_tool_selection_help(
    goal_id="task_123",
    step_description="Send notification to team",
    available_tools=search_results,
    search_query="send notification",
    issue="Multiple communication tools available"
)
```

### Parameter Assistance
When tool parameters are incorrect:
```python
corrected_params = intervention_hub.request_parameter_assistance(
    goal_id="task_123",
    tool_id="email_sender",
    tool_schema=email_schema,
    attempted_params={"to": "invalid_email"},
    error_message="Invalid email address format"
)
```

### Plan Review
When generated plans need human oversight:
```python
reviewed_plan = intervention_hub.request_plan_review(
    goal_id="task_123",
    original_goal="Deploy new feature",
    generated_plan=["Update code", "Run tests", "Deploy to production"],
    concerns="Plan lacks staging environment step"
)
```

### Step Guidance
When steps repeatedly fail:
```python
guidance = intervention_hub.request_step_guidance(
    goal_id="task_123",
    failed_step="Connect to database",
    error_history=["Connection timeout", "Authentication failed"],
    context={"retries": 3, "last_success": "2 hours ago"}
)
```

### Decision Points
When the agent needs human decision-making:
```python
choice = intervention_hub.request_decision(
    goal_id="task_123",
    decision_point="Multiple API versions available",
    options=["Use v1 (stable)", "Use v2 (beta)", "Use both"],
    context={"current_version": "v1", "features_needed": ["async", "webhooks"]}
)
```

## Implementation Examples

### CLI Implementation
For command-line interaction:
```python
from jentic_agents.communication import CLIInbox, CLIOutbox, CLIInterventionHub

inbox = CLIInbox()
outbox = CLIOutbox(verbose=True)
intervention_hub = CLIInterventionHub()
```

### Web/API Implementation  
For web-based interaction:
```python
from jentic_agents.communication import WebInbox, WebOutbox, WebInterventionHub

inbox = WebInbox(port=8080)
outbox = WebOutbox(websocket_url="ws://localhost:8080/progress")
intervention_hub = WebInterventionHub(api_endpoint="/intervention")
```

### Message Queue Implementation
For asynchronous/distributed systems:
```python
from jentic_agents.communication import QueueInbox, QueueOutbox, QueueInterventionHub

inbox = QueueInbox(queue_name="agent_goals")
outbox = QueueOutbox(queue_name="agent_results") 
intervention_hub = QueueInterventionHub(queue_name="agent_interventions")
```

## Integration with BulletPlanReasoner

The intervention hub can be integrated into the existing `BulletPlanReasoner` to add human-in-the-loop capabilities:

```python
class HITLBulletPlanReasoner(BulletPlanReasoner):
    def __init__(self, intervention_hub: BaseInterventionHub, **kwargs):
        super().__init__(**kwargs)
        self.intervention_hub = intervention_hub
    
    def reflect(self, current_step: Step, err_msg: str) -> bool:
        """Enhanced reflection with human intervention."""
        if current_step.reflection_attempts >= 1:
            # Ask for human guidance instead of giving up
            guidance = self.intervention_hub.request_step_guidance(
                goal_id=self.current_goal_id,
                failed_step=current_step.text,
                error_history=[err_msg],
                context={"attempts": current_step.reflection_attempts}
            )
            current_step.text = guidance
            current_step.status = "pending"
            current_step.tool_id = None
            return True
        else:
            return super().reflect(current_step, err_msg)
```

## Benefits

1. **Robustness**: Agents can handle ambiguous or complex scenarios with human guidance
2. **Transparency**: Users see real-time progress and can intervene when needed
3. **Flexibility**: Different communication channels for different deployment scenarios
4. **Learning**: Human interventions can be logged to improve future autonomous performance
5. **Trust**: Users maintain control and oversight of agent actions

## Running the Example

Try the human-in-the-loop example:

```bash
cd jentic_agents
python -m examples.hitl_example
```

This will start an interactive agent that demonstrates:
- Goal clarification for ambiguous requests
- Plan review for complex tasks  
- Step guidance for failures
- Decision points for multiple options
- Progress reporting throughout execution 