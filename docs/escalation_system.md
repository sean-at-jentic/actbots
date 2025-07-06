# Simple Escalation System

The escalation system provides a lightweight way for agents to request human help when **they** decide they need it. Unlike the complex HITL system, this approach gives the agent full autonomy to choose when to escalate.

## Philosophy

- **Agent-driven**: The agent chooses when to escalate, not the system
- **No automatic triggers**: No forced escalation based on failure counts or heuristics
- **Simple interface**: Just one method: `ask_human(question, context)`
- **Fail gracefully**: If no human is available, the agent continues autonomously

## Key Components

### BaseEscalation (Interface)
```python
from jentic_agents.communication.escalation import BaseEscalation

class BaseEscalation(ABC):
    def ask_human(self, question: str, context: Optional[str] = None) -> str:
        """Ask a human for help with a question."""
        pass
    
    def is_available(self) -> bool:
        """Check if human escalation is available."""
        pass
```

### CLIEscalation (Implementation)
```python
from jentic_agents.communication.escalation import CLIEscalation

escalation = CLIEscalation()
response = escalation.ask_human(
    "How should I search for weather data?", 
    "I tried 'weather' but found no tools"
)
```

### NoEscalation (Null Implementation)
```python
from jentic_agents.communication.escalation import NoEscalation

escalation = NoEscalation()  # Always returns empty string, is_available() = False
```

## Integration with Reasoner

Add escalation to the `BulletPlanReasoner`:

```python
from jentic_agents.communication.escalation import CLIEscalation
from jentic_agents.reasoners.bullet_list_reasoner import BulletPlanReasoner

escalation = CLIEscalation()
reasoner = BulletPlanReasoner(
    jentic=jentic_client,
    memory=memory,
    llm=llm,
    escalation=escalation  # Add escalation system
)
```

## How It Works

The agent can escalate in several scenarios:

### 1. During Reflection (Step Failures)
When a step fails, the agent chooses what to do:
```
Failed step: Connect to database
Error: Connection timeout

Agent decides:
- TRY: <revised approach>
- ESCALATE: <question for human>  
- SKIP
```

### 2. During Tool Selection
When no tools are found:
```
No tools found for: Send notification

Agent decides:
- ESCALATE: How should I search for notification tools?
- CONTINUE: (proceed with failure)
```

### 3. Agent-Initiated (Future)
Agents can proactively escalate anytime:
```python
if self.escalation.is_available():
    guidance = self.escalation.ask_human(
        "Should I use the staging or production environment?",
        "About to deploy changes"
    )
```

## Example Usage

```python
#!/usr/bin/env python3
import os
from jentic_agents.communication.escalation import CLIEscalation
from jentic_agents.reasoners.bullet_list_reasoner import BulletPlanReasoner
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.utils.llm import LiteLLMChatLLM

# Initialize components
jentic = JenticClient()
memory = ScratchPadMemory()
llm = LiteLLMChatLLM(model="gpt-4o")

# Create escalation system
escalation = CLIEscalation()

# Create reasoner with escalation
reasoner = BulletPlanReasoner(
    jentic=jentic,
    memory=memory, 
    llm=llm,
    escalation=escalation
)

# Run goal - agent will escalate if it chooses to
result = reasoner.run("deploy the new feature")
```

## Benefits

1. **Agent autonomy**: Agent decides when help is needed
2. **No interruption**: Agents work independently until they choose to escalate
3. **Simple interface**: Just ask_human() method
4. **Flexible**: Works with CLI, web, API, or any interface
5. **Fail-safe**: Works without humans (NoEscalation)

## Comparison to HITL System

| Feature | HITL System | Escalation System |
|---------|-------------|-------------------|
| **Trigger** | Automatic (heuristics, failures) | Agent choice |
| **Complexity** | High (3 components) | Low (1 component) |
| **Interruption** | Frequent | Only when agent decides |
| **Agent autonomy** | Limited | Full |
| **Setup** | Complex | Simple |

## Running the Example

Try the escalation system:

```bash
python -m jentic_agents.examples.escalation_example
```

This will demonstrate various scenarios where the agent might choose to escalate to a human for help. 