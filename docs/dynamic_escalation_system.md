# Dynamic Escalation System

The Dynamic Escalation System provides a flexible, agent-driven approach to human-in-the-loop (HITL) assistance. Unlike traditional systems that rely on automatic triggers, this system gives the AI agent full autonomy to request human help whenever it determines it's needed.

## Overview

### Key Features
- **Agent-initiated**: The agent chooses when to escalate, not the system
- **Universal coverage**: Escalation available at every phase of execution
- **Multiple escalation patterns**: Different ways for the agent to request help
- **Contextual guidance**: Rich context provided to humans for better assistance
- **Graceful fallback**: System continues autonomously if human help isn't available
- **Proactive assistance**: Agent can ask for guidance before problems occur

### Philosophy
This system empowers the agent to be proactive about getting help, rather than struggling unnecessarily or failing due to lack of information. It maintains the agent's autonomy while providing a safety net of human expertise.

## Escalation Points

The agent can request human assistance at any of these phases:

### 1. Planning Phase
**When**: During initial goal breakdown and plan creation
**Triggers**: 
- Unclear or ambiguous goals
- Missing context or requirements
- Complex multi-step scenarios requiring domain expertise

**Example**:
```
Goal: "deploy the feature"
Agent response: HUMAN: The goal "deploy the feature" is unclear. Should I deploy to staging or production? What feature specifically?
```

### 2. Tool Selection Phase
**When**: When searching for and choosing appropriate tools
**Triggers**:
- No suitable tools found
- Multiple similar tools with unclear differences
- Uncertainty about which tool best fits the use case

**Example**:
```
Agent response: HUMAN: I found several Discord API tools but I'm not sure which one is best for sending a simple message. Can you help me choose?
```

### 3. Parameter Generation Phase
**When**: Generating parameters for tool execution
**Triggers**:
- Missing critical information (IDs, tokens, credentials)
- Ambiguous parameter requirements
- Need for specific configuration values

**Example**:
```
Agent response: NEED_HUMAN_INPUT: channel_id, webhook_token
```

### 4. Tool Execution Phase
**When**: Tool execution fails and agent chooses to escalate via reflection
**Agent Decision**: Agent analyzes the error and decides if human help would be useful

**Example**:
```
Tool execution failed: Invalid channel_id
Agent in reflection: HUMAN: This tool failed with an invalid channel_id error. Can you provide the correct Discord channel ID?
```

### 5. Reasoning Phase
**When**: Performing internal analysis or data processing
**Triggers**:
- Need for domain expertise
- Unclear data interpretation requirements
- Complex analytical decisions

**Example**:
```
Agent response: HUMAN: I need to analyze this sales data but I'm not sure what format the output should be. What would be most useful?
```

### 6. Reflection Phase
**When**: Analyzing failures and planning recovery
**Triggers**:
- Repeated failures on the same step
- Complex error scenarios
- Need for debugging assistance

**Example**:
```
Agent response: HUMAN: This step keeps failing with authentication errors. Do you have the correct API credentials configured?
```

### 7. Proactive Guidance
**When**: Agent decides it wants guidance before continuing
**Agent Decision**: Agent evaluates its confidence and situation autonomously

**Example**:
```
Agent proactive check: HUMAN: I'm about to delete some database records. Should I create a backup first?
```

## Escalation Patterns

The agent uses these patterns to request help:

### Primary Patterns
- `HUMAN: <question>` - General guidance request
- `ASK_HUMAN: <question>` - Alternative format
- `ESCALATE: <question>` - Debugging/error assistance
- `NEED_HELP: <question>` - Understanding assistance
- `NEED_HUMAN_INPUT: param1, param2` - Specific parameter values

### Response Processing
When the agent uses any escalation pattern, the system:
1. Detects the escalation request
2. Extracts the question/request
3. Provides contextual information to the human
4. Waits for human response
5. Incorporates the response into continued execution

## Implementation Architecture

### Core Components

#### 1. Universal Escalation Processor
```python
def _process_llm_response_for_escalation(self, response: str, context: str = "") -> str:
    """Check if LLM response contains escalation request and handle it."""
```

#### 2. Direct Escalation Method
```python
def _request_human_help(self, question: str, context: str = "") -> str:
    """Direct method for requesting human help from anywhere in the code."""
```

#### 3. Proactive Escalation Check
```python
def _check_for_proactive_escalation(self, state: ReasonerState, iteration: int) -> bool:
    """Let the agent proactively ask for human guidance."""
```

#### 4. Enhanced Error Handling
Automatic escalation for recoverable errors in tool execution, with intelligent error classification and recovery suggestions.

### Integration Points

The escalation system is integrated into every major phase:

```python
# Planning with escalation
enhanced_prompt = f"""
{planning_template}

ESCALATION OPTION:
If this goal is unclear, incomplete, or you need more information to create a good plan, 
you can ask for clarification by starting your response with:
HUMAN: <your question>
"""

# Tool selection with escalation  
enhanced_select_prompt = f"""
{select_template}

ESCALATION OPTION:
If none of these tools seem appropriate or you're unsure which to choose,
you can ask for human guidance by responding with:
HUMAN: <your question about tool selection>
"""

# And similarly for all other phases...
```

## Usage Examples

### Basic Setup
```python
from jentic_agents.communication.hitl.cli_intervention_hub import CLIInterventionHub
from jentic_agents.reasoners.bullet_list_reasoner import BulletPlanReasoner

# Create escalation system
escalation = CLIInterventionHub()

# Create reasoner with escalation
reasoner = BulletPlanReasoner(
    jentic=jentic_client,
    memory=memory,
    llm=llm,
    escalation=escalation  # Enable dynamic escalation
)

# Run goal - agent will escalate if it chooses to
result = reasoner.run("deploy the application to production")
```

### Custom Escalation Hub
```python
class WebEscalationHub(BaseInterventionHub):
    def ask_human(self, question: str, context: Optional[str] = None) -> str:
        # Send to web interface, Slack, email, etc.
        return self.send_to_human_interface(question, context)
    
    def is_available(self) -> bool:
        return self.check_human_availability()
```

### No Escalation Mode
```python
from jentic_agents.communication.hitl.base_intervention_hub import NoEscalation

# Agent works completely autonomously
reasoner = BulletPlanReasoner(
    jentic=jentic_client,
    memory=memory,
    llm=llm,
    escalation=NoEscalation()  # No human help available
)
```

## Escalation Flow Examples

### Scenario 1: Goal Clarification
```
User: "Help me with the project"
Agent: HUMAN: The goal "help me with the project" is too vague. Can you specify:
       - Which project?
       - What kind of help do you need?
       - What's the desired outcome?
Human: "I need to deploy the web application to staging for testing"
Agent: [Creates specific plan for staging deployment]
```

### Scenario 2: Tool Selection Help
```
Agent: [Searches for "Discord message tools"]
Agent: HUMAN: I found 3 Discord tools:
       1. discord-webhook-sender
       2. discord-bot-messenger  
       3. discord-channel-poster
       Which one should I use for sending a simple status update?
Human: "Use discord-webhook-sender, it's the simplest for basic messages"
Agent: [Selects discord-webhook-sender and continues]
```

### Scenario 3: Parameter Assistance
```
Agent: [Needs to send Discord message]
Agent: NEED_HUMAN_INPUT: channel_id, webhook_url
Human: {"channel_id": "123456789", "webhook_url": "https://discord.com/api/webhooks/..."}
Agent: [Executes tool with provided parameters]
```

### Scenario 4: Error Recovery
```
Agent: [Tool execution fails with "Invalid channel_id"]
Agent: The tool execution failed with what appears to be a recoverable error:
       Invalid channel_id
       
       Can you help resolve this? You can:
       1. Provide corrected parameter values
       2. Suggest alternative approach
       3. Tell me to skip this step
Human: {"channel_id": "987654321"}
Agent: [Retries with corrected channel_id and succeeds]
```

### Scenario 5: Proactive Guidance
```
Agent: [After 2 iterations with some failures]
Agent: HUMAN: I've had a couple of failures trying to connect to the database.
       Should I continue with the current approach or try a different method?
Human: "Try using the backup database connection string"
Agent: [Updates approach based on guidance]
```

## Benefits

### For Agents
- **Reduced frustration**: Don't struggle with unclear requirements
- **Better success rates**: Get help before failures occur
- **Learning opportunity**: Incorporate human expertise
- **Confidence**: Know help is available when needed

### For Users  
- **Better outcomes**: Agent asks for clarification rather than guessing
- **Control**: Maintain oversight without constant monitoring
- **Efficiency**: Focused assistance only when needed
- **Transparency**: Clear visibility into agent decision points

### For Systems
- **Reliability**: Graceful handling of ambiguous situations
- **Adaptability**: Dynamic response to various scenarios
- **Scalability**: Works with any escalation interface
- **Maintainability**: Clean separation of autonomous vs. assisted operation

## Configuration Options

### Escalation Frequency
Control how often the agent checks for proactive guidance:
```python
# Override in subclass
def _should_check_for_human_guidance(self, state, iteration):
    return iteration % 5 == 0  # Every 5 iterations instead of 3
```

### Escalation Frequency
Control how often the agent gets the opportunity to proactively check for guidance:
```python
# Override in subclass
def _should_check_for_human_guidance(self, state, iteration):
    return iteration % 6 == 0  # Every 6 iterations instead of 4
```

### Escalation Patterns
Add custom escalation patterns:
```python
escalation_patterns = [
    r"^HUMAN:\s*(.+)",
    r"^CUSTOM_HELP:\s*(.+)",  # Your custom pattern
    r"^NEED_GUIDANCE:\s*(.+)"
]
```

## Best Practices

### For Agent Prompts
1. **Be specific**: "I need the Discord channel ID for notifications" vs "I need help"
2. **Provide options**: Suggest multiple approaches when possible
3. **Include context**: Always explain what you're trying to accomplish
4. **Be actionable**: Ask for specific information or decisions

### For Human Responses
1. **Be precise**: Provide exact values, not approximate guidance
2. **Include rationale**: Explain why you recommend a particular approach
3. **Use structured formats**: JSON for parameters, clear instructions for guidance
4. **Consider alternatives**: Suggest backup plans when appropriate

### For System Design
1. **Graceful degradation**: Always have fallback behavior
2. **Context preservation**: Maintain conversation history
3. **Clear interfaces**: Simple, consistent escalation patterns
4. **Flexible backends**: Support multiple escalation channels

## Comparison with Traditional HITL

| Aspect | Traditional HITL | Dynamic Escalation |
|--------|------------------|-------------------|
| **Trigger** | Automatic (failures, timeouts) | Agent choice |
| **Timing** | Reactive (after problems) | Proactive + reactive |
| **Coverage** | Limited touchpoints | Universal coverage |
| **Context** | Often limited | Rich, phase-specific |
| **Agent autonomy** | Reduced | Preserved |
| **Human interruption** | Frequent | Only when needed |
| **Complexity** | High infrastructure | Simple interface |
| **Adaptability** | Fixed rules | Dynamic decisions |

## Future Enhancements

### Planned Features
- **Learning from escalations**: Track successful patterns
- **Smart escalation timing**: ML-based prediction of when help is needed
- **Multi-human consultation**: Route questions to domain experts
- **Escalation templates**: Pre-formatted requests for common scenarios
- **Async escalation**: Non-blocking human assistance
- **Escalation analytics**: Track patterns and optimize

### Integration Possibilities
- **Slack/Teams bots**: Direct integration with chat platforms
- **Ticketing systems**: Create support tickets for complex issues
- **Expert routing**: Route questions to appropriate specialists
- **Knowledge bases**: Check documentation before escalating
- **Video calls**: Escalate to live conversations for complex issues

The Dynamic Escalation System represents a significant advance in human-AI collaboration, providing the flexibility and intelligence needed for real-world autonomous agent deployment while maintaining the safety net of human expertise when needed. 