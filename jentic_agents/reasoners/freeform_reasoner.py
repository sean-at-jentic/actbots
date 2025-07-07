"""FreeformReasoner - A minimal, conversational reasoning loop.

This reasoner gives the LLM maximum autonomy by:
1. Providing the full tool catalogue upfront
2. Allowing inline tool invocation via structured annotations
3. Maintaining a single conversational context
4. Minimal interruption - only stepping in for tool execution and safety
5. Emergent planning rather than forced structure

The LLM controls its own flow: thinking, planning, acting, and self-correcting
all happen in a natural conversation with embedded tool calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_reasoner import BaseReasoner, ReasoningResult
from ..platform.jentic_client import JenticClient
from ..utils.llm import BaseLLM, LiteLLMChatLLM
from ..memory.scratch_pad import ScratchPadMemory
from ..utils.logger import get_logger
from ..communication.hitl.base_intervention_hub import BaseInterventionHub, NoEscalation

logger = get_logger(__name__)

# Safety limits
MAX_ITERATIONS = 50
MAX_TOOL_CALLS_PER_TURN = 5
MAX_CONTEXT_TOKENS = 100000  # Rough estimate for context management


@dataclass
class ConversationState:
    """Tracks the ongoing conversation and execution state."""

    goal: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None
    error_message: Optional[str] = None


class FreeformReasoner(BaseReasoner):
    """A minimal reasoner that lets the LLM drive the entire process."""

    def __init__(
        self,
        jentic: JenticClient,
        memory: ScratchPadMemory,
        llm: Optional[BaseLLM] = None,
        model: str = "gpt-4o",
        max_iterations: int = MAX_ITERATIONS,
        include_tool_catalogue: bool = True,
        intervention_hub: Optional[BaseInterventionHub] = None,
    ) -> None:
        """Initialize the freeform reasoner.

        Args:
            jentic: Client for tool search and execution
            memory: Memory system for state persistence
            llm: Language model interface
            model: Model name if no LLM provided
            max_iterations: Safety limit on conversation turns
            include_tool_catalogue: Whether to provide full tool list upfront
            intervention_hub: Human intervention hub for escalations
        """
        super().__init__()
        self.jentic = jentic
        self.memory = memory
        self.llm = llm or LiteLLMChatLLM(model=model)
        self.max_iterations = max_iterations
        self.include_tool_catalogue = include_tool_catalogue
        self.intervention_hub = intervention_hub or NoEscalation()

        logger.info(
            f"Initialized FreeformReasoner with model={model}, max_iterations={max_iterations}"
        )

    def run(self, goal: str, max_iterations: Optional[int] = None) -> ReasoningResult:
        """Execute the conversational reasoning loop."""
        logger.info("=== STARTING FREEFORM REASONING ===")
        logger.info(f"Goal: {goal}")

        max_iters = max_iterations or self.max_iterations
        state = ConversationState(goal=goal)

        # Initialize conversation with system prompt and goal
        self._initialize_conversation(state)

        while state.iteration_count < max_iters and not state.is_complete:
            logger.info(f"=== ITERATION {state.iteration_count + 1}/{max_iters} ===")

            try:
                # Get LLM response
                response = self._get_llm_response(state)
                logger.info(f"LLM response: {response[:200]}...")

                # Check for completion signals
                if self._check_completion(response, state):
                    logger.info("Completion detected!")
                    break

                # Extract and execute any tool calls
                tool_results = self._execute_embedded_tools(response, state)

                # Add LLM response to conversation
                state.messages.append({"role": "assistant", "content": response})

                # Add tool results to conversation if any
                if tool_results:
                    results_text = self._format_tool_results(tool_results)
                    state.messages.append(
                        {"role": "user", "content": f"Tool results:\n{results_text}"}
                    )

                state.iteration_count += 1

            except Exception as e:
                logger.error(f"Error in iteration {state.iteration_count + 1}: {e}")
                state.error_message = str(e)
                break

        # Finalize result
        return self._create_result(state, max_iters)

    def _initialize_conversation(self, state: ConversationState) -> None:
        """Set up the initial conversation context."""
        logger.info("Initializing conversation context")

        # Build system prompt
        system_prompt = self._build_system_prompt()
        state.messages.append({"role": "system", "content": system_prompt})

        # Add goal as first user message
        goal_prompt = f"""Goal: {state.goal}

Please work toward achieving this goal. You can:
- Think through the problem step by step
- Use tools by embedding calls like <tool name="tool_id">{{json_args}}</tool>
- Access and store information in memory
- Self-correct if something goes wrong
- Announce completion with "TASK COMPLETE:" when done

Begin working on this goal now."""

        state.messages.append({"role": "user", "content": goal_prompt})
        logger.info("Conversation initialized")

    def _build_system_prompt(self) -> str:
        """Create the system prompt with tool catalogue and instructions."""
        logger.info("Building system prompt")

        base_prompt = """You are an autonomous AI assistant with access to a suite of tools and a memory system.

CAPABILITIES:
- Use tools by embedding: <tool name="tool_id">{"param": "value"}</tool>
- Access memory with: <memory_get key="key_name"/>
- Store in memory with: <memory_set key="key_name" value="value" description="optional"/>
- Think freely - no rigid planning required
- Self-correct and adapt your approach

TOOL CALL FORMAT:
<tool name="exact_tool_id">
{"parameter1": "value1", "parameter2": "value2"}
</tool>

TOOL SEARCH:
If you don't see the tool you need in the available tools list, search for more:
<tool name="jentic_tool_search">
{"query": "keywords describing what you need", "top_k": 10}
</tool>

MEMORY FORMAT:
<memory_get key="some_key"/>
<memory_set key="result_data" value="the actual data" description="What this data represents"/>

HUMAN ESCALATION:
If you're stuck, missing critical information, or need human input, use:
<escalate_to_human reason="why you need help" question="specific question for human"/>

COMPLETION:
When you've achieved the goal, respond with: "TASK COMPLETE: [brief summary]"

IMPORTANT:
- Be autonomous - don't ask permission, just act
- Use tools liberally to gather information and take actions
- If you don't see a needed tool, search for it with jentic_tool_search first
- Store important results in memory for later use
- Think out loud about your reasoning
- If something fails, try a different approach
- Only escalate to human as a last resort after searching for tools"""

        # Add tool catalogue if requested
        if self.include_tool_catalogue:
            tool_catalogue = self._get_tool_catalogue()
            if tool_catalogue:
                base_prompt += f"\n\nAVAILABLE TOOLS:\n{tool_catalogue}"

        # Add current memory state
        memory_summary = self._get_memory_summary()
        if memory_summary:
            base_prompt += f"\n\nCURRENT MEMORY:\n{memory_summary}"

        return base_prompt

    def _get_tool_catalogue(self) -> str:
        """Get a formatted catalogue of available tools."""
        logger.info("Fetching tool catalogue")
        try:
            # Search for all tools with a broad query
            tools = self.jentic.search("", top_k=50)  # Get many tools

            if not tools:
                return "No tools available."

            catalogue_lines = []
            for tool in tools[:20]:  # Limit to top 20 to control prompt size
                if isinstance(tool, dict):
                    name = tool.get("name", tool.get("id", "Unknown"))
                    tool_id = tool.get("id", "Unknown")
                    description = tool.get("description", "No description")
                    api_name = tool.get("api_name", "")
                else:
                    name = getattr(tool, "name", "Unknown")
                    tool_id = getattr(tool, "id", "Unknown")
                    description = getattr(tool, "description", "No description")
                    api_name = getattr(tool, "api_name", "")

                display_name = f"{name} ({api_name})" if api_name else name
                catalogue_lines.append(f"- {tool_id}: {display_name} - {description}")

            return "\n".join(catalogue_lines)

        except Exception as e:
            logger.warning(f"Failed to get tool catalogue: {e}")
            return "Tool catalogue unavailable."

    def _get_memory_summary(self) -> str:
        """Get a summary of current memory contents."""
        try:
            if hasattr(self.memory, "keys") and hasattr(self.memory, "retrieve"):
                keys = list(self.memory.keys())
                if not keys:
                    return "Memory is empty."

                summary_lines = []
                for key in keys[:10]:  # Limit to avoid prompt bloat
                    try:
                        value = self.memory.retrieve(key)
                        if isinstance(value, str) and len(value) > 100:
                            preview = value[:100] + "..."
                        else:
                            preview = str(value)
                        summary_lines.append(f"- {key}: {preview}")
                    except Exception:
                        summary_lines.append(f"- {key}: <unavailable>")

                return "\n".join(summary_lines)
        except Exception as e:
            logger.debug(f"Could not get memory summary: {e}")

        return "Memory status unknown."

    def _get_llm_response(self, state: ConversationState) -> str:
        """Get the next response from the LLM."""
        logger.info("Getting LLM response")

        # Manage context length by truncating old messages if needed
        messages = self._manage_context_length(state.messages)

        try:
            response = self.llm.chat(messages=messages)
            logger.debug(f"Raw LLM response: {response}")
            return response.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM communication failed: {e}")

    def _manage_context_length(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Manage context length by keeping system + recent messages."""
        if len(messages) <= 10:  # Small conversation, keep all
            return messages

        # Keep system message + last 8 messages
        system_msg = messages[0] if messages[0]["role"] == "system" else None
        recent_messages = messages[-8:]

        if system_msg and recent_messages[0] != system_msg:
            return [system_msg] + recent_messages
        return recent_messages

    def _check_completion(self, response: str, state: ConversationState) -> bool:
        """Check if the response indicates task completion."""
        completion_patterns = [
            r"TASK COMPLETE:",
            r"GOAL ACHIEVED:",
            r"COMPLETED:",
            r"FINISHED:",
            r"DONE:",
        ]

        for pattern in completion_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                logger.info(f"Completion pattern found: {pattern}")
                state.is_complete = True
                state.final_answer = response
                return True

        return False

    def _execute_embedded_tools(
        self, response: str, state: ConversationState
    ) -> List[Dict[str, Any]]:
        """Extract and execute tool calls embedded in the response."""
        logger.info("Extracting embedded tool calls")

        tool_results = []

        # Extract tool calls
        tool_calls = self._extract_tool_calls(response)
        logger.info(f"Found {len(tool_calls)} tool calls")

        # Extract memory operations
        memory_ops = self._extract_memory_operations(response)
        logger.info(f"Found {len(memory_ops)} memory operations")

        # Check for human escalation requests
        escalation = self._extract_escalation_request(response)
        if escalation:
            logger.info(f"Human escalation requested: {escalation['reason']}")
            return self._handle_human_escalation(escalation, state)

        # Safety check
        if len(tool_calls) > MAX_TOOL_CALLS_PER_TURN:
            logger.warning(
                f"Too many tool calls ({len(tool_calls)}), limiting to {MAX_TOOL_CALLS_PER_TURN}"
            )
            tool_calls = tool_calls[:MAX_TOOL_CALLS_PER_TURN]

        # Execute tool calls
        for tool_call in tool_calls:
            try:
                result = self._execute_single_tool(tool_call, state)
                tool_results.append(result)
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                tool_results.append(
                    {"tool_id": tool_call["tool_id"], "error": str(e), "success": False}
                )

        # Execute memory operations
        for mem_op in memory_ops:
            try:
                self._execute_memory_operation(mem_op)
            except Exception as e:
                logger.error(f"Memory operation failed: {e}")

        return tool_results

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from <tool name="id">json</tool> tags."""
        tool_calls = []

        # Pattern to match <tool name="tool_id">json_content</tool>
        pattern = r'<tool\s+name="([^"]+)"\s*>(.*?)</tool>'
        matches = re.findall(pattern, text, re.DOTALL)

        for tool_id, json_content in matches:
            try:
                args = json.loads(json_content.strip())
                tool_calls.append({"tool_id": tool_id, "args": args})
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in tool call for {tool_id}: {e}")
                tool_calls.append(
                    {"tool_id": tool_id, "args": {}, "parse_error": str(e)}
                )

        return tool_calls

    def _extract_memory_operations(self, text: str) -> List[Dict[str, Any]]:
        """Extract memory operations from the response."""
        operations = []

        # Extract memory_get operations
        get_pattern = r'<memory_get\s+key="([^"]+)"\s*/>'
        get_matches = re.findall(get_pattern, text)
        for key in get_matches:
            operations.append({"type": "get", "key": key})

        # Extract memory_set operations
        set_pattern = r'<memory_set\s+key="([^"]+)"\s+value="([^"]+)"(?:\s+description="([^"]+)")?\s*/>'
        set_matches = re.findall(set_pattern, text)
        for key, value, description in set_matches:
            operations.append(
                {
                    "type": "set",
                    "key": key,
                    "value": value,
                    "description": description or "Set via freeform reasoning",
                }
            )

        return operations

    def _extract_escalation_request(self, text: str) -> Optional[Dict[str, str]]:
        """Extract human escalation request from the response."""
        pattern = r'<escalate_to_human\s+reason="([^"]+)"\s+question="([^"]+)"\s*/>'
        match = re.search(pattern, text)

        if match:
            return {"reason": match.group(1), "question": match.group(2)}
        return None

    def _handle_human_escalation(
        self, escalation: Dict[str, str], state: ConversationState
    ) -> List[Dict[str, Any]]:
        """Handle a request for human escalation."""
        reason = escalation["reason"]
        question = escalation["question"]

        logger.info("=== HUMAN ESCALATION REQUESTED ===")
        logger.info(f"Reason: {reason}")
        logger.info(f"Question: {question}")

        # Check if human intervention is available
        if not self.intervention_hub.is_available():
            logger.warning("Human intervention not available")
            return [
                {
                    "tool_id": "human_escalation",
                    "result": {
                        "human_response": "Human intervention not available - please continue with best effort.",
                        "question_asked": question,
                        "reason": reason,
                    },
                    "success": False,
                }
            ]

        # Build context for human
        context_parts = [
            f"Current goal: {state.goal}",
            f"Reason for escalation: {reason}",
        ]

        # Add recent conversation context
        if len(state.messages) > 2:
            context_parts.append("Recent context:")
            for msg in state.messages[-3:]:
                role = msg["role"].upper()
                content = (
                    msg["content"][:200] + "..."
                    if len(msg["content"]) > 200
                    else msg["content"]
                )
                context_parts.append(f"{role}: {content}")

        context = "\n".join(context_parts)

        # Use the intervention hub to ask for help
        try:
            human_response = self.intervention_hub.ask_human(question, context)
            logger.info(f"Human provided response: {human_response}")

            # Return the human response as a "tool result" to continue the conversation
            # The LLM will interpret and use the response as needed
            return [
                {
                    "tool_id": "human_escalation",
                    "result": human_response,  # Direct response, no parsing
                    "success": True,
                }
            ]

        except Exception as e:
            logger.error(f"Escalation handler failed: {e}")
            return [
                {
                    "tool_id": "human_escalation",
                    "result": {
                        "human_response": "Human escalation failed - please continue with best effort.",
                        "question_asked": question,
                        "reason": reason,
                        "error": str(e),
                    },
                    "success": False,
                }
            ]

    def _handle_tool_search(
        self, args: Dict[str, Any], state: ConversationState
    ) -> Dict[str, Any]:
        """Handle the special jentic_tool_search pseudo-tool."""
        query = args.get("query", "")
        top_k = args.get("top_k", 10)

        logger.info(f"Executing tool search for query: '{query}' (top_k={top_k})")

        try:
            # Search for tools using the provided query
            tools = self.jentic.search(query, top_k=top_k)

            if not tools:
                return {
                    "tool_id": "jentic_tool_search",
                    "result": f"No tools found for query: '{query}'",
                    "success": True,
                }

            # Format the search results
            tool_lines = []
            for i, tool in enumerate(tools, 1):
                if isinstance(tool, dict):
                    name = tool.get("name", tool.get("id", "Unknown"))
                    tool_id = tool.get("id", "Unknown")
                    description = tool.get("description", "No description")
                    api_name = tool.get("api_name", "")
                else:
                    name = getattr(tool, "name", "Unknown")
                    tool_id = getattr(tool, "id", "Unknown")
                    description = getattr(tool, "description", "No description")
                    api_name = getattr(tool, "api_name", "")

                display_name = f"{name} ({api_name})" if api_name else name
                tool_lines.append(f"{i}. {tool_id}: {display_name} - {description}")

            search_results = "\n".join(tool_lines)

            # Record the tool search in the call history
            call_record = {
                "tool_id": "jentic_tool_search",
                "args": args,
                "result": f"Found {len(tools)} tools",
                "iteration": state.iteration_count,
            }
            state.tool_calls.append(call_record)

            return {
                "tool_id": "jentic_tool_search",
                "result": f"Found {len(tools)} tools matching '{query}':\n{search_results}",
                "success": True,
            }

        except Exception as e:
            logger.error(f"Tool search failed: {e}")
            return {"tool_id": "jentic_tool_search", "error": str(e), "success": False}

    def _execute_single_tool(
        self, tool_call: Dict[str, Any], state: ConversationState
    ) -> Dict[str, Any]:
        """Execute a single tool call."""
        tool_id = tool_call["tool_id"]
        args = tool_call.get("args", {})

        logger.info(f"Executing tool: {tool_id} with args: {args}")

        if "parse_error" in tool_call:
            return {
                "tool_id": tool_id,
                "error": f"JSON parse error: {tool_call['parse_error']}",
                "success": False,
            }

        # Handle special jentic_tool_search pseudo-tool
        if tool_id == "jentic_tool_search":
            return self._handle_tool_search(args, state)

        try:
            # Resolve memory placeholders in arguments
            resolved_args = self.memory.resolve_placeholders(args)

            # Execute the tool
            result = self.jentic.execute(tool_id, resolved_args)

            # Record the tool call
            call_record = {
                "tool_id": tool_id,
                "args": resolved_args,
                "result": result,
                "iteration": state.iteration_count,
            }
            state.tool_calls.append(call_record)

            return {"tool_id": tool_id, "result": result, "success": True}

        except Exception as e:
            logger.error(f"Tool execution failed for {tool_id}: {e}")
            return {"tool_id": tool_id, "error": str(e), "success": False}

    def _execute_memory_operation(self, operation: Dict[str, Any]) -> None:
        """Execute a memory get or set operation."""
        op_type = operation["type"]

        if op_type == "get":
            key = operation["key"]
            try:
                value = self.memory.retrieve(key)
                logger.info(f"Memory get: {key} -> {value}")
            except KeyError:
                logger.warning(f"Memory key not found: {key}")

        elif op_type == "set":
            key = operation["key"]
            value = operation["value"]
            description = operation.get("description", "")

            try:
                self.memory.set(key, value, description)
                logger.info(f"Memory set: {key} = {value}")
            except Exception as e:
                logger.error(f"Memory set failed: {e}")

    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """Format tool execution results for the conversation."""
        formatted_lines = []

        for result in results:
            tool_id = result["tool_id"]
            success = result.get("success", False)

            if success:
                tool_result = result.get("result")
                formatted_lines.append(f"✓ {tool_id}: {tool_result}")
            else:
                error = result.get("error", "Unknown error")
                formatted_lines.append(f"✗ {tool_id}: Error - {error}")

        return "\n".join(formatted_lines)

    def _create_result(
        self, state: ConversationState, max_iters: int
    ) -> ReasoningResult:
        """Create the final reasoning result."""
        logger.info("Creating final reasoning result")

        if state.is_complete:
            success = True
            final_answer = state.final_answer or "Task completed successfully."
            error_message = None
        elif state.error_message:
            success = False
            final_answer = "Task failed due to error."
            error_message = state.error_message
        else:
            success = False
            final_answer = "Task incomplete - reached iteration limit."
            error_message = f"Reached maximum iterations ({max_iters})"

        result = ReasoningResult(
            final_answer=final_answer,
            iterations=state.iteration_count,
            tool_calls=state.tool_calls,
            success=success,
            error_message=error_message,
        )

        logger.info(
            f"Final result: success={success}, iterations={state.iteration_count}, tool_calls={len(state.tool_calls)}"
        )
        return result

    # BaseReasoner interface implementation (simplified for freeform approach)
    def _init_state(self, goal: str, context: Dict[str, Any]) -> Any:
        """Initialize state - not used in freeform approach."""
        return ConversationState(goal=goal)

    def plan(self, state: Any) -> Any:
        """Planning happens implicitly in conversation."""
        pass

    def select_tool(self, plan_step: Any, state: Any) -> str:
        """Tool selection happens inline in conversation."""
        return ""

    def act(self, tool_id: str, state: Any) -> Any:
        """Action execution happens inline in conversation."""
        pass

    def observe(self, observation: Any, state: Any) -> Any:
        """Observation happens inline in conversation."""
        return state

    def evaluate(self, state: Any) -> bool:
        """Evaluation happens inline in conversation."""
        return False

    def reflect(self, current_step: Any, err_msg: str) -> bool:
        """Reflection happens implicitly in conversation."""
        return True
