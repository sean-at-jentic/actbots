"""
Standard reasoning implementation using ReAct pattern with Jentic SDK integration.
"""

import logging
from typing import Any, Dict, List, Optional
import json

from ..platform.jentic_client import JenticClient
from .base_reasoner import BaseReasoner, ReasoningResult
from ..utils.llm import BaseLLM, LiteLLMChatLLM

logger = logging.getLogger(__name__)


class StandardReasoner(BaseReasoner):
    """
    Concrete implementation of ReAct reasoning loop with Jentic SDK integration.

    Uses OpenAI for reasoning and Jentic platform for tool discovery and execution.
    """

    def __init__(
        self,
        jentic_client: JenticClient,
        llm: Optional[BaseLLM] = None,
        model: str = "gpt-4",
        max_tool_calls_per_iteration: int = 3,
    ):
        """
        Initialize the standard reasoner.

        Args:
            jentic_client: Client for Jentic platform operations
            llm: LLM client for LLM calls (if None, creates default)
            model: OpenAI model to use for reasoning
            max_tool_calls_per_iteration: Max tool calls per reasoning iteration
        """
        self.jentic_client = jentic_client
        self.llm = llm or LiteLLMChatLLM(model=model)
        self.model = model
        self.max_tool_calls_per_iteration = max_tool_calls_per_iteration

    def run(self, goal: str, max_iterations: int = 10) -> ReasoningResult:
        """
        Execute the reasoning loop for a given goal.
        """
        logger.info(
            f"Reasoning started for goal: {goal} | Max iterations: {max_iterations}"
        )

        observations: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        failed_attempts: List[str] = []

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            try:
                context = {
                    "iteration": iteration + 1,
                    "observations": observations,
                    "failed_attempts": failed_attempts,
                    "tool_calls": tool_calls,
                }

                # Plan
                plan = self.plan(goal, context)
                logger.info(f"Plan: {plan}")

                # Check if we can already answer (only if we have observations)
                if observations and self.evaluate(goal, observations):
                    final_answer = self._generate_final_answer(goal, observations)
                    return ReasoningResult(
                        final_answer=final_answer,
                        iterations=iteration + 1,
                        tool_calls=tool_calls,
                        success=True,
                    )

                # Search for tools
                available_tools = self.jentic_client.search(plan, top_k=5)

                # Select tool
                selected_tool = self.select_tool(plan, available_tools)
                if selected_tool:
                    logger.info(f"Tool selected: {selected_tool['id']}")
                else:
                    logger.info("No tool selected for this step.")

                if selected_tool is None:
                    # No tool needed, try to generate answer
                    if observations:
                        final_answer = self._generate_final_answer(goal, observations)
                        return ReasoningResult(
                            final_answer=final_answer,
                            iterations=iteration + 1,
                            tool_calls=tool_calls,
                            success=True,
                        )
                    else:
                        failed_attempts.append(
                            f"No suitable tool found for plan: {plan}"
                        )
                        continue

                # Load tool details
                tool_details = self.jentic_client.load(selected_tool["id"])

                # Act
                action_params = self.act(tool_details, plan)

                # Execute tool
                execution_result = self.jentic_client.execute(
                    selected_tool["id"], action_params
                )

                tool_calls.append(
                    {
                        "tool_id": selected_tool["id"],
                        "tool_name": selected_tool["name"],
                        "params": action_params,
                        "result": execution_result,
                    }
                )

                # Observe
                observation = self.observe(execution_result)
                observations.append(observation)

                logger.info("Step executed and observed.")

            except Exception as e:
                error_msg = f"Error in iteration {iteration + 1}: {str(e)}"
                logger.error(error_msg)
                failed_attempts.append(error_msg)

                # Reflect on failure
                if failed_attempts:
                    reflection = self.reflect(goal, observations, failed_attempts)
                    logger.info("Reflection attempted on failure.")

        # Max iterations reached
        if observations:
            final_answer = self._generate_final_answer(goal, observations)
            success = True
        else:
            final_answer = "I was unable to find a solution within the iteration limit."
            success = False

        return ReasoningResult(
            final_answer=final_answer,
            iterations=max_iterations,
            tool_calls=tool_calls,
            success=success,
            error_message="Max iterations reached" if not success else None,
        )

    def plan(self, goal: str, context: Dict[str, Any]) -> str:
        """Generate a plan for achieving the goal."""
        messages = [
            {
                "role": "system",
                "content": """You are a planning assistant. Given a goal and context, create a clear, actionable plan.
                Keep plans concise and focused on the next immediate step needed.""",
            },
            {
                "role": "user",
                "content": f"""Goal: {goal}
                
Context:
- Iteration: {context.get('iteration', 1)}
- Previous observations: {context.get('observations', [])}
- Failed attempts: {context.get('failed_attempts', [])}

What should be the next step in the plan to achieve this goal?""",
            },
        ]

        response = self.llm.chat(messages=messages, max_tokens=200, temperature=0.7)

        return response.strip()

    def select_tool(
        self, plan: str, available_tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select the most appropriate tool for executing the current plan."""
        if not available_tools:
            return None

        if len(available_tools) == 1:
            return available_tools[0]

        tool_descriptions = "\n".join(
            [
                f"- {tool['id']}: {tool['name']} - {tool.get('description', '')}"
                for tool in available_tools
            ]
        )

        messages = [
            {
                "role": "system",
                "content": """You are a tool selection assistant. Given a plan and available tools, 
                select the most appropriate tool or respond with 'NONE' if no tool is suitable.""",
            },
            {
                "role": "user",
                "content": f"""Plan: {plan}

Available tools:
{tool_descriptions}

Which tool ID should be used, or 'NONE' if no tool is needed?
Respond with just the tool ID or 'NONE'.""",
            },
        ]

        response = self.llm.chat(
            messages=messages, max_tokens=50, temperature=0.0  # deterministic selection
        )

        selected_id = (response or "").strip()
        logger.info(f"LLM tool selection response: '{selected_id}'")

        # If model explicit says NONE
        if selected_id.upper() == "NONE":
            return None

        # Fallback: if model returns empty string, use first candidate
        if selected_id == "":
            logger.warning(
                "LLM returned empty tool id – defaulting to first available tool"
            )
            return available_tools[0]

        # Find the selected tool
        for tool in available_tools:
            if tool["id"] == selected_id:
                logger.info(f"Found matching tool: {tool['id']}")
                return tool

        # If no exact match found, return None (don't fallback)
        logger.warning(f"No tool found matching ID: '{selected_id}'")
        return None

    def act(self, tool: Dict[str, Any], plan: str) -> Dict[str, Any]:
        """Execute an action using the selected tool."""
        tool_params = tool.get("parameters", {})

        if not tool_params:
            return {}

        messages = [
            {
                "role": "system",
                "content": f"""You are an action parameter generator. Given a tool and plan, 
                generate appropriate parameters for the tool. 
                
Tool: {tool['name']}
Description: {tool.get('description', '')}
Parameters: {json.dumps(tool_params, indent=2)}

Respond with a JSON object containing the parameter values.""",
            },
            {
                "role": "user",
                "content": f"""Plan: {plan}

Generate parameters for this tool as a JSON object.""",
            },
        ]

        response = self.llm.chat(messages=messages, max_tokens=200, temperature=0.3)

        try:
            params = json.loads(response.strip())
            return params
        except json.JSONDecodeError:
            logger.warning("Failed to parse action parameters, using empty dict")
            return {}

    def observe(self, action_result: Dict[str, Any]) -> str:
        """Process and interpret the result of an action."""
        if action_result.get("status") == "success":
            return f"Tool executed successfully. Result: {action_result.get('result', 'No result provided')}"
        else:
            return f"Tool execution failed. Error: {action_result.get('error', 'Unknown error')}"

    def evaluate(self, goal: str, observations: List[str]) -> bool:
        """Evaluate whether the goal has been achieved based on observations."""
        if not observations:
            return False

        messages = [
            {
                "role": "system",
                "content": """You are an evaluation assistant. Determine if a goal has been achieved 
                based on the observations. Respond with 'YES' if achieved, 'NO' if not.""",
            },
            {
                "role": "user",
                "content": f"""Goal: {goal}

Observations:
{chr(10).join(f"- {obs}" for obs in observations)}

Has the goal been achieved? Respond with YES or NO only.""",
            },
        ]

        response = self.llm.chat(messages=messages, max_tokens=10, temperature=0.1)

        return response.strip().upper() == "YES"

    def reflect(
        self, goal: str, observations: List[str], failed_attempts: List[str]
    ) -> str:
        """Reflect on failures and generate improved strategies."""
        messages = [
            {
                "role": "system",
                "content": """You are a reflection assistant. Analyze failures and suggest improvements 
                for achieving the goal.""",
            },
            {
                "role": "user",
                "content": f"""Goal: {goal}

Observations so far:
{chr(10).join(f"- {obs}" for obs in observations)}

Failed attempts:
{chr(10).join(f"- {attempt}" for attempt in failed_attempts)}

What insights can help improve the approach?""",
            },
        ]

        response = self.llm.chat(messages=messages, max_tokens=200, temperature=0.7)

        return response.strip()

    def _generate_final_answer(self, goal: str, observations: List[str]) -> str:
        """Generate final answer based on goal and observations."""
        messages = [
            {
                "role": "system",
                "content": """You are a final answer generator. Based on the goal and observations, 
                provide a clear, comprehensive answer.""",
            },
            {
                "role": "user",
                "content": f"""Goal: {goal}

Observations:
{chr(10).join(f"- {obs}" for obs in observations)}

Provide a final answer to the goal based on these observations.""",
            },
        ]

        response = self.llm.chat(messages=messages, max_tokens=300, temperature=0.5)

        return response.strip()
