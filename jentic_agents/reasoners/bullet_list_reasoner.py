# bullet_plan_reasoner.py
"""BulletPlanReasoner — a *plan‑first, late‑bind* reasoning loop.

This class implements the **BulletPlan** strategy described in chat:

1. *Plan* — LLM produces a natural‑language indented Markdown bullet list
   of steps (potentially nested). No tools are named at this stage.
2. *Select* — at run‑time, for **each** step the reasoner
   • searches Jentic for suitable tools,
   • offers the top‑k candidates to the LLM,
   • receives an index of the chosen tool (or a request to refine the
     search query).
3. *Act* — loads the chosen tool spec, prompts the LLM for parameters
   (with memory enumeration), executes the tool and stores results.
4. *Observe / Evaluate / Reflect* — passes tool output back to LLM so it
   can mark the step complete, retry, or patch the plan.

The class extends *BaseReasoner* so it can be swapped into any
*BaseAgent* unchanged.

NOTE ▸ For brevity, this file depends on the following external pieces
(which already exist in the repo skeleton):

* `JenticClient` – thin wrapper around `jentic_sdk` with `.search()`,
  `.load()`, `.execute()`.
* `MemoryItem` dataclass and helper `prompt_memory_enumeration()` from the
  earlier discussion.
* A generic `call_llm(messages: list[dict], **kw)` helper that wraps the
  chosen OpenAI/Gemini client.

Where full implementations would be lengthy (e.g. robust Markdown plan
parser, reflection logic) the code inserts *TODO* comments so the
autonomous coding agent can fill them out.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_reasoner import BaseReasoner
from ..platform.jentic_client import JenticClient  # local wrapper, not the raw SDK
from ..utils.llm import BaseLLM, LiteLLMChatLLM
from ..memory.scratch_pad import ScratchPadMemory
from ..utils.logger import get_logger
from .base_reasoner import StepType
from ..communication.hitl.base_intervention_hub import BaseInterventionHub, NoEscalation

# Initialize module logger using the shared logging utility
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Remove automatic retry limits - let the agent choose when to escalate

# ---------------------------------------------------------------------------
# Helper data models
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One bullet‑plan step.

    Only the *raw* natural‑language text is strictly required. Parsing of
    optional directives (e.g. `store_result_as:`) can be layered on via
    regex or a YAML code fence inside the bullet body.
    """

    text: str
    indent: int = 0  # 0 = top‑level, 1 = first sub‑bullet, …
    store_key: Optional[str] = None  # where to stash the result in memory
    goal_context: Optional[str] = None  # extracted goal context from parentheses
    status: str = "pending"  # pending | running | done | failed
    result: Any = None
    tool_id: Optional[str] = None  # chosen Jentic tool
    params: Optional[Dict[str, Any]] = None  # chosen Jentic tool parameters
    reflection_attempts: int = 0  # track how many times we've tried to fix this step


@dataclass
class ReasonerState:
    goal: str
    plan: deque[Step] = field(default_factory=deque)
    history: List[str] = field(default_factory=list)  # raw trace lines
    goal_completed: bool = False  # Track if the main goal has been achieved


# ---------------------------------------------------------------------------
# Markdown bullet‑list parsing helpers
# ---------------------------------------------------------------------------

BULLET_RE = re.compile(r"^(?P<indent>\s*)([-*]|\d+\.)\s+(?P<content>.+)$")


def parse_bullet_plan(markdown: str) -> deque[Step]:
    """Very lenient parser that turns an indented bullet list into Step objects."""
    logger.info(f"Parsing bullet plan from markdown:\n{markdown}")
    steps: List[Step] = []
    for line_num, line in enumerate(markdown.splitlines(), 1):
        if not line.strip():
            continue  # skip blanks
        m = BULLET_RE.match(line)
        if not m:
            logger.debug(f"Line {line_num} doesn't match bullet pattern: {line}")
            continue
        indent_spaces = len(m.group("indent"))
        indent_level = indent_spaces // 2  # assume two‑space indents
        content = m.group("content").strip()

        # Parse goal context from parentheses: "... ( goal: actual goal text )"
        goal_context = None
        goal_match = re.search(r'\(\s*goal:\s*([^)]+)\s*\)', content)
        if goal_match:
            goal_context = goal_match.group(1).strip()
            # Remove the goal context from the main content
            content = re.sub(r'\s*\(\s*goal:[^)]+\s*\)', '', content).strip()
            logger.debug(f"Extracted goal context: {goal_context}")

        # Simple directive detection:  "… -> store: weather"
        store_key = None
        if "->" in content:
            content, directive = [part.strip() for part in content.split("->", 1)]
            if directive.startswith("store:"):
                store_key = directive.split(":", 1)[1].strip()
                logger.debug(f"Found store directive: {store_key}")

        step = Step(text=content, indent=indent_level, store_key=store_key, goal_context=goal_context)
        steps.append(step)
        logger.debug(f"Parsed step: text='{step.text}', goal_context='{step.goal_context}', store_key='{step.store_key}'")

    # ------------------------------------------------------------------
    # Skip container/meta bullets so we only execute leaf actions.
    # A container is detected when the next bullet has a larger indent
    # level than the current one.
    leaf_steps: List[Step] = []
    for idx, step in enumerate(steps):
        next_indent = steps[idx + 1].indent if idx + 1 < len(steps) else step.indent
        if next_indent > step.indent:
            logger.debug(f"Skipping container step: '{step.text}'")
            continue  # don't enqueue parent/meta bullets
        leaf_steps.append(step)

    logger.info(
        f"Parsed {len(leaf_steps)} leaf steps from bullet plan (original {len(steps)})"
    )
    return deque(leaf_steps)


# ---------------------------------------------------------------------------
# BulletPlanReasoner implementation
# ---------------------------------------------------------------------------


class BulletPlanReasoner(BaseReasoner):
    """Concrete Reasoner that follows the BulletPlan strategy."""

    @staticmethod
    def _load_prompt(prompt_name: str) -> str:
        """Load a prompt from the prompts directory."""
        current_dir = Path(__file__).parent.parent
        prompt_path = current_dir / "prompts" / f"{prompt_name}.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise RuntimeError(f"Prompt file not found: {prompt_path}")

    def __init__(
        self,
        jentic: JenticClient,
        memory: ScratchPadMemory,
        llm: Optional[BaseLLM] = None,
        model: str = "gpt-4o",
        max_iters: int = 20,
        search_top_k: int = 15,
        intervention_hub: Optional[BaseInterventionHub] = None,
    ) -> None:
        logger.info(f"Initializing BulletPlanReasoner with model={model}, max_iters={max_iters}, search_top_k={search_top_k}")
        super().__init__()
        self.jentic = jentic
        self.memory = memory
        self.llm = llm or LiteLLMChatLLM(model=model)
        self.max_iters = max_iters
        self.search_top_k = search_top_k
        self.escalation = intervention_hub or NoEscalation()
        logger.info("BulletPlanReasoner initialization complete")

    # ------------------------------------------------------------------
    # Universal escalation helpers
    # ------------------------------------------------------------------

    def _process_llm_response_for_escalation(self, response: str, context: str = "") -> str:
        """
        Check if LLM response contains XML escalation request and handle it.
        
        Returns:
            Processed response (either original or human response if escalation occurred)
        """
        response = response.strip()
        
        # Check for XML escalation pattern (same as FreeformReasoner)
        escalation_pattern = r'<escalate_to_human\s+reason="([^"]+)"\s+question="([^"]+)"\s*/>'
        match = re.search(escalation_pattern, response)
        
        if match:
            reason = match.group(1).strip()
            question = match.group(2).strip()
            logger.info(f"🤖➡️👤 LLM requested escalation: {reason}")
            
            if self.escalation.is_available():
                try:
                    human_response = self.escalation.ask_human(question, context)
                    if human_response.strip():
                        logger.info(f"👤➡️🤖 Human provided response: {human_response}")
                        return human_response
                    else:
                        logger.warning("👤 No response from human, continuing with original")
                except Exception as e:
                    logger.warning(f"Escalation failed: {e}")
            else:
                logger.warning("⚠️ Escalation requested but not available")
            
            # Remove the escalation tag from the response
            return re.sub(escalation_pattern, '', response).strip()
        
        return response

    def _request_human_help(self, question: str, context: str = "") -> str:
        """
        Direct method for requesting human help from anywhere in the code.
        
        Returns:
            Human response or empty string if not available
        """
        logger.info(f"🤖➡️👤 Direct escalation request: {question}")
        
        if self.escalation.is_available():
            try:
                response = self.escalation.ask_human(question, context)
                logger.info(f"👤➡️🤖 Human response received")
                return response
            except Exception as e:
                logger.warning(f"Direct escalation failed: {e}")
        else:
            logger.warning("⚠️ Direct escalation requested but not available")
        
        return ""

    # ------------------------------------------------------------------
    # BaseReasoner hook implementations
    # ------------------------------------------------------------------

    def _init_state(self, goal: str, context: Dict[str, Any]) -> ReasonerState:
        logger.info(f"Initializing state for goal: {goal}")
        logger.debug(f"Context: {context}")
        state = ReasonerState(goal=goal)
        logger.debug(f"Created initial state: {state}")
        return state

    # 1. PLAN -----------------------------------------------------------
    def plan(self, state: ReasonerState):
        logger.info("=== PLAN PHASE ===")
        if not state.plan:  # first call → create plan
            logger.info("No existing plan, generating new plan")
            bullet_plan_template = self._load_prompt("bullet_plan")
            
            # Enhanced planning prompt with confidence guidance
            enhanced_prompt = f"""
{bullet_plan_template.format(goal=state.goal)}

CONFIDENCE GUIDANCE:
- Interpret the goal using reasonable assumptions about what the user likely wants
- For "send a Discord message", assume they want to use Discord's API/webhook system
- For "open Discord", assume they want to send a message via Discord tools, not literally open the app
- Only escalate if the goal has multiple genuinely different interpretations

If you must escalate for clarification, use: <escalate_to_human reason="need clarification" question="your specific question"/>
Otherwise, provide the markdown bullet plan as requested.
"""
            
            messages = [{"role": "user", "content": enhanced_prompt}]
            logger.info("Calling LLM for plan generation")
            response = self.llm.chat(messages=messages)
            logger.info(f"LLM planning response:\n{response}")
            
            # Process for escalation
            context = f"Goal: {state.goal}\nPhase: Planning"
            processed_response = self._process_llm_response_for_escalation(response, context)
            
            # If response changed due to escalation, use the human guidance as the new goal
            if processed_response != response:
                logger.info("Planning was escalated to human, updating goal with guidance")
                state.goal = processed_response
                # Re-run planning with the clarified goal
                return self.plan(state)
            
            logger.info("Extracting fenced code from response")
            plan_markdown = self._extract_fenced_code(processed_response)
            logger.debug(f"Extracted plan markdown:\n{plan_markdown}")
            
            logger.info("Parsing bullet plan")
            state.plan = parse_bullet_plan(plan_markdown)
            state.history.append(f"Plan generated ({len(state.plan)} steps)")
            
            logger.info(f"Generated plan with {len(state.plan)} steps:")
            for i, step in enumerate(state.plan):
                logger.info(f"  Step {i+1}: {step.text}")
                if step.store_key:
                    logger.debug(f"    Store key: {step.store_key}")
        else:
            logger.info(f"Using existing plan with {len(state.plan)} remaining steps")
        
        if state.plan:
            current_step = state.plan[0]
            logger.info(f"Current step to execute: {current_step.text}")
            return current_step
        else:
            logger.warning("No steps in plan!")
            return None

    # 2. SELECT TOOL ----------------------------------------------------
    def select_tool(self, plan_step: Step, state: ReasonerState):
        logger.info("=== TOOL SELECTION PHASE ===")
        logger.info(f"Selecting tool for step: {plan_step.text}")
        logger.debug(f"Step goal_context: {plan_step.goal_context}")
        logger.debug(f"State goal: {state.goal}")
        
        if plan_step.tool_id:
            logger.info(f"Step already has tool_id: {plan_step.tool_id}")
            return plan_step.tool_id

        # Get AI to extract better search keywords from the goal
        search_query = self._extract_search_keywords_with_ai(plan_step, state)
        
        logger.info(f"Using generated search query: {search_query}")

        # Search Jentic by enhanced NL description
        logger.info(f"Searching Jentic for tools matching: {search_query}")
        hits = self.jentic.search(search_query, top_k=self.search_top_k)
        logger.info(f"Jentic search returned {len(hits)} results")
        
        if not hits:
            logger.error(f"No tools found for search query: {search_query}")
            
            # Enhanced escalation for tool selection
            if self.escalation.is_available():
                question = f"""
No tools were found for the step: "{plan_step.text}"
Search query used: "{search_query}"

This could mean:
1. The search query needs to be refined
2. We need to use a different approach
3. The required tool might not exist

How should I proceed? You can:
- Suggest a better search query
- Recommend a different approach  
- Provide specific tool guidance
- Tell me to skip this step
"""
                context = f"Step: {plan_step.text}\nSearch query: {search_query}\nGoal: {state.goal}"
                
                try:
                    human_response = self._request_human_help(question, context)
                    if human_response.strip():
                        # Check if human provided a search query
                        if "search:" in human_response.lower():
                            new_query = human_response.split(":", 1)[1].strip()
                            logger.info(f"Human provided new search query: {new_query}")
                            hits = self.jentic.search(new_query, top_k=self.search_top_k)
                            if hits:
                                logger.info(f"Human-guided search found {len(hits)} tools")
                            else:
                                raise RuntimeError(f"Human-guided search also found no tools")
                        elif "skip" in human_response.lower():
                            logger.info("Human advised to skip this step")
                            raise RuntimeError("Human advised to skip step")
                        else:
                            # Treat response as a new step description
                            logger.info("Human provided alternative approach")
                            plan_step.text = human_response
                            return self.select_tool(plan_step, state)
                    else:
                        raise RuntimeError(f"No tool found for step: {plan_step.text}")
                except Exception as e:
                    logger.error(f"Tool selection escalation failed: {e}")
                    raise RuntimeError(f"No tool found for step: {plan_step.text}")
            else:
                raise RuntimeError(f"No tool found for step: {plan_step.text}")

        logger.info("Found tool candidates:")
        tool_lines_list = []
        for i, h in enumerate(hits):
            if isinstance(h, dict):
                name = h.get('name', h.get('id', 'Unknown'))
                api_name = h.get('api_name')
                description = h.get('description', '')
                hit_id = h.get('id', 'Unknown')
            else:
                name = getattr(h, 'name', 'Unknown')
                api_name = getattr(h, 'api_name', None)
                description = getattr(h, 'description', '')
                hit_id = getattr(h, 'id', 'Unknown')

            display_name = f"{name} ({api_name})" if api_name else name
            logger.info(f"  {i+1}. {display_name} (ID: {hit_id}) - {description}")
            tool_lines_list.append(f"{i+1}. {display_name} — {description}")
        
        tool_lines = "\n".join(tool_lines_list)
        
        # Include goal context in the selection prompt for better decision making
        goal_info = ""
        if plan_step.goal_context:
            goal_info = f"\nGoal context: {plan_step.goal_context}"
        elif state.goal:
            goal_info = f"\nOverall goal: {state.goal}"
        
        select_tool_template = self._load_prompt("select_tool")
        
        # Enhanced tool selection with confidence guidance
        enhanced_select_prompt = f"""
{select_tool_template.format(
    plan_step_text=plan_step.text,
    goal_info=goal_info,
    tool_lines=tool_lines
)}

SELECTION GUIDANCE:
- Choose the tool that best matches the task requirements
- If multiple tools could work, pick the one that seems most suitable
- Only escalate if you genuinely cannot determine which tool is appropriate
- Make your choice decisively based on the tool descriptions

If you must escalate, use: <escalate_to_human reason="tool selection uncertainty" question="your specific question"/>
Otherwise, respond with just the number of your chosen tool.
"""
        
        logger.debug(f"Tool selection prompt:\n{enhanced_select_prompt}")
        
        messages = [{"role": "user", "content": enhanced_select_prompt}]
        logger.info("Calling LLM for tool selection")
        reply = self.llm.chat(messages=messages).strip()
        logger.info(f"LLM tool selection response: '{reply}'")

        # Process for escalation
        context = f"Step: {plan_step.text}\nAvailable tools: {len(hits)}\nGoal: {state.goal}"
        processed_reply = self._process_llm_response_for_escalation(reply, context)
        
        if processed_reply != reply:
            # Human provided guidance, use it as new selection criteria
            logger.info("Tool selection escalated, processing human guidance")
            # Try to match human guidance to available tools or search again
            for i, h in enumerate(hits):
                tool_name = h.get('name', '') if isinstance(h, dict) else getattr(h, 'name', '')
                if processed_reply.lower() in tool_name.lower():
                    selected_hit = hits[i]
                    tool_id = selected_hit.get('id') if isinstance(selected_hit, dict) else getattr(selected_hit, 'id', None)
                    logger.info(f"Human guidance matched tool: {tool_name} (ID: {tool_id})")
                    plan_step.tool_id = tool_id
                    return tool_id
            
            # If no match, try searching with human guidance
            try:
                new_hits = self.jentic.search(processed_reply, top_k=self.search_top_k)
                if new_hits:
                    selected_hit = new_hits[0]  # Use first result
                    tool_id = selected_hit.get('id') if isinstance(selected_hit, dict) else getattr(selected_hit, 'id', None)
                    tool_name = selected_hit.get('name', tool_id) if isinstance(selected_hit, dict) else getattr(selected_hit, 'name', tool_id)
                    logger.info(f"Human guidance found new tool: {tool_name} (ID: {tool_id})")
                    plan_step.tool_id = tool_id
                    return tool_id
            except Exception as e:
                logger.warning(f"Could not search with human guidance: {e}")
            
            # Fallback to original selection logic
            reply = processed_reply

        # Detect a "no suitable tool" reply signalled by leading 0 (e.g. "0", "0.", "0 -", etc.)
        if re.match(r"^\s*0\D?", reply):
            logger.warning("LLM couldn't find a suitable tool")
            raise RuntimeError("LLM couldn't find a suitable tool.")

        try:
            # Robustly extract the *first* integer that appears in the reply, e.g.
            # "3. inspect-request-data …" → 3
            # "Option 2: foo" → 2
            # "0" → 0
            # Handle verbose responses by looking for various patterns
            
            # First try to find a boxed answer (common in verbose responses)
            boxed_match = re.search(r'\$\\boxed\{(\d+)\}\$', reply)
            if boxed_match:
                idx = int(boxed_match.group(1)) - 1
                logger.debug(f"Found boxed answer, parsed tool index: {idx}")
            else:
                # Look for "Number: X" pattern (from our prompt)
                number_pattern = re.search(r'Number:\s*(\d+)', reply, re.IGNORECASE)
                if number_pattern:
                    idx = int(number_pattern.group(1)) - 1
                    logger.debug(f"Found 'Number:' pattern, parsed tool index: {idx}")
                else:
                    # Look for "final answer is X" or similar patterns
                    final_answer_match = re.search(r'(?:final answer is|answer is|therefore[,\s]+(?:tool\s+)?|the best match is)[:\s]*(\d+)', reply, re.IGNORECASE)
                    if final_answer_match:
                        idx = int(final_answer_match.group(1)) - 1
                        logger.debug(f"Found final answer pattern, parsed tool index: {idx}")
                    else:
                        # Fallback to finding the first integer
                        m = re.search(r"\d+", reply)
                        if not m:
                            raise ValueError("No leading integer found in LLM reply")
                        idx = int(m.group(0)) - 1
                        logger.debug(f"Used fallback method, parsed tool index: {idx}")
            
            if idx < 0 or idx >= len(hits):
                logger.error(f"Tool index {idx} out of range (0-{len(hits)-1})")
                raise IndexError(f"Tool index out of range")
                
            selected_hit = hits[idx]
            logger.debug(f"Selected hit: {selected_hit}")
            
            tool_id = selected_hit.get('id') if isinstance(selected_hit, dict) else getattr(selected_hit, 'id', None)
            tool_name = selected_hit.get('name', tool_id) if isinstance(selected_hit, dict) else getattr(selected_hit, 'name', tool_id)
            
            logger.info(f"Selected tool: {tool_name} (ID: {tool_id})")
            plan_step.tool_id = tool_id
            return tool_id
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing tool selection reply '{reply}': {e}")
            raise RuntimeError(f"Invalid tool index reply: {reply}")

    def _extract_search_keywords_with_ai(self, plan_step: Step, state: ReasonerState) -> str:
        """Use an LLM to rephrase a technical plan step into a high-quality,
        capability-focused search query for the Jentic tool marketplace."""

        # Combine step text with goal context for a richer prompt
        context_text = plan_step.text
        if plan_step.goal_context:
            context_text += f" (Context: This is part of a larger goal to '{plan_step.goal_context}')"

        keyword_extraction_template = self._load_prompt("keyword_extraction")
        
        # Enhanced keyword extraction with confidence guidance
        enhanced_prompt = f"""
{keyword_extraction_template.format(context_text=context_text)}

GUIDANCE:
- Generate search keywords based on reasonable interpretation of the task
- Focus on the core functionality needed, not literal interpretation
- For "open Discord", search for "Discord messaging" or "Discord API" tools
- For "send message", search for "messaging" or "notification" tools
- Only escalate if the task is genuinely incomprehensible

If you must escalate, use: <escalate_to_human reason="unclear task requirements" question="your specific question"/>
Otherwise, provide the search keywords.
"""

        logger.info("Calling LLM for keyword extraction")
        messages = [{"role": "user", "content": enhanced_prompt}]
        keywords = self.llm.chat(messages=messages).strip()
        
        # Process for escalation
        context = f"Step: {plan_step.text}\nGoal: {state.goal}"
        processed_keywords = self._process_llm_response_for_escalation(keywords, context)

        # Clean up the response, removing potential quotes
        processed_keywords = processed_keywords.strip('"\'')

        logger.info(f"AI extracted keywords: '{processed_keywords}'")
        return processed_keywords

    # 3. ACT ------------------------------------------------------------
    def act(self, tool_id: str, state: ReasonerState, current_step: Step):
        """
        Generate parameters for and execute a Jentic tool.
        """
        logger.info("=== ACTION PHASE ===")
        logger.info(f"Executing action with tool_id: {tool_id}")
        
        logger.info("Loading tool information from Jentic")
        tool_info = self.jentic.load(tool_id)
        
        # NOTE: Pass only the current step's text to the parameter generation AI.
        # This prevents the LLM from trying to use details from the *history*
        # to find a tool, which are not valid inputs for the tool itself.
        current_step_text = current_step.text
        params = self._generate_params_with_ai(tool_id, state, current_step_text)
        current_step.params = params  # Store params for context

        logger.info(f"Executing tool {tool_id} with args: {params}")
        try:
            result = self.jentic.execute(tool_id, params)
            logger.info(f"Tool execution result: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise RuntimeError(f"Tool execution failed: {e}")

    def _generate_params_with_ai(self, tool_id: str, state: ReasonerState, step_text: str) -> Dict[str, Any]:
        """
        Use AI to generate parameters for a Jentic tool call based on the step description.
        
        Args:
            tool_id: The Jentic tool identifier
            state: Current reasoning state with goal and context
            step_text: Natural language description of what to do
            
        Returns:
            Dictionary of parameters to pass to the tool
        """
        logger.info(f"Generating parameters for tool {tool_id} with step: {step_text}")
        
        # Load the parameter generation prompt
        param_prompt_template = self._load_prompt("param_generation")
        
        # Get tool schema from the loaded tool info
        tool_info = self.jentic.load(tool_id)
        tool_schema = tool_info.get("parameters", {})
        required_params = tool_info.get("required", [])
        
        # Build context from memory and current state
        memory_context = ""
        if hasattr(self.memory, 'get_recent_entries'):
            recent_entries = self.memory.get_recent_entries(limit=5)
            if recent_entries:
                memory_context = "\n".join([f"- {entry}" for entry in recent_entries])

        # NEW: build enumeration of all memory items for the prompt template
        memory_enum = ""
        if hasattr(self.memory, 'enumerate_for_prompt'):
            try:
                memory_enum = self.memory.enumerate_for_prompt()
            except Exception as e:
                logger.warning(f"Failed to enumerate memory for prompt: {e}")
                memory_enum = "(memory enumeration unavailable)"
        else:
            memory_enum = "(memory enumeration not supported)"

        # Enhanced parameter generation with confidence guidance
        enhanced_param_prompt = f"""
{param_prompt_template.format(
    goal=state.goal,
    step=step_text,
    tool_id=tool_id,
    tool_schema=json.dumps(tool_schema, indent=2),
    required_params=", ".join(required_params) if required_params else "None",
    memory_context=memory_context or "No recent context available",
    memory_enum=memory_enum or "(memory empty)"
)}

PARAMETER GUIDANCE:
- Generate parameters confidently based on the goal and step description
- Use reasonable default values when specific details aren't provided
- For messages, use the step text or goal as the message content
- For channels/targets, use generic identifiers that can be corrected later
- Only request human input for truly unknowable values (specific IDs, tokens, credentials)

If you need specific values only the human knows, respond with: NEED_HUMAN_INPUT: [parameter names]
If you need clarification about the task, use: <escalate_to_human reason="parameter clarification needed" question="your specific question"/>
Otherwise, provide the JSON parameters confidently.
"""
        
        logger.debug(f"Parameter generation prompt: {enhanced_param_prompt}")
        
        try:
            # Get AI response
            response = self.llm.chat(messages=[{"role": "user", "content": enhanced_param_prompt}])
            logger.debug(f"AI parameter response: {response}")
            
            response = response.strip()
            
            # Process general escalation first
            context = f"Step: {step_text}\nTool: {tool_id}\nPhase: Parameter Generation"
            processed_response = self._process_llm_response_for_escalation(response, context)
            
            if processed_response != response:
                # Human provided guidance, re-generate parameters with the guidance
                logger.info("Parameter generation escalated, incorporating human guidance")
                guidance_prompt = f"""
Based on this human guidance: "{processed_response}"

Please generate parameters for tool {tool_id} to accomplish: {step_text}

Tool schema: {json.dumps(tool_schema, indent=2)}
Required parameters: {", ".join(required_params) if required_params else "None"}

Return ONLY the JSON object with parameters.
"""
                try:
                    guided_response = self.llm.chat(messages=[{"role": "user", "content": guidance_prompt}])
                    response = guided_response.strip()
                except Exception as e:
                    logger.warning(f"Failed to re-generate with human guidance: {e}")
                    response = processed_response
            
            # Check if LLM is asking for human input upfront
            if response.startswith("NEED_HUMAN_INPUT:"):
                missing_params_str = response[len("NEED_HUMAN_INPUT:"):].strip()
                missing_params = [p.strip() for p in missing_params_str.split(",")]
                logger.info(f"🤖➡️👤 LLM identified missing parameters: {missing_params}")
                
                if self.escalation.is_available():
                    question = (
                        "I need the following information to execute this tool:\n"
                        f"{', '.join(missing_params)}\n\n"
                        "Please provide the values. You can respond with either:\n"
                        "1. A JSON object like: {\"channel_id\": \"123456789\", \"message\": \"hello\"}\n"
                        "2. Simple key: value pairs on separate lines"
                    )
                    context = (
                        f"Step: {step_text}\n"
                        f"Tool: {tool_id}\n"
                        f"Tool Schema: {json.dumps(tool_schema, indent=2)}"
                    )
                    
                    try:
                        human_reply = self.escalation.ask_human(question, context)
                        if human_reply.strip():
                            logger.info(f"👤➡️🤖 Human provided response: {human_reply}")
                            
                            # Let the LLM re-generate parameters with the human guidance
                            guidance_prompt = f"""
Based on this human guidance: "{human_reply}"

Please generate parameters for tool {tool_id} to accomplish: {step_text}

Tool schema: {json.dumps(tool_schema, indent=2)}
Required parameters: {", ".join(required_params) if required_params else "None"}

Return ONLY the JSON object with parameters. Use the human guidance to fill in the correct values.
"""
                            try:
                                guided_response = self.llm.chat(messages=[{"role": "user", "content": guidance_prompt}])
                                response = guided_response.strip()
                                logger.info("✅ LLM re-generated parameters with human guidance")
                            except Exception as e:
                                logger.warning(f"Failed to re-generate with human guidance: {e}")
                                # Fallback: return empty params and let tool execution fail gracefully
                                return {}
                        else:
                            logger.warning("👤 Human provided empty response")
                            return {}
                    except Exception as esc_err:
                        logger.warning(f"HITL escalation failed: {esc_err}")
                        return {}
                else:
                    logger.warning(f"⚠️ LLM needs human input but HITL not available: {missing_params}")
                    # Return minimal params with placeholders as fallback
                    params = {}
                    for param in missing_params:
                        if param in ["message", "content", "text"]:
                            params[param] = step_text
                        else:
                            params[param] = f"MISSING_{param.upper()}"
                    return params
            
            # Parse JSON response (normal case where LLM has all the info)
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            params = json.loads(response)
            logger.info("✅ LLM provided complete parameters without needing human input")
            
            # ------------------------------------------------------------------
            # Legacy fallback: Validate parameters and escalate if we find issues
            # (This should be less common now with the improved prompt)
            # ------------------------------------------------------------------
            missing_params = [p for p in required_params if p not in params]
            if missing_params:
                logger.warning(f"Missing required parameters: {missing_params}")
                # Apply trivial defaults for very common fields
                for param in missing_params:
                    if param in ["message", "content", "text"]:
                        params[param] = step_text  # use step text as a sensible default
                    elif param in ["channel_id", "chat_id"] and "discord" in tool_id.lower():
                        params[param] = "MISSING_CHANNEL_ID"  # obvious placeholder

            # Detect placeholder-style values that still need human input
            def _looks_placeholder(value: Any) -> bool:
                return (
                    isinstance(value, str)
                    and any(tok in value.lower() for tok in (
                        "provide", "missing", "todo", "tbd", "please", "enter", "specify",
                        "your_", "example", "placeholder", "fill", "replace", "here"
                    ))
                )

            unresolved = [p for p in required_params if p not in params]
            unresolved += [k for k, v in params.items() if _looks_placeholder(v)]

            # Human-in-the-loop escalation
            if unresolved and self.escalation.is_available():
                logger.info(f"🤖➡️👤 HITL escalation triggered for unresolved parameters: {unresolved}")
                question = (
                    "I need concrete values for the following parameters before I can execute the tool:\n"
                    f"{', '.join(unresolved)}\n"
                    "Please reply with a JSON object mapping each parameter name to its value."
                )
                context = (
                    f"Step: {step_text}\n"
                    f"Tool ID: {tool_id}\n\nCurrent parameters (placeholders shown as-is):\n"
                    f"{json.dumps(params, indent=2)}"
                )

                try:
                    human_reply = self.escalation.ask_human(question, context)
                    if human_reply.strip():
                        logger.info(f"👤➡️🤖 Human provided response: {human_reply}")
                        
                        # Let the LLM interpret the human response and update parameters
                        update_prompt = f"""
Based on this human response: "{human_reply}"

Current parameters: {json.dumps(params, indent=2)}

Please update the parameters using the human guidance. Return the complete updated parameter object as JSON.

Tool schema: {json.dumps(tool_schema, indent=2)}
"""
                        try:
                            updated_response = self.llm.chat(messages=[{"role": "user", "content": update_prompt}])
                            if updated_response.startswith("```json"):
                                updated_response = updated_response[7:]
                            if updated_response.endswith("```"):
                                updated_response = updated_response[:-3]
                            updated_params = json.loads(updated_response.strip())
                            if isinstance(updated_params, dict):
                                params.update(updated_params)
                                logger.info(f"✅ Updated parameters with LLM interpretation of human input")
                        except Exception as parse_err:
                            logger.warning(f"Failed to parse LLM interpretation of human response: {parse_err}")
                    else:
                        logger.warning("👤 Human provided empty response")
                except Exception as esc_err:
                    logger.warning(f"HITL escalation failed or skipped: {esc_err}")

                # Re-evaluate unresolved params post-escalation
                unresolved = [p for p in required_params if p not in params]
                unresolved += [k for k, v in params.items() if _looks_placeholder(v)]
                if unresolved:
                    logger.warning(f"Still unresolved parameters after HITL: {unresolved}")
                else:
                    logger.info("✅ All parameters resolved after HITL")
            elif unresolved:
                logger.warning(f"⚠️ Unresolved parameters detected but HITL not available: {unresolved}")
            else:
                logger.info("✅ All required parameters are present and valid")
            
            logger.info(f"Generated parameters: {json.dumps(params, indent=2)}")
            return params
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Raw response: {response}")
            # Return minimal parameters
            return {"error": f"Failed to parse parameters: {str(e)}"}
        except Exception as e:
            logger.error(f"Error generating parameters: {e}")
            return {"error": f"Parameter generation failed: {str(e)}"}

    # 4. OBSERVE --------------------------------------------------------
    def observe(self, observation: Any, state: ReasonerState):
        logger.info("=== OBSERVATION PHASE ===")
        logger.info(f"Processing observation: {observation}")
        
        if not state.plan:
            logger.error("No current step to observe - plan is empty!")
            return state
            
        current_step = state.plan[0]
        logger.info(f"Updating step: {current_step.text}")
        
        # Unpack tool results to store only the meaningful, serializable output.
        # This prevents storing non-serializable objects like OperationResult in memory.
        value_to_store = observation
        if isinstance(observation, dict) and "result" in observation:
            result_obj = observation.get("result")
            if hasattr(result_obj, "output"):
                logger.debug("Unpacking tool result object to store its output.")
                value_to_store = result_obj.output

        current_step.result = value_to_store
        current_step.status = "done"
        logger.debug(f"Step status updated to: {current_step.status}")

        if current_step.store_key:
            logger.info(f"Storing result in memory with key: {current_step.store_key}")
            self.memory.set(
                key=current_step.store_key,
                value=value_to_store,
                description=f"Result from step '{current_step.text}'",
            )
            logger.debug(f"Memory updated with key '{current_step.store_key}'")

        history_entry = f"{current_step.text} -> done"
        state.history.append(history_entry)
        logger.debug(f"Added to history: {history_entry}")
        
        # Check if we got a successful API response that created something
        if self._check_successful_creation(observation):
            logger.info("Detected successful creation/completion. Marking goal as complete.")
            state.goal_completed = True
            state.plan.clear()  # Clear remaining steps
        else:
            logger.info("Removing completed step from plan")
            state.plan.popleft()  # advance to next step
            
        logger.info(f"Remaining steps in plan: {len(state.plan)}")
        
        return state

    def _check_successful_creation(self, observation: Any) -> bool:
        """Check if the observation shows we successfully created/completed something."""
        
        if isinstance(observation, dict):
            result = observation.get('result')
            if result and hasattr(result, 'success') and result.success:
                if hasattr(result, 'output') and result.output:
                    output = result.output
                    if isinstance(output, dict):
                        # Look for creation indicators: IDs, timestamps, URLs
                        creation_indicators = ['id', 'message_id', 'timestamp', 'url']
                        found = [key for key in creation_indicators if key in output]
                        if found:
                            logger.info(f"Found creation indicators: {found}")
                            return True
        
        return False

    # 5. EVALUATE -------------------------------------------------------
    def evaluate(self, state: ReasonerState) -> bool:
        logger.info("=== EVALUATION PHASE ===")
        is_complete = not state.plan
        logger.info(f"Plan complete: {is_complete} (remaining steps: {len(state.plan)})")
        
        if is_complete:
            logger.info("All steps completed successfully!")
        else:
            logger.info(f"Next step to execute: {state.plan[0].text if state.plan else 'None'}")
            
        return is_complete

    # 6. REFLECT (optional) --------------------------------------------
    def reflect(self, current_step: Step, err_msg: str) -> bool:
        """
        Analyze a failed step and decide whether to retry, revise, or escalate.
        """
        logger.info("=== REFLECTION PHASE ===")
        logger.info(f"Reflecting on failed step: {current_step.text}")
        logger.info(f"Error message: {err_msg}")
        logger.info(f"Reflection attempts so far: {current_step.reflection_attempts}")
        
        current_step.reflection_attempts += 1
        
        # Enhanced reflection with dynamic escalation
        reflection_prompt = f"""
You are an AI agent reflecting on a failed step. You have several options:

1. Try a different approach: TRY: <revised step description>
2. Ask for human guidance: <escalate_to_human reason="step failure" question="your specific question"/>
3. Skip this step: SKIP

Failed step: {current_step.text}
Error: {err_msg}
Attempts so far: {current_step.reflection_attempts}

What's your choice?"""

        try:
            response = self.llm.chat(messages=[{"role": "user", "content": reflection_prompt}]).strip()
            logger.info(f"Reflection decision: {response}")
            
            # Process escalation first
            context = f"""
Step: {current_step.text}
Tool: {current_step.tool_id}
Params: {json.dumps(current_step.params, indent=2)}
Error: {err_msg}
Attempts: {current_step.reflection_attempts}
Phase: Reflection
"""
            processed_response = self._process_llm_response_for_escalation(response, context)
            
            if processed_response != response:
                # Human provided guidance via escalation
                logger.info("Reflection escalated, using human guidance")
                current_step.text = processed_response
                current_step.status = "pending"
                current_step.tool_id = None
                return True
            
            # Handle traditional reflection responses
            if response.startswith("TRY:"):
                revised_step = response[4:].strip()
                logger.info(f"Agent chose to try again with revised step: {revised_step}")
                current_step.text = revised_step
                current_step.status = "pending"
                current_step.tool_id = None
                return True
                    
            elif response.startswith("SKIP"):
                logger.info("Agent chose to skip this step")
                return False
                
            else:
                # Fallback if response format is unexpected
                logger.warning(f"Unexpected reflection response format: {response}")
                return self._try_fallback_approach(current_step)
                
        except Exception as e:
            logger.error(f"Error during reflection: {e}")
            return self._try_fallback_approach(current_step)

    def _try_fallback_approach(self, current_step: Step) -> bool:
        """Fallback approach when reflection fails - extract key words."""
        words = current_step.text.split()
        key_words = [w.strip('.,!?:;').lower() for w in words if len(w) > 3 and w.isalpha()][:4]
        
        if key_words:
            revised_step = " ".join(key_words)
        else:
            revised_step = "general purpose tool"
        
        revised_step = revised_step.strip().replace('\n', ' ')[:80]
        
        if revised_step:
            logger.info(f"Fallback: Simplified step from '{current_step.text}' to '{revised_step}'")
            current_step.text = revised_step
            current_step.status = "pending"
            current_step.tool_id = None
            return True
        else:
            logger.warning("Could not generate meaningful revision")
            return False

    # 7. STEP CLASSIFICATION --------------------------------------------
    def _classify_step(self, step: Step, state: ReasonerState) -> StepType:
        """Classify a plan step as TOOL_USING or REASONING via a lightweight LLM prompt.

        The prompt is intentionally minimal to control token cost and reduce
        hallucination risk.  If the LLM response is not recognised, we fall
        back to TOOL_USING to keep the agent progressing.
        """
        logger.info("Classifying step: '%s'", step.text)

        # Summarise memory keys only (avoid dumping large payloads).
        mem_keys: List[str] = []
        if hasattr(self.memory, "keys"):
            try:
                mem_keys = list(self.memory.keys())  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not list memory keys: %s", exc)
        context_summary = (
            "Memory keys: " + ", ".join(mem_keys) if mem_keys else "Memory is empty."
        )

        prompt = (
            "You are a classifier that decides whether a plan step needs an external "
            "API/tool (`tool-using`) or can be solved by internal reasoning over the "
            "already-available data (`reasoning`).\n\n"
            f"Context: {context_summary}\n"
            f"Step: '{step.text}'\n\n"
            "Reply with exactly 'tool-using' or 'reasoning'."
        )

        try:
            reply = (
                self.llm.chat(messages=[{"role": "user", "content": prompt}])
                .strip()
                .lower()
            )
            logger.debug("Classifier reply: %s", reply)
            if "reason" in reply:
                return StepType.REASONING
            if "tool" in reply:
                return StepType.TOOL_USING
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM classification error: %s", exc)

        # Default/fallback
        logger.info("Falling back to TOOL_USING classification")
        return StepType.TOOL_USING

    # 8. EXECUTE REASONING STEP ----------------------------------------
    def _execute_reasoning_step(self, step: Step, state: ReasonerState) -> Any:
        """Run an internal-reasoning step via the LLM.

        The prompt receives a *compact* JSON view of memory to keep context
        size under control.  The LLM should output ONLY the result (no
        explanatory text).  We make a best-effort attempt to parse JSON if it
        looks like JSON; otherwise we return the raw string.
        """
        logger.info("Executing reasoning step: '%s'", step.text)

        # Build a JSON payload of *relevant* memory keys.
        # 1. Explicitly include any key that appears in the step text (e.g. "search_results").
        # 2. For other keys, include only a truncated preview to keep token usage reasonable.
        mem_payload: Dict[str, Any] = {}
        referenced_keys = {k for k in getattr(self.memory, 'keys', lambda: [])() if k in step.text}

        try:
            all_keys = self.memory.keys()
            for k in all_keys:
                v = self.memory.retrieve(k)
                if k in referenced_keys:
                    # Include full value (may still be large JSON)
                    mem_payload[k] = v
                else:
                    # Provide short preview for context only
                    if isinstance(v, str):
                        mem_payload[k] = v[:200] + ("…" if len(v) > 200 else "")
                    else:
                        mem_payload[k] = v  # non-string values are usually small JSON anyway
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not build memory payload: %s", exc)

        reasoning_template = self._load_prompt("reasoning_prompt")
        
        # Enhanced reasoning with escalation capability
        enhanced_reasoning_prompt = f"""
{reasoning_template.format(
    step=step.text, 
    mem=json.dumps(mem_payload, indent=2)
)}

ESCALATION OPTION:
If you need clarification about the task or additional information to complete it,
you can ask for help by using:
<escalate_to_human reason="reasoning clarification needed" question="your specific question"/>

Otherwise, provide the result as specified above.
"""

        try:
            reply = self.llm.chat(messages=[{"role": "user", "content": enhanced_reasoning_prompt}]).strip()
            logger.debug("Reasoning LLM reply: %s", reply)

            # Process for escalation
            context = f"Step: {step.text}\nPhase: Reasoning\nGoal: {state.goal}"
            processed_reply = self._process_llm_response_for_escalation(reply, context)
            
            if processed_reply != reply:
                # Human provided guidance, use it as the reasoning result
                logger.info("Reasoning step escalated, using human guidance as result")
                return self._resolve_placeholders(processed_reply)

            # Attempt to parse JSON result if present. If successful, resolve
            # placeholders within the structure. Otherwise, resolve on the raw string.
            if processed_reply.startswith("{") and processed_reply.endswith("}"):
                try:
                    parsed_json = json.loads(processed_reply)
                    return self._resolve_placeholders(parsed_json)
                except json.JSONDecodeError:
                    # Not valid JSON, fall through to treat as a raw string
                    pass
        
            return self._resolve_placeholders(processed_reply)
        except Exception as exc:  # noqa: BLE001
            logger.error("Reasoning step failed: %s", exc)
            return f"Error during reasoning: {exc}"

    # ------------------------------------------------------------------
    # REQUIRED PUBLIC API (BaseReasoner)
    # ------------------------------------------------------------------

    def run(self, goal: str, max_iterations: int = 10):  # type: ignore[override]
        """Execute the reasoning loop until all plan steps are done or iteration cap reached."""
        logger.info("=== STARTING REASONING LOOP ===")
        logger.info(f"Goal: {goal}")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"Escalation available: {self.escalation.is_available()}")
        
        from .base_reasoner import ReasoningResult  # local import to avoid circular

        state = self._init_state(goal, {})
        tool_calls: List[Dict[str, Any]] = []

        iteration = 0
        while iteration < max_iterations:
            logger.info(f"=== ITERATION {iteration + 1}/{max_iterations} ===")
            
            # Check if goal is already marked as completed
            if state.goal_completed:
                logger.info("Goal marked as completed! Breaking from loop")
                break
            
            # Agent can proactively check if it needs human help before proceeding
            if iteration > 0 and self._should_check_for_human_guidance(state, iteration):
                if self._check_for_proactive_escalation(state, iteration):
                    continue  # Human guidance may have modified the state
            
            # Ensure we have at least one step planned.
            if not state.plan:
                logger.info("No plan exists, generating plan")
                self.plan(state)

            if self.evaluate(state):
                logger.info("Goal achieved! Breaking from loop")
                break  # goal achieved

            if not state.plan:
                logger.error("No steps in plan after planning phase!")
                break

            current_step = state.plan[0]
            logger.info(f"Executing step: {current_step.text}")

            step_type = self._classify_step(current_step, state)
            logger.info(f"Step classified as: {step_type.value}")

            try:
                if step_type is StepType.TOOL_USING:
                    tool_id = self.select_tool(current_step, state)
                    logger.info(f"Selected tool: {tool_id}")

                    result = self.act(tool_id, state, current_step)
                    logger.info(f"Action completed with result type: {type(result)}")

                    # If tool execution failed, raise an exception to trigger reflection
                    inner_result = result.get("result")
                    if hasattr(inner_result, "success") and not inner_result.success:
                        err_msg = f"Tool '{tool_id}' failed: {getattr(inner_result, 'error', 'No error details')}"
                        logger.error(err_msg)
                        raise RuntimeError(err_msg)

                    tool_calls.append({
                        "tool_id": tool_id,
                        "step": current_step.text,
                        "result": result,
                    })
                else:
                    result = self._execute_reasoning_step(current_step, state)
                    logger.info("Reasoning step output produced")

                self.observe(result, state)
                logger.info("Observation phase completed")
                
            except Exception as e:  # noqa: BLE001
                logger.error(f"Step execution failed: {e}")
                logger.exception("Full exception details:")
                
                err_msg = str(e)
                state.history.append(f"Step failed: {err_msg}")

                # Ask the LLM to repair / re-phrase the step
                logger.info("Attempting to reflect and revise step")
                if not self.reflect(current_step, err_msg):
                    # If reflection returns False we remove the step to avoid loops
                    logger.warning("Reflection failed, marking step as failed and removing")
                    current_step.status = "failed"
                    state.plan.popleft()
                else:
                    logger.info("Step revised, will retry on next iteration")

            iteration += 1
            logger.info(f"Iteration {iteration} completed")

        logger.info("=== REASONING LOOP COMPLETE ===")
        success = state.goal_completed or self.evaluate(state)
        logger.info(f"Final success status: {success} (goal_completed: {state.goal_completed})")
        logger.info(f"Total tool calls made: {len(tool_calls)}")
        logger.info(f"Final history: {state.history}")

        final_answer = "Goal completed." if success else "Unable to complete goal within iteration limit."
        logger.info(f"Final answer: {final_answer}")

        result = ReasoningResult(
            final_answer=final_answer,
            iterations=len(tool_calls),
            tool_calls=tool_calls,
            success=success,
            error_message=None if success else "Max iterations reached or failure during steps",
        )
        logger.info(f"Returning result: {result}")
        return result
    
    def _should_check_for_human_guidance(self, state: ReasonerState, iteration: int) -> bool:
        """Determine if the agent should proactively check for human guidance."""
        if not self.escalation.is_available():
            return False
            
        # Let the agent decide every few iterations if it wants to check for guidance
        # Remove automatic failure/complexity triggers - agent should decide
        return iteration > 0 and iteration % 4 == 0  # Check every 4 iterations
    
    def _check_for_proactive_escalation(self, state: ReasonerState, iteration: int) -> bool:
        """Let the agent proactively ask for human guidance."""
        if not self.escalation.is_available():
            return False
            
        # Build context for the agent to decide
        remaining_steps = list(state.plan)
        recent_failures = [h for h in state.history if "failed" in h.lower()]
        
        escalation_check_prompt = f"""
You are working on the goal: "{state.goal}"

Current situation:
- Iteration: {iteration}
- Completed steps: {len(state.history)}
- Remaining steps: {len(remaining_steps)}

Remaining plan:
{chr(10).join([f"- {step.text}" for step in remaining_steps[:3]])}
{"..." if len(remaining_steps) > 3 else ""}

Recent activity:
{chr(10).join(state.history[-3:]) if state.history else "None"}

Do you want to ask a human for guidance at this point? You have full autonomy to decide based on:
- Your confidence in the current approach
- Whether you need clarification about anything
- If you want confirmation before proceeding
- Any other reason you think human input would be helpful

Respond with:
- <escalate_to_human reason="your reason" question="your specific question"/> - to ask for guidance
- CONTINUE - to proceed without asking

Your choice:"""

        try:
            response = self.llm.chat(messages=[{"role": "user", "content": escalation_check_prompt}]).strip()
            logger.info(f"Proactive escalation check response: {response}")
            
            # Process escalation request
            context = f"Goal: {state.goal}\nIteration: {iteration}\nPhase: Proactive Check"
            processed_response = self._process_llm_response_for_escalation(response, context)
            
            if processed_response != response:
                # Human provided guidance, incorporate it
                logger.info("Agent proactively escalated, incorporating human guidance")
                
                # Update goal or add guidance to history
                if "goal:" in processed_response.lower():
                    # Update goal if human provided clarification
                    state.goal = processed_response
                    state.plan.clear()  # Clear plan to regenerate with new goal
                else:
                    # Add as contextual guidance
                    state.history.append(f"Human guidance: {processed_response}")
                
                return True
                
        except Exception as e:
            logger.warning(f"Proactive escalation check failed: {e}")
        
        return False

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_fenced_code(text: str) -> str:
        """Return the first triple‑backtick‑fenced block, else raise."""
        logger.debug("Extracting fenced code from text")
        m = re.search(r"```[\s\S]+?```", text)
        if not m:
            logger.error("No fenced plan in LLM response")
            raise RuntimeError("No fenced plan in LLM response")
        fenced = m.group(0)
        logger.debug(f"Found fenced block: {fenced}")
        
        # Remove opening and closing fences (```)
        inner = fenced.strip("`")  # remove all backticks at ends
        # After stripping, drop any leading language hint (e.g. ```markdown)
        if "\n" in inner:
            inner = inner.split("\n", 1)[1]  # drop first line (language) if present
        # Remove trailing fence that may remain after stripping leading backticks
        if inner.endswith("```"):
            inner = inner[:-3]
        result = inner.strip()
        logger.debug(f"Extracted inner content: {result}")
        return result

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        """Parse JSON even if the LLM wrapped it in a Markdown fence."""
        logger.debug(f"Parsing JSON from text: {text}")
        text = text.strip()
        
        # Check if text is wrapped in markdown code fences
        if text.startswith("```") and "```" in text[3:]:
            # Extract content between markdown fences
            pattern = r"```(?:json)?\s*([\s\S]+?)\s*```"
            match = re.search(pattern, text)
            if match:
                text = match.group(1).strip()
                logger.debug(f"Removed markdown fences from JSON")
        
        try:
            result = json.loads(text or "{}")
            logger.debug(f"Parsed JSON result: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Failed to parse JSON: {e}\n{text}")

    def _resolve_placeholders(self, obj: Any) -> Any:
        """Delegate placeholder resolution to ScratchPadMemory."""
        logger.debug(f"Resolving placeholders in: {obj}")
        try:
            result = self.memory.resolve_placeholders(obj)
            logger.debug(f"Placeholder resolution result: {result}")
            return result
        except KeyError as e:
            logger.warning(f"Memory placeholder resolution failed: {e}")
            logger.warning("Continuing with unresolved placeholders - this may cause tool execution to fail")
            # Return the original object with unresolved placeholders
            return obj

    
