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
import re
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_reasoner import BaseReasoner
from ..platform.jentic_client import JenticClient  # local wrapper, not the raw SDK
from ..utils.llm import BaseLLM, LiteLLMChatLLM
from ..memory.scratch_pad import ScratchPadMemory
from ..utils.logger import get_logger

# Initialize module logger using the shared logging utility
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum number of self-healing attempts for a single plan step.
MAX_REFLECTION_ATTEMPTS = 2

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

    BULLET_PLAN_PROMPT = textwrap.dedent(
        """
        You are a planning assistant.  Given the user goal below, output a
        *markdown bullet list* plan to achieve it.

        Requirements:
        • Each action is on its own line, starting with "- ".  Use 2 spaces
          per indentation level to indicate sub‑steps.
        • Be concrete and include the target object and purpose.
        • If a step's result should be kept for later, append "-> store:
          <key>" where <key> is a short variable name.
        • For steps that might fail (e.g., finding an item), add a sub-bullet with a backup plan starting with `-> if fails:`.
        • Do not mention any specific external tool names.
        • Enclose ONLY the list in triple backticks.
        • Always append the goal to the end of each step.

        Example:
        Goal: 'Find an interesting nytimes article that came out recently'

        ```
        - Find recent news articles about 'artificial intelligence' -> store: search_results (goal: Find an interesting nytimes article that came out recently)
          -> if fails: Report that the article search failed.
        - From the search_results, identify the most interesting article -> store: interesting_article (goal: Find an interesting nytimes article that came out recently)
          -> if fails: Report that no interesting articles were found.
        - Extract the title, URL, and summary from the interesting_article -> store: article_info (goal: Find an interesting nytimes article that came out recently)
        - Return the article_info to the user (goal: Find an interesting nytimes article that came out recently)
        ```

        Real:
        Goal: {goal}
        ```
        """
    )

    PARAM_GENERATION_PROMPT = textwrap.dedent(
        """
        You are about to call the tool **{tool_id}**.
        {tool_schema}
        {memory_enum}

        Current goal: {goal}

        Provide ONLY a JSON object with the arguments for the call. DO NOT WRAP IN MARKDOWN CODE BLOCKS.
        IMPORTANT: Return ONLY the raw JSON object without any ```json or ``` markers.

        IMPORTANT RULES:
        1. Extract actual parameter values from the goal context when possible
        2. For IDs extracted from URLs, parse the relevant parts (e.g., entity IDs, resource IDs, etc.)
        3. **SMART MEMORY EXTRACTION**: When memory contains structured data (arrays, objects), extract specific values you need:
           - Find items by matching attributes
           - Extract the actual values from the matching items
           - DO NOT use placeholders when actual data is available in memory
        4. Only use ${{memory.<key>}} placeholders for values that are explicitly listed above in available memory AND cannot be extracted
        5. If a required parameter cannot be determined from the goal or memory, use a descriptive placeholder
        6. Do NOT output markdown formatting - provide raw JSON only

        Note: Authentication credentials will be automatically injected by the platform.
        """
    )

    def __init__(
        self,
        jentic: JenticClient,
        memory: ScratchPadMemory,
        llm: Optional[BaseLLM] = None,
        model: str = "gpt-4o",
        max_iters: int = 20,
        search_top_k: int = 15,
    ) -> None:
        logger.info(f"Initializing BulletPlanReasoner with model={model}, max_iters={max_iters}, search_top_k={search_top_k}")
        super().__init__()
        self.jentic = jentic
        self.memory = memory
        self.llm = llm or LiteLLMChatLLM(model=model)
        self.max_iters = max_iters
        self.search_top_k = search_top_k
        logger.info("BulletPlanReasoner initialization complete")

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
            prompt = self.BULLET_PLAN_PROMPT.format(goal=state.goal)
            logger.debug(f"Planning prompt:\n{prompt}")
            
            messages = [{"role": "user", "content": prompt}]
            logger.info("Calling LLM for plan generation")
            response = self.llm.chat(messages=messages)
            logger.info(f"LLM planning response:\n{response}")
            
            logger.info("Extracting fenced code from response")
            plan_markdown = self._extract_fenced_code(response)
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
            raise RuntimeError(f"No tool found for step: {plan_step.text}")

        logger.info("Found tool candidates:")
        for i, hit in enumerate(hits):
            hit_name = hit.get('name', hit.get('id', 'Unknown')) if isinstance(hit, dict) else getattr(hit, 'name', 'Unknown')
            hit_desc = hit.get('description', '') if isinstance(hit, dict) else getattr(hit, 'description', '')
            hit_id = hit.get('id', 'Unknown') if isinstance(hit, dict) else getattr(hit, 'id', 'Unknown')
            logger.info(f"  {i+1}. {hit_name} (ID: {hit_id}) - {hit_desc}")

        tool_lines = "\n".join([
            f"{i+1}. {h.get('name', h.get('id', 'Unknown')) if isinstance(h, dict) else getattr(h, 'name', 'Unknown')} — {h.get('description', '') if isinstance(h, dict) else getattr(h, 'description', '')}"
            for i, h in enumerate(hits)
        ])
        
        # Include goal context in the selection prompt for better decision making
        goal_info = ""
        if plan_step.goal_context:
            goal_info = f"\nGoal context: {plan_step.goal_context}"
        elif state.goal:
            goal_info = f"\nOverall goal: {state.goal}"
        
        select_prompt = f"""Current plan step:
"{plan_step.text}"{goal_info}

Candidate tools discovered (reply with ONLY the *number* of the best match):
{tool_lines}

IMPORTANT: 
- Reply with ONLY a single number (1, 2, 3, etc.). 
- Choose tools that match the platform/service mentioned in the goal context.
- If none fit, reply with "0".

Number:"""
        
        logger.debug(f"Tool selection prompt:\n{select_prompt}")
        
        messages = [{"role": "user", "content": select_prompt}]
        logger.info("Calling LLM for tool selection")
        reply = self.llm.chat(messages=messages).strip()
        logger.info(f"LLM tool selection response: '{reply}'")

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

        keyword_prompt = textwrap.dedent(f"""
            You are an expert at rephrasing a technical developer task into a clear,
            capability-focused search query for a tool marketplace. Your goal is to
            generate a query that accurately describes the desired functionality.

            **Example 1:**
            Task: "Search jentic for a 'nytimes' api to search for articles"
            Search Query: "New York Times API for searching articles"

            **Example 2:**
            Task: "Search jentic for a 'Discord' api to post a message"
            Search Query: "Discord API for posting messages"

            **Example 3:**
            Task: "Find a tool to create a new lead in Salesforce"
            Search Query: "Salesforce API for lead creation"

            **Real Task:**
            Task: "{context_text}"

            Search Query:""")

        logger.info("Calling LLM for keyword extraction")
        messages = [{"role": "user", "content": keyword_prompt}]
        keywords = self.llm.chat(messages=messages).strip()

        # Clean up the response, removing potential quotes
        keywords = keywords.strip('"\'')

        logger.info(f"AI extracted keywords: '{keywords}'")
        return keywords

    # 3. ACT ------------------------------------------------------------
    def act(self, tool_id: str, state: ReasonerState):
        logger.info("=== ACTION PHASE ===")
        logger.info(f"Executing action with tool_id: {tool_id}")
        
        logger.info("Loading tool information from Jentic")
        tool_info = self.jentic.load(tool_id)
        logger.debug(f"Tool info: {tool_info}")
        
        # Use tool info directly without filtering credentials
        # Jentic platform should handle credential injection automatically
        if isinstance(tool_info, dict):
            tool_schema = tool_info
        elif hasattr(tool_info, 'schema_summary') and isinstance(tool_info.schema_summary, dict):
            tool_schema = tool_info.schema_summary
        else:
            tool_schema = str(tool_info)
        logger.debug(f"Tool schema: {tool_schema}")

        logger.info("Enumerating memory for prompt")
        memory_enum = self.memory.enumerate_for_prompt()
        logger.debug(f"Memory enumeration: {memory_enum}")

        def _escape_braces(text: str) -> str:
            """Escape curly braces so str.format doesn't treat them as placeholders."""
            return text.replace('{', '{{').replace('}', '}}')

        # Convert tool schema to string for formatting if it's a dict
        tool_schema_str = str(tool_schema) if isinstance(tool_schema, dict) else tool_schema

        prompt = self.PARAM_GENERATION_PROMPT.format(
            tool_id=tool_id,
            tool_schema=_escape_braces(tool_schema_str),
            memory_enum=_escape_braces(memory_enum),
            goal=state.goal,
        )
        logger.debug(f"Parameter generation prompt:\n{prompt}")
        
        messages = [{"role": "user", "content": prompt}]
        logger.info("Calling LLM for parameter generation")
        args_json = self.llm.chat(messages=messages)
        logger.info(f"LLM parameter response:\n{args_json}")
        
        try:
            logger.info("Parsing JSON parameters")
            args: Dict[str, Any] = self._safe_json_loads(args_json)
            logger.debug(f"Parsed args: {args}")
        except ValueError as e:
            logger.error(f"Failed to parse JSON args: {e}")
            logger.error(f"Raw args_json: {args_json}")
            raise RuntimeError(f"LLM produced invalid JSON args: {e}\n{args_json}")

        # Host‑side memory placeholder substitution (simple impl)
        logger.info("Resolving memory placeholders")
        concrete_args = self._resolve_placeholders(args)
        logger.debug(f"Concrete args after placeholder resolution: {concrete_args}")

        logger.info(f"Executing tool {tool_id} with args: {concrete_args}")
        result = self.jentic.execute(tool_id, concrete_args)
        logger.info(f"Tool execution result: {result}")
        return result

    # 4. OBSERVE --------------------------------------------------------
    def observe(self, observation: Any, state: ReasonerState):
        logger.info("=== OBSERVATION PHASE ===")
        logger.info(f"Processing observation: {observation}")
        
        if not state.plan:
            logger.error("No current step to observe - plan is empty!")
            return state
            
        current_step = state.plan[0]
        logger.info(f"Updating step: {current_step.text}")
        
        current_step.result = observation
        current_step.status = "done"
        logger.debug(f"Step status updated to: {current_step.status}")

        if current_step.store_key:
            logger.info(f"Storing result in memory with key: {current_step.store_key}")
            # Save into memory with a generic description
            self.memory.set(
                key=current_step.store_key,
                value=observation,
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
        logger.info("=== REFLECTION PHASE ===")
        logger.info(f"Reflecting on failed step: {current_step.text}")
        logger.info(f"Error message: {err_msg}")
        logger.info(f"Reflection attempts so far: {current_step.reflection_attempts}")
        
        # Limit reflection attempts to prevent infinite loops
        if current_step.reflection_attempts >= MAX_REFLECTION_ATTEMPTS:
            logger.warning(
                "Max reflection attempts (%s) reached for step, giving up",
                MAX_REFLECTION_ATTEMPTS,
            )
            return False
            
        current_step.reflection_attempts += 1
        
        # Generic fallback - extract key action words
        words = current_step.text.split()
        # Filter for meaningful words (nouns, verbs) and take the first few
        key_words = [w.strip('.,!?:;').lower() for w in words if len(w) > 3 and w.isalpha()][:4]
        
        if key_words:
            revised_step = " ".join(key_words)
        else:
            revised_step = "general purpose tool"
        
        # Ensure single line and reasonable length
        if revised_step:
            revised_step = revised_step.strip().replace('\n', ' ')[:80]
            
            logger.info(f"Simplified step from '{current_step.text}' to '{revised_step}'")
            current_step.text = revised_step
            current_step.status = "pending"
            current_step.tool_id = None
            return True
        else:
            logger.warning("Could not generate meaningful revision")
            return False

    # ------------------------------------------------------------------
    # REQUIRED PUBLIC API (BaseReasoner)
    # ------------------------------------------------------------------

    def run(self, goal: str, max_iterations: int = 10):  # type: ignore[override]
        """Execute the reasoning loop until all plan steps are done or iteration cap reached."""
        logger.info("=== STARTING REASONING LOOP ===")
        logger.info(f"Goal: {goal}")
        logger.info(f"Max iterations: {max_iterations}")
        
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

            try:
                tool_id = self.select_tool(current_step, state)
                logger.info(f"Selected tool: {tool_id}")
                
                result = self.act(tool_id, state)
                logger.info(f"Action completed with result type: {type(result)}")

                tool_call_record = {
                    "tool_id": tool_id,
                    "step": current_step.text,
                    "result": result,
                }
                tool_calls.append(tool_call_record)
                logger.debug(f"Recorded tool call: {tool_call_record}")

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

    
