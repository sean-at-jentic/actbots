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
import logging
import os
import re
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_reasoner import BaseReasoner
from ..platform.jentic_client import JenticClient  # local wrapper, not the raw SDK
from ..utils.llm import BaseLLM, LiteLLMChatLLM
from ..memory.scratch_pad import ScratchPadMemory

# Configure logging to file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "bullet_reasoner.log")

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent duplicate logs
logger.propagate = False

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
    status: str = "pending"  # pending | running | done | failed
    result: Any = None
    tool_id: Optional[str] = None  # chosen Jentic tool
    reflection_attempts: int = 0  # track how many times we've tried to fix this step


@dataclass
class ReasonerState:
    goal: str
    plan: deque[Step] = field(default_factory=deque)
    history: List[str] = field(default_factory=list)  # raw trace lines


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

        # Simple directive detection:  "… -> store: weather"
        store_key = None
        if "->" in content:
            content, directive = [part.strip() for part in content.split("->", 1)]
            if directive.startswith("store:"):
                store_key = directive.split(":", 1)[1].strip()
                logger.debug(f"Found store directive: {store_key}")

        step = Step(text=content, indent=indent_level, store_key=store_key)
        steps.append(step)
        logger.debug(f"Parsed step: {step}")

    logger.info(f"Parsed {len(steps)} steps from bullet plan")
    return deque(steps)


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
        • Do not mention any specific external tool names.
        • Enclose ONLY the list in triple backticks.

        Example:
        Goal: 'Find an interesting nytimes article that came out recently'

        ```
        - Find an appropriate api or workflow to use to achieve the goal
          - Search jentic for nytimes api
          - Load execution info for the nytimes api operation or workflow which best suits the goal
        - Execute the nytimes api operation or workflow -> store: nytimes_api_result
        - Decide on an interesting article
        - Store that information: -> store: nytimes_article_info
        - return the title, url, and summary of that article
        ```

        Real:
        Goal: {goal}
        ```
        """
    )

    CANDIDATE_SELECTION_PROMPT = textwrap.dedent(
        """
        Current plan step:
        "{step_text}"

        Candidate tools discovered (reply with the *number* of the best match):
        {tool_lines}
        If none fit, reply with "0" and suggest a better search query.
        """
    )

    PARAM_GENERATION_PROMPT = textwrap.dedent(
        """
        You are about to call the tool **{tool_id}**.
        {tool_schema}
        {memory_enum}

        Provide ONLY a JSON object with the arguments for the call. Do not
        wrap in markdown. Use ${{memory.<key>}} placeholders for values that
        must be filled from memory.
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
        
        if plan_step.tool_id:
            logger.info(f"Step already has tool_id: {plan_step.tool_id}")
            return plan_step.tool_id

        # Search Jentic by NL description
        logger.info(f"Searching Jentic for tools matching: {plan_step.text}")
        hits = self.jentic.search(plan_step.text, top_k=self.search_top_k)
        logger.info(f"Jentic search returned {len(hits)} results")
        
        if not hits:
            logger.error(f"No tools found for step: {plan_step.text}")
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
        
        select_prompt = self.CANDIDATE_SELECTION_PROMPT.format(
            step_text=plan_step.text, tool_lines=tool_lines
        )
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
            m = re.search(r"\d+", reply)
            if not m:
                raise ValueError("No leading integer found in LLM reply")
            idx = int(m.group(0)) - 1
            logger.debug(f"Parsed tool index from LLM reply: {idx}")
            
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

    # 3. ACT ------------------------------------------------------------
    def act(self, tool_id: str):
        logger.info("=== ACTION PHASE ===")
        logger.info(f"Executing action with tool_id: {tool_id}")
        
        logger.info("Loading tool information from Jentic")
        tool_info = self.jentic.load(tool_id)
        logger.debug(f"Tool info: {tool_info}")
        
        tool_schema = tool_info.schema_summary if hasattr(tool_info, 'schema_summary') else str(tool_info)
        logger.debug(f"Tool schema: {tool_schema}")

        logger.info("Enumerating memory for prompt")
        memory_enum = self.memory.enumerate_for_prompt()
        logger.debug(f"Memory enumeration: {memory_enum}")

        def _escape_braces(text: str) -> str:
            """Escape curly braces so str.format doesn't treat them as placeholders."""
            return text.replace('{', '{{').replace('}', '}}')

        prompt = self.PARAM_GENERATION_PROMPT.format(
            tool_id=tool_id,
            tool_schema=_escape_braces(tool_schema),
            memory_enum=_escape_braces(memory_enum),
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
        
        logger.info("Removing completed step from plan")
        state.plan.popleft()  # advance to next step
        logger.info(f"Remaining steps in plan: {len(state.plan)}")
        
        return state

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
        MAX_REFLECTION_ATTEMPTS = 2
        if current_step.reflection_attempts >= MAX_REFLECTION_ATTEMPTS:
            logger.warning(f"Max reflection attempts ({MAX_REFLECTION_ATTEMPTS}) reached for step, giving up")
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
                
                result = self.act(tool_id)
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
        success = self.evaluate(state)
        logger.info(f"Final success status: {success}")
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
        # Remove leading/trailing triple backticks if present
        if text.startswith("```") and text.endswith("```"):
            text = text.strip("`").strip()
            logger.debug("Removed markdown fences from JSON")
        
        result = json.loads(text or "{}")
        logger.debug(f"Parsed JSON result: {result}")
        return result

    def _resolve_placeholders(self, obj: Any) -> Any:
        """Delegate placeholder resolution to ScratchPadMemory."""
        logger.debug(f"Resolving placeholders in: {obj}")
        result = self.memory.resolve_placeholders(obj)
        logger.debug(f"Placeholder resolution result: {result}")
        return result
