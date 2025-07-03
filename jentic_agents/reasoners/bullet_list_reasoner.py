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
    failed: bool = False  # Track if the plan has failed steps


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

    # -------------------------------------------------------------------
    # Heuristic search helpers (generic, no hard-coding)
    # -------------------------------------------------------------------

    def _build_search_query(self, step: "Step") -> str:
        """Turn a plan step into a concise capability search query."""
        text = step.text.lower()
        text = re.sub(r"[\"'`]", "", text)  # strip quotes
        tokens = [t for t in re.split(r"\W+", text) if t]
        stop = {"the", "a", "an", "to", "in", "on", "for", "with", "of", "and"}
        tokens = [t for t in tokens if t not in stop]
        return " ".join(tokens[:8])  # keep first few meaningful words

    def _select_tool_with_llm(
        self, step: "Step", hits: List[Dict[str, Any]], state: "ReasonerState"
    ) -> Optional[str]:
        """
        Asks the LLM to choose the best tool from a list of candidates.
        """
        if not hits:
            return None

        # Format the list of tools for the LLM prompt.
        candidate_prompt = "\n".join(
            [
                f"{idx + 1}. ID: {tool['id']}, Name: {tool['name']}, Description: {tool['description']}"
                for idx, tool in enumerate(hits)
            ]
        )
        
        select_tool_template = self._load_prompt("select_tool")
        prompt = select_tool_template.format(
            goal=state.goal,
            plan_step=step.text,
            memory_keys=", ".join(self.memory.keys()),
            tool_candidates=candidate_prompt
        )

        try:
            response = self.llm.chat(messages=[{"role": "user", "content": prompt}])
            # The LLM should return a single number.
            tool_index = int(response.strip()) - 1

            if 0 <= tool_index < len(hits):
                selected_tool = hits[tool_index]
                logger.info(f"LLM selected tool #{tool_index + 1}: {selected_tool['id']} ({selected_tool['name']})")
                return selected_tool["id"]
            else:
                logger.warning(f"LLM returned an invalid tool index: {tool_index + 1}. Falling back to heuristic selection.")
                return hits[0]["id"]  # Fallback to the first tool
        except (ValueError, IndexError) as e:
            logger.error(f"Error during LLM tool selection: {e}. Falling back to heuristic selection.")
            return hits[0]["id"] # Fallback to the first tool

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
            bullet_plan_template = self._load_prompt("bullet_plan")
            prompt = bullet_plan_template.format(goal=state.goal)
            logger.debug(f"Planning prompt:\n{prompt}")
            
            messages = [{"role": "user", "content": prompt}]
            logger.info("Calling LLM for plan generation")
            response = self.llm.chat(messages=messages)
            logger.info(f"LLM planning response:\n{response}")

            # The plan is inside a markdown code fence.
            plan_md = self._extract_fenced_code(response)
            state.plan = parse_bullet_plan(plan_md)
            
            logger.info(f"Generated plan with {len(state.plan)} steps:")
            for i, step in enumerate(state.plan):
                logger.info(f"  Step {i+1}: {step.text}")

            state.history.append(f"Plan generated ({len(state.plan)} steps)")
        # except Exception as e:
        #     logger.error(f"Failed to generate or parse plan: {e}")
        #     state.failed = True
        #     state.history.append("Failed to generate plan")

    # 2. SELECT TOOL ----------------------------------------------------
    def select_tool(self, plan_step: Step, state: ReasonerState):
        """
        Selects a tool for a given plan step. It now uses an LLM to choose from search results.
        """
        logger.info("=== TOOL SELECTION PHASE ===")
        logger.info(f"Selecting tool for step: {plan_step.text}")

        search_query = self._build_search_query(plan_step)
        logger.info(f"Heuristic search query: {search_query}")
        search_hits = self.jentic.search(search_query, top_k=self.search_top_k)

        if not search_hits:
            raise RuntimeError(f"No tools found for query: '{search_query}'")

        logger.info(f"Jentic search returned {len(search_hits)} results")

        # Use the LLM to choose the best tool from the search results.
        tool_id = self._select_tool_with_llm(plan_step, search_hits, state)
        
        if not tool_id:
            # This can happen if the LLM fails to choose or returns an invalid format.
            # As a fallback, we can take the first result.
            logger.warning("LLM tool selection failed. Falling back to the first search hit.")
            tool_id = search_hits[0]['id']

        plan_step.tool_id = tool_id
        return tool_id

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

        param_generation_template = self._load_prompt("param_generation")
        prompt = param_generation_template.format(
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

        # Unpack tool results to store only the meaningful, serializable output.
        value_to_store = observation
        success = True  # Assume success unless proven otherwise

        # Case 1: Tool execution result (dict with 'result' object)
        if isinstance(observation, dict) and "result" in observation:
            result_obj = observation.get("result")
            # Check for explicit success=False from Jentic SDK
            if hasattr(result_obj, "success") and result_obj.success is False:
                success = False
                # If there's an error message, store it.
                if hasattr(result_obj, "error") and result_obj.error:
                    value_to_store = {"error": result_obj.error, "details": result_obj.output}
                    logger.warning(f"Tool execution failed: {result_obj.error}")
                else:
                    value_to_store = {"error": "Tool execution failed without a specific message."}
            elif hasattr(result_obj, "output"):
                logger.debug("Unpacking tool result object to store its output.")
                value_to_store = result_obj.output

        # Case 2: Reasoning step result (raw string or dict)
        elif isinstance(observation, str) and observation.strip().lower() in ("null", ""):
            success = False
            value_to_store = {"error": "Reasoning step produced no output."}
            logger.warning(value_to_store["error"])
        elif isinstance(observation, dict) and "error" in observation:
            success = False
            value_to_store = observation # Keep the error info
            logger.warning(f"Reasoning step returned an error: {observation['error']}")

        current_step.result = value_to_store

        if success:
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

            logger.info("Removing completed step from plan")
            state.plan.popleft()  # Advance to the next step
        else:
            current_step.status = "failed"
            state.failed = True
            logger.warning(f"Step '{current_step.text}' failed. Marking plan as failed.")
            history_entry = f"{current_step.text} -> failed"
            state.history.append(history_entry)
            # We do NOT pop the step, allowing reflection to potentially fix it.

        logger.info(f"Remaining steps in plan: {len(state.plan)}")
        return state

    # 5. EVALUATE -------------------------------------------------------
    def evaluate(self, state: ReasonerState) -> bool:
        logger.info("=== EVALUATION PHASE ===")
        is_complete = (not state.plan) and (not state.failed)
        logger.info(
            "Plan complete: %s (remaining steps: %s, failed: %s)",
            is_complete,
            len(state.plan),
            state.failed,
        )

        if is_complete:
            logger.info("All steps completed successfully!")
        elif state.failed:
            logger.warning("Plan has failed steps; cannot mark goal as complete.")
        else:
            logger.info(
                "Next step to execute: %s",
                state.plan[0].text if state.plan else "None",
            )

        return is_complete

    # 6. REFLECT (optional) --------------------------------------------
    def reflect(self, current_step: Step, err_msg: str, state: "ReasonerState") -> bool:
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
        
        reflection_template = self._load_prompt("reflection_prompt")
        prompt = reflection_template.format(
            goal=state.goal,
            failed_step_text=current_step.text,
            error_message=err_msg,
            history="\n".join(state.history)
        )

        logger.info("Calling LLM for reflection")
        revised_step = self.llm.chat(messages=[{"role": "user", "content": prompt}]).strip()

        if "AUTH_FAILURE" in revised_step:
            logger.error("Reflection indicates an unrecoverable authentication failure.")
            return False

        if revised_step:
            logger.info(f"LLM revised step from '{current_step.text}' to '{revised_step}'")
            current_step.text = revised_step
            current_step.status = "pending"
            current_step.tool_id = None
            return True
        else:
            logger.warning("LLM reflection did not provide a revised step.")
            return False

    # 7. STEP CLASSIFICATION --------------------------------------------
    def _classify_step(self, step: Step, state: ReasonerState) -> StepType:
        """Rule-based classification to avoid an extra LLM call."""
        logger.info("(Rule) classifying step: '%s'", step.text)

        text_lower = step.text.lower()

        tool_verbs = [
            "send", "post", "create", "add", "upload", "delete", "get",
            "retrieve", "access", "list", "search", "find",
        ]
        reasoning_verbs = [
            "analyze", "extract", "identify", "summarize", "summary", "summaries",
        ]

        # TOOL_USING if any tool verb appears
        if any(v in text_lower for v in tool_verbs):
            return StepType.TOOL_USING

        # REASONING if a reasoning verb appears AND a referenced memory key exists.
        if any(v in text_lower for v in reasoning_verbs):
            try:
                # Check if any of the keys in memory are mentioned in the step text.
                # This is a heuristic to see if the step has data to operate on.
                all_memory_keys = self.memory.keys()
                if any(key in text_lower for key in all_memory_keys):
                    logger.info("Classifying as REASONING because step references existing memory.")
                    return StepType.REASONING
            except Exception:  # noqa: BLE001
                # If memory access fails for some reason, fall through.
                pass

        # Default to TOOL_USING (safer)
        logger.info("Classifying as TOOL_USING by default.")
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
        reasoning_prompt = reasoning_template.format(
            step=step.text, 
            mem=json.dumps(mem_payload, indent=2)
        )

        try:
            reply = self.llm.chat(messages=[{"role": "user", "content": reasoning_prompt}]).strip()
            logger.debug("Reasoning LLM reply: %s", reply)

            # Attempt to parse JSON result if present. If successful, resolve
            # placeholders within the structure. Otherwise, resolve on the raw string.
            if reply.startswith("{") and reply.endswith("}"):
                try:
                    parsed_json = json.loads(reply)
                    return self._resolve_placeholders(parsed_json)
                except json.JSONDecodeError:
                    # Not valid JSON, fall through to treat as a raw string
                    pass
        
            return self._resolve_placeholders(reply)
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
        
        from .base_reasoner import ReasoningResult  # local import to avoid circular

        state = self._init_state(goal, {})
        tool_calls: List[Dict[str, Any]] = []

        iteration = 0
        while iteration < max_iterations:
            logger.info(f"=== ITERATION {iteration + 1}/{max_iterations} ===")

            # If the plan failed in a previous iteration, and reflection didn't fix it, stop.
            if state.failed:
                logger.error("A step has failed and could not be recovered. Terminating loop.")
                break

            # Check if goal is already marked as completed
            if state.goal_completed:
                logger.info("Goal marked as completed! Breaking from loop")
                break
            
            # Ensure we have at least one step planned.
            if iteration == 0 and not state.plan:
                logger.info("Generating initial plan")
                self.plan(state)
            
            # If the plan is empty, we are done.
            if not state.plan:
                logger.info("Plan is empty. Goal is considered complete.")
                state.goal_completed = True
                break

            current_step = state.plan[0]
            # If the current step is already 'done' or 'failed', something is wrong.
            # This can happen if reflection logic is faulty. For now, we'll log and skip.
            if current_step.status not in ("pending", "running"):
                 logger.warning(f"Skipping step '{current_step.text}' with unexpected status '{current_step.status}'")
                 state.plan.popleft()
                 continue

            logger.info(f"Executing step: {current_step.text}")

            step_type = self._classify_step(current_step, state)
            logger.info(f"Step classified as: {step_type.value}")

            try:
                if step_type is StepType.TOOL_USING:
                    tool_id = self.select_tool(current_step, state)
                    logger.info(f"Selected tool: {tool_id}")

                    result = self.act(tool_id, state)
                    logger.info(f"Action completed with result type: {type(result)}")

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
                logger.error(f"Step execution failed with an exception: {e}")
                logger.exception("Full exception details:")
                
                err_msg = str(e)
                state.history.append(f"Step failed: {err_msg}")
                current_step.status = "failed"
                state.failed = True # Mark plan as failed

                # Ask the LLM to repair / re-phrase the step
                logger.info("Attempting to reflect and revise step")
                if self.reflect(current_step, err_msg, state):
                    logger.info("Step revised, will retry on next iteration. Un-marking plan as failed for now.")
                    state.failed = False # Allow the loop to continue for a retry
                else:
                    # If reflection returns False we give up on this step.
                    logger.warning("Reflection failed. The plan will now terminate.")


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

    
