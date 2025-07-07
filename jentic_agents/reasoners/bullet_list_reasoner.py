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

# local utils
from ..utils.json_cleanser import strip_backtick_fences, cleanse
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

# Maximum number of reflection attempts before giving up on a failed step
MAX_REFLECTION_ATTEMPTS = 3

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
    tool_name: Optional[str] = None  # chosen tool name for logging
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
    """Very lenient parser that turns an indented bullet list OR JSON array into Step objects."""
    # Use shared helper for fence stripping
    markdown_stripped = strip_backtick_fences(markdown)
    # Now check for JSON array
    if markdown_stripped.startswith('[') and markdown_stripped.endswith(']'):
        try:
            logger.info("Parsing plan as JSON array")
            json_steps = json.loads(markdown_stripped)
            steps = []
            for step_data in json_steps:
                if isinstance(step_data, dict):
                    text = step_data.get('text', '')
                    step_type = step_data.get('step_type', '')
                    store_key = step_data.get('store_key')
                    # Extract goal context from parentheses if present
                    goal_context = None
                    goal_match = re.search(r'\(\s*goal:\s*([^)]+)\s*\)', text)
                    if goal_match:
                        goal_context = goal_match.group(1).strip()
                        # Remove the goal context from the main content
                        text = re.sub(r'\s*\(\s*goal:[^)]+\s*\)', '', text).strip()
                    step = Step(
                        text=text,
                        indent=0,  # JSON format doesn't use indentation
                        store_key=store_key,
                        goal_context=goal_context
                    )
                    # Store step_type as an attribute for later use
                    step.step_type = step_type
                    steps.append(step)
            logger.info(f"Parsed {len(steps)} steps from plan (JSON mode)")
            return deque(steps)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse as JSON: {e}, falling back to markdown parsing")
            # Fall through to markdown parsing
    # Original markdown bullet parsing logic
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
        f"Parsed {len(leaf_steps)} steps from plan (original {len(steps)})"
    )
    return deque(leaf_steps)


# ---------------------------------------------------------------------------
# BulletPlanReasoner implementation
# ---------------------------------------------------------------------------


class BulletPlanReasoner(BaseReasoner):
    """Concrete Reasoner that follows the BulletPlan strategy."""

    @staticmethod
    def _load_prompt(prompt_name: str):
        """Load a prompt from the prompts directory. Return JSON if file is JSON, else string."""
        current_dir = Path(__file__).parent.parent
        prompt_path = current_dir / "prompts" / f"{prompt_name}.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('{'):
                    return json.loads(content)
                return content
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise RuntimeError(f"Prompt file not found: {prompt_path}")

    # -------------------------------------------------------------------
    # Heuristic search helpers (generic, no hard-coding)
    # -------------------------------------------------------------------

    def _build_search_query(self, step: "Step") -> str:
        """
        Turn a plan step into an API-hub search query using the LLM-based keyword-extraction prompt only.
        If the LLM fails, raise an error.
        """
        try:
            kw_template = self._load_prompt("keyword_extraction")
            if isinstance(kw_template, dict):
                kw_template["inputs"]["context_text"] = step.text
                prompt = json.dumps(kw_template, ensure_ascii=False)
            else:
                prompt = kw_template.format(context_text=step.text)

            reply = self.llm.chat([{"role": "user", "content": prompt}]).strip()
            if reply:
                logger.info("LLM keyword-extraction produced query: %s", reply)
                return reply
            else:
                raise RuntimeError("LLM keyword-extraction returned empty query.")
        except Exception as e:
            logger.error(f"Keyword-extraction prompt failed: {e}")
            raise RuntimeError(f"Keyword-extraction prompt failed: {e}")

    def _select_tool_with_llm(
        self, step: "Step", hits: List[Dict[str, Any]], state: "ReasonerState"
    ) -> Optional[str]:
        """Ask the LLM to choose the best tool from `hits`.

        Selection order:
        1. Numeric index in the LLM reply (1-based).
        2. Tool ID or name substring present in reply.
        3. First hit whose provider (api_name domain) appears in the plan step.
        4. Fallback to the first search hit.
        """
        if not hits:
            return None

        # Build numbered candidate list
        numbered_lines: List[str] = []
        for idx, h in enumerate(hits, 1):
            name = h.get("name") if isinstance(h, dict) else getattr(h, "name", None)
            if not name or not str(name).strip():
                name = h.get("id", "Unknown") if isinstance(h, dict) else getattr(h, "id", "Unknown")
            api_name = h.get("api_name") if isinstance(h, dict) else getattr(h, "api_name", None)
            desc = h.get("description", "") if isinstance(h, dict) else getattr(h, "description", "")
            display = f"{name} ({api_name})" if api_name else name
            numbered_lines.append(f"{idx}. {display} — {desc}")
        candidate_block = "\n".join(numbered_lines)

        # Fill the prompt
        prompt_tpl = self._load_prompt("select_tool")
        if isinstance(prompt_tpl, dict):
            prompt_tpl["inputs"].update(
                {
                    "goal": state.goal,
                    "plan_step": step.text,
                    "memory_keys": ", ".join(self.memory.keys()),
                    "tool_candidates": candidate_block,
                }
            )
            prompt = json.dumps(prompt_tpl, ensure_ascii=False)
        else:
            prompt = prompt_tpl.format(
                goal=state.goal,
                plan_step=step.text,
                memory_keys=", ".join(self.memory.keys()),
                tool_candidates=candidate_block,
            )

        raw_reply = self.llm.chat(messages=[{"role": "user", "content": prompt}]).strip()
        logger.debug("LLM tool-selection reply: %s", raw_reply)

        # 1️⃣ Numeric selection
        match = re.search(r"(\d+)", raw_reply)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(hits):
                chosen = hits[idx]
                return chosen["id"] if isinstance(chosen, dict) else getattr(chosen, "id", "unknown")

        # 2️⃣ ID or name substring
        lower_reply = raw_reply.lower()
        for h in hits:
            hid = h["id"] if isinstance(h, dict) else getattr(h, "id", "")
            hname = h.get("name", "") if isinstance(h, dict) else getattr(h, "name", "")
            if hid and hid.lower() in lower_reply:
                return hid
            if hname and hname.lower() in lower_reply:
                return h["id"] if isinstance(h, dict) else getattr(h, "id", "unknown")

        # 3️⃣ Provider heuristic
        step_lower = step.text.lower()
        for h in hits:
            api_name = h.get("api_name", "").lower() if isinstance(h, dict) else getattr(h, "api_name", "").lower()
            if api_name and api_name.split(".")[0] in step_lower:
                return h["id"] if isinstance(h, dict) else getattr(h, "id", "unknown")

        # 4️⃣ Fallback first hit
        first = hits[0]
        step.tool_name = first.get("name", None) if isinstance(first, dict) else getattr(first, "name", None)
        return first["id"] if isinstance(first, dict) else getattr(first, "id", "unknown")

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
            if isinstance(bullet_plan_template, dict):
                # Fill in the goal in the JSON template
                bullet_plan_template["inputs"]["goal"] = state.goal
                prompt = json.dumps(bullet_plan_template, ensure_ascii=False)
            else:
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

    # 2. SELECT TOOL ----------------------------------------------------
    def select_tool(self, plan_step: Step, state: ReasonerState):
        """
        Selects a tool for a given plan step. It now uses an LLM to choose from search results.
        """
        logger.info("=== TOOL SELECTION PHASE ===")
        logger.info(f"Selecting tool for step: {plan_step.text}")

        # ---- Fast path: "Execute <memory_key> ..." → reuse stored tool_id ----
        exec_match = re.match(r"execute\s+([\w\-_]+)", plan_step.text.strip(), re.IGNORECASE)
        if exec_match:
            mem_key = exec_match.group(1)
            if mem_key in self.memory.keys():
                stored = self.memory.retrieve(mem_key)
                if isinstance(stored, dict) and "id" in stored:
                    logger.info(f"Reusing tool_id from memory key '{mem_key}': {stored['id']}")
                    plan_step.tool_id = stored["id"]
                    return stored["id"]

        search_query = self._build_search_query(plan_step)
        # Only log the search query and tool candidates, not every phase or step
        logger.info(f"Search query: {search_query}")
        search_hits = self.jentic.search(search_query, top_k=self.search_top_k)

        if not search_hits:
            logger.error(f"No tools found for query: '{search_query}'")
            raise RuntimeError(f"No tools found for query: '{search_query}'")

        # Sort hits so that those whose provider (api_name or its domain) is mentioned in the plan step are at the top
        step_text_lower = plan_step.text.lower()
        def provider_mentioned(hit):
            api_name = hit.get('api_name', '').lower() if isinstance(hit, dict) else getattr(hit, 'api_name', '').lower()
            if not api_name:
                return False
            domain_part = api_name.split('.')[0]
            return (api_name in step_text_lower) or (domain_part in step_text_lower)
        search_hits = sorted(search_hits, key=lambda h: not provider_mentioned(h))

        tool_names_list = []
        tool_lines_list = []
        for i, h in enumerate(search_hits):
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

        select_tool_template = self._load_prompt("select_tool")
        if isinstance(select_tool_template, dict):
            select_tool_template["inputs"]["goal"] = state.goal
            select_tool_template["inputs"]["plan_step"] = plan_step.text
            select_tool_template["inputs"]["memory_keys"] = ", ".join(self.memory.keys())
            select_tool_template["inputs"]["tool_candidates"] = tool_lines
            prompt = json.dumps(select_tool_template, ensure_ascii=False)
        else:
            prompt = select_tool_template.format(
                goal=state.goal,
                plan_step=plan_step.text,
                memory_keys=", ".join(self.memory.keys()),
                tool_candidates=tool_lines
            )

        try:
            response = self.llm.chat(messages=[{"role": "user", "content": prompt}])
            raw_reply = response.strip()
            num_match = re.search(r"(\d+)", raw_reply)
            if num_match:
                tool_index = int(num_match.group(1)) - 1
            else:
                tool_index = None
            if tool_index is not None and 0 <= tool_index < len(search_hits):
                selected_tool = search_hits[tool_index]
                logger.info(
                    "LLM selected tool #%s: %s (%s)",
                    tool_index + 1,
                    selected_tool["id"] if isinstance(selected_tool, dict) else getattr(selected_tool, 'id', 'unknown'),
                    selected_tool.get("name", "unnamed") if isinstance(selected_tool, dict) else getattr(selected_tool, 'name', 'unnamed'),
                )
                # store tool name
                plan_step.tool_name = selected_tool.get("name", None) if isinstance(selected_tool, dict) else getattr(selected_tool, "name", None)
                return selected_tool["id"] if isinstance(selected_tool, dict) else getattr(selected_tool, 'id', 'unknown')
            # No valid tool selected
            raise RuntimeError("LLM tool selection failed: No valid tool index or provider match. Aborting step.")
        except (ValueError, IndexError, Exception) as e:
            logger.error(f"Error during LLM tool selection: {e}. Using first search hit as fallback.")
            return search_hits[0]["id"]

    # 3. ACT ------------------------------------------------------------
    def act(self, tool_id: str, state: ReasonerState, current_step: Step):
        """
        Generate parameters for and execute a Jentic tool.
        """
        logger.info("=== ACTION PHASE ===")
        logger.info(f"Executing action with tool_id: {tool_id}")

        # If the tool_id is a memory key (e.g., 'send_op'), resolve it to the actual tool_id
        if tool_id in self.memory.keys():
            stored = self.memory.retrieve(tool_id)
            # If stored is a dict with an 'id' field, use that
            if isinstance(stored, dict) and 'id' in stored:
                logger.info(f"Resolved memory key '{tool_id}' to tool_id: {stored['id']}")
                tool_id = stored['id']
            else:
                logger.warning(f"Memory key '{tool_id}' did not resolve to a valid tool_id. Using as-is.")

        logger.info("Loading tool information from Jentic")
        tool_info = self.jentic.load(tool_id)
        logger.debug(f"Tool info: {tool_info}")

        # Extract required fields if present
        required_fields = tool_info.get('required', []) if isinstance(tool_info, dict) else []
        logger.debug(f"Tool schema for parameter generation: {tool_info}")

        logger.info("Enumerating memory for prompt")
        memory_enum = self.memory.enumerate_for_prompt()
        logger.debug(f"Memory enumeration: {memory_enum}")

        def _escape_braces(text: str) -> str:
            """Escape curly braces so str.format doesn't treat them as placeholders."""
            return text.replace('{', '{{').replace('}', '}}')

        # Convert tool schema to string for formatting if it's a dict
        tool_schema_str = str(tool_info) if isinstance(tool_info, dict) else tool_info

        param_generation_template = self._load_prompt("param_generation")
        if isinstance(param_generation_template, dict):
            param_generation_template["inputs"]["tool_id"] = tool_id
            param_generation_template["inputs"]["tool_schema"] = _escape_braces(tool_schema_str)
            param_generation_template["inputs"]["memory_enum"] = _escape_braces(memory_enum)
            param_generation_template["inputs"]["goal"] = state.goal
            prompt = json.dumps(param_generation_template, ensure_ascii=False)
        else:
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

        # Check for missing required fields and re-prompt if needed
        missing_fields = [field for field in required_fields if field not in args]
        if missing_fields:
            logger.warning(f"Missing required fields in LLM output: {missing_fields}. Re-prompting LLM.")
            # Add explicit instruction to include all required fields
            explicit_prompt = (
                f"{prompt}\n\nIMPORTANT: You MUST include all required fields in the parameters: {', '.join(required_fields)}."
            )
            messages = [{"role": "user", "content": explicit_prompt}]
            args_json = self.llm.chat(messages=messages)
            logger.info(f"LLM parameter response (re-prompt):\n{args_json}")
            try:
                args: Dict[str, Any] = self._safe_json_loads(args_json)
                logger.debug(f"Parsed args after re-prompt: {args}")
            except ValueError as e:
                logger.error(f"Failed to parse JSON args after re-prompt: {e}")
                logger.error(f"Raw args_json: {args_json}")
                raise RuntimeError(f"LLM produced invalid JSON args after re-prompt: {e}\n{args_json}")

        # Host‑side memory placeholder substitution (simple impl)
        logger.info("Resolving memory placeholders")
        concrete_args = self._resolve_placeholders(args)
        logger.debug(f"Concrete args after placeholder resolution: {concrete_args}")

        logger.info(f"Executing tool {tool_id} with selected arguments.")
        result = self.jentic.execute(tool_id, concrete_args)
        # Support both dict and OperationResult
        success = None
        if isinstance(result, dict):
            inner = result.get('result', None)
            if hasattr(inner, 'success'):
                success = getattr(inner, 'success', None)
            elif isinstance(inner, dict):
                success = inner.get('success', None)
        elif hasattr(result, 'success'):
            success = getattr(result, 'success', None)
        logger.info(f"Tool execution completed. Success: {success}")
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
            # If result_obj is a WorkflowResult or OperationResult, check its .success
            if hasattr(result_obj, "success"):
                if not result_obj.success:
                    success = False
                    if hasattr(result_obj, "error") and result_obj.error:
                        value_to_store = {"error": result_obj.error, "details": getattr(result_obj, "output", None)}
                        logger.warning(f"Tool execution failed: {result_obj.error}")
                    else:
                        value_to_store = {"error": "Tool execution failed without a specific message."}
                else:
                    # Only mark as done if .success is True
                    if hasattr(result_obj, "output"):
                        value_to_store = result_obj.output
            else:
                # Fallback: treat as before
                if hasattr(result_obj, "output"):
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
            f"Plan complete: {is_complete} | Remaining steps: {len(state.plan)} | Failed: {state.failed}"
        )

        if is_complete:
            logger.info("All steps completed successfully.")
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
        logger.info(f"Reflecting on failed step: {current_step.text} | Error: {err_msg} | Attempts: {current_step.reflection_attempts}")
        
        # Limit reflection attempts to prevent infinite loops
        if current_step.reflection_attempts >= MAX_REFLECTION_ATTEMPTS:
            logger.warning(
                "Max reflection attempts (%s) reached for step, giving up",
                MAX_REFLECTION_ATTEMPTS,
            )
            return False
            
        current_step.reflection_attempts += 1
        
        reflection_template = self._load_prompt("reflection_prompt")
        if isinstance(reflection_template, dict):
            reflection_template["inputs"]["goal"] = state.goal
            reflection_template["inputs"]["failed_step_text"] = current_step.text
            reflection_template["inputs"]["error_message"] = err_msg
            reflection_template["inputs"]["history"] = "\n".join(state.history)
            prompt = json.dumps(reflection_template, ensure_ascii=False)
        else:
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
        if isinstance(reasoning_template, dict):
            reasoning_template["inputs"]["step"] = step.text
            reasoning_template["inputs"]["memory"] = json.dumps(mem_payload, indent=2)
            reasoning_prompt = json.dumps(reasoning_template, ensure_ascii=False)
        else:
            reasoning_prompt = reasoning_template.format(
                step=step.text,
                mem=json.dumps(mem_payload, indent=2)
            )

        try:
            reply = self.llm.chat(messages=[{"role": "user", "content": reasoning_prompt}]).strip()
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
        logger.info(f"Reasoning started for goal: {goal} | Max iterations: {max_iterations}")
        
        from .base_reasoner import ReasoningResult  # local import to avoid circular

        state = self._init_state(goal, {})
        tool_calls: List[Dict[str, Any]] = []

        iteration = 0
        while iteration < max_iterations:
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            # If the plan failed in a previous iteration, and reflection didn't fix it, stop.
            if state.failed:
                logger.error("A step has failed and could not be recovered. Terminating loop.")
                break

            # Check if goal is already marked as completed
            if state.goal_completed:
                logger.info("Goal marked as completed! Breaking from loop")
                break
            
            # Agent can proactively check if it needs human help before proceeding
            if iteration > 0 and self._should_check_for_human_guidance(state, iteration):
                if self._check_for_proactive_escalation(state, iteration):
                    continue  # Human guidance may have modified the state
            
            # Ensure we have at least one step planned.
            if iteration == 0 and not state.plan:
                logger.info("Generating plan for goal.")
                self.plan(state)
            
            # If the plan is empty, we are done.
            if not state.plan:
                logger.info("Plan empty. Marking goal as complete.")
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

            # Use explicit step_type if present
            step_type = getattr(current_step, "step_type", None)
            if step_type:
                logger.info(f"Step type: {step_type}")
            else:
                step_type_enum = self._classify_step(current_step, state)
                step_type = step_type_enum.value.upper()
                logger.info(f"Step type: {step_type}")

            try:
                if step_type == "SEARCH":
                    tool_id = self.select_tool(current_step, state)
                    logger.info(f"Tool selected: {tool_id} ({current_step.tool_name})")
                    # Optionally store tool_id in memory if store_key is present
                    if current_step.store_key:
                        self.memory.set(
                            key=current_step.store_key,
                            value={"id": tool_id},
                            description=f"Tool ID for step '{current_step.text}'",
                        )
                        logger.debug(f"Stored tool_id in memory with key '{current_step.store_key}'")
                    result = {"tool_id": tool_id}
                elif step_type == "EXECUTE":
                    tool_id = self.select_tool(current_step, state)
                    logger.info(f"Tool selected: {tool_id} ({current_step.tool_name})")
                    result = self.act(tool_id, state, current_step)
                    logger.info(f"Action result type: {type(result)}")
                    tool_calls.append({
                        "tool_id": tool_id,
                        "step": current_step.text,
                        "result": result,
                    })
                elif step_type == "REASON":
                    result = self._execute_reasoning_step(current_step, state)
                    logger.info("Reasoning step completed.")
                elif step_type in ["TOOL_USING", "TOOL"]:  # Support both formats
                    tool_id = self.select_tool(current_step, state)
                    logger.info(f"Tool selected: {tool_id} ({current_step.tool_name})")
                    result = self.act(tool_id, state, current_step)
                    logger.info(f"Action result type: {type(result)}")
                    tool_calls.append({
                        "tool_id": tool_id,
                        "step": current_step.text,
                        "result": result,
                    })
                elif step_type in ["REASONING"]:  # Support both formats
                    result = self._execute_reasoning_step(current_step, state)
                    logger.info("Reasoning step completed.")
                else:
                    logger.warning(f"Unknown step_type '{step_type}', defaulting to EXECUTE.")
                    tool_id = self.select_tool(current_step, state)
                    logger.info(f"Tool selected: {tool_id} ({current_step.tool_name})")
                    result = self.act(tool_id, state, current_step)
                    logger.info(f"Action result type: {type(result)}")
                    tool_calls.append({
                        "tool_id": tool_id,
                        "step": current_step.text,
                        "result": result,
                    })

                self.observe(result, state)
                
            except Exception as e:  # noqa: BLE001
                logger.error(f"Step execution failed: {e}")
                logger.info("Attempting reflection on failed step.")
                if self.reflect(current_step, str(e), state):
                    logger.info("Step revised, retrying.")
                    state.failed = False
                else:
                    logger.warning("Reflection failed. Ending reasoning loop.")


            iteration += 1
        logger.info(f"Reasoning loop complete. Success: {state.goal_completed or not state.failed}")
        logger.info(f"Total tool calls: {len(tool_calls)}")
        final_answer = "Goal completed." if state.goal_completed else "Unable to complete goal."
        logger.info(f"Final answer: {final_answer}")
        if state.goal_completed:
            tool_summary = "\n  Used tool(s):\n" + "\n".join([
                f"    {call['tool_id']} — {call['step']}" for call in tool_calls
            ])
            logger.info(f"\n✅ Goal Completed Successfully\n{final_answer}\n{tool_summary}")
        result = ReasoningResult(
            final_answer=final_answer,
            iterations=len(tool_calls),
            tool_calls=tool_calls,
            success=state.goal_completed,
            error_message=None if state.goal_completed else "Max iterations reached or failure during steps",
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

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        """Parse JSON even if the LLM wrapped it in a Markdown fence."""
        logger.debug(f"Parsing JSON from text: {text}")
        text = strip_backtick_fences(text.strip())
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

            # --- sanitize any strings that still include markdown code fences ---
            sanitized = cleanse(result)
            logger.debug(f"Placeholder resolution result: {sanitized}")
            return sanitized
        except KeyError as e:
            logger.warning(f"Memory placeholder resolution failed: {e}")
            logger.warning("Continuing with unresolved placeholders - this may cause tool execution to fail")
            # Return the original object with unresolved placeholders
            return obj
