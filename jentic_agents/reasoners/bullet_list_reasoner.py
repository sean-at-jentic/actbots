# bullet_plan_reasoner.py
"""BulletPlanReasoner — a *plan‑first, late‑bind* reasoning loop.

This class implements the **BulletPlan** strategy described in chat:

1. *Plan* — LLM produces a natural‑language indented Markdown bullet list
   of steps (potentially nested). No tools are named at this stage.
2. *Select* — at run‑time, for **each** step the reasoner
   • searches Jentic for suitable tools,
   • offers the top‑k candidates to the LLM,
   • receives an index of the chosen tool (or a request to refine the
     search query).
3. *Act* — loads the chosen tool spec, prompts the LLM for parameters
   (with memory enumeration), executes the tool and stores results.
4. *Observe / Evaluate / Reflect* — passes tool output back to LLM so it
   can mark the step complete, retry, or patch the plan.

The class extends *BaseReasoner* so it can be swapped into any
*BaseAgent* unchanged.

NOTE ▸ For brevity, this file depends on the following external pieces
(which already exist in the repo skeleton):

* `JenticClient` – thin wrapper around `jentic_sdk` with `.search()`,
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

import re
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from reasoners.base_reasoner import BaseReasoner
from platform.jentic_client import JenticClient  # local wrapper, not the raw SDK
from utils.llm import call_llm  # thin helper around openai.chat.completions
from utils.memory import Memory, prompt_memory_enumeration  # simple dict‑based store

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
    steps: List[Step] = []
    for line in markdown.splitlines():
        if not line.strip():
            continue  # skip blanks
        m = BULLET_RE.match(line)
        if not m:
            # Ignore lines outside the list (e.g. fences)
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

        steps.append(Step(text=content, indent=indent_level, store_key=store_key))

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
        wrap in markdown. Use ${memory.<key>} placeholders for values that
        must be filled from memory.
        """
    )

    def __init__(
        self,
        jentic: JenticClient,
        memory: Memory,
        planner_model: str = "gpt-4o-mini",
        executor_model: str = "gpt-4o-mini",
        max_iters: int = 20,
    ) -> None:
        super().__init__(max_iters=max_iters)
        self.jentic = jentic
        self.memory = memory
        self.planner_model = planner_model
        self.executor_model = executor_model

    # ------------------------------------------------------------------
    # BaseReasoner hook implementations
    # ------------------------------------------------------------------

    def _init_state(self, goal: str, context: Dict[str, Any]) -> ReasonerState:
        return ReasonerState(goal=goal)

    # 1. PLAN -----------------------------------------------------------
    def plan(self, state: ReasonerState):
        if not state.plan:  # first call → create plan
            prompt = self.BULLET_PLAN_PROMPT.format(goal=state.goal)
            response = call_llm(model=self.planner_model, messages=[{"role": "user", "content": prompt}])
            plan_markdown = self._extract_fenced_code(response)
            state.plan = parse_bullet_plan(plan_markdown)
            state.history.append(f"Plan generated ({len(state.plan)} steps)")
        return state.plan[0]  # peek without popping

    # 2. SELECT TOOL ----------------------------------------------------
    def select_tool(self, plan_step: Step, state: ReasonerState):
        if plan_step.tool_id:
            return plan_step.tool_id  # already chosen in a retry loop

        # Search Jentic by NL description
        hits = self.jentic.search(plan_step.text, top_k=5)
        if not hits:
            # Could ask LLM to rephrase; for now fail fast
            raise RuntimeError(f"No tool found for step: {plan_step.text}")

        tool_lines = "\n".join(f"{i+1}. {h.name} — {h.description}" for i, h in enumerate(hits))
        select_prompt = self.CANDIDATE_SELECTION_PROMPT.format(
            step_text=plan_step.text, tool_lines=tool_lines
        )
        reply = call_llm(model=self.executor_model, messages=[{"role": "user", "content": select_prompt}]).strip()

        if reply.startswith("0"):
            # TODO: handle rephrase suggestion (reply may include new query)
            raise RuntimeError("LLM couldn't find a suitable tool.")

        try:
            idx = int(reply.split()[0]) - 1
            plan_step.tool_id = hits[idx].id
            return plan_step.tool_id
        except (ValueError, IndexError):
            raise RuntimeError(f"Invalid tool index reply: {reply}")

    # 3. ACT ------------------------------------------------------------
    def act(self, tool_id: str):
        tool_info = self.jentic.load(tool_id)
        tool_schema = tool_info.schema_summary  # str

        memory_enum = prompt_memory_enumeration(self.memory)

        prompt = self.PARAM_GENERATION_PROMPT.format(
            tool_id=tool_id,
            tool_schema=tool_schema,
            memory_enum=memory_enum,
        )
        args_json = call_llm(model=self.executor_model, messages=[{"role": "user", "content": prompt}])
        try:
            args: Dict[str, Any] = self._safe_json_loads(args_json)
        except ValueError as e:
            raise RuntimeError(f"LLM produced invalid JSON args: {e}\n{args_json}")

        # Host‑side memory placeholder substitution (simple impl)
        concrete_args = self._resolve_placeholders(args)

        result = self.jentic.execute(tool_id, concrete_args)
        return result

    # 4. OBSERVE --------------------------------------------------------
    def observe(self, observation: Any, state: ReasonerState):
        current_step = state.plan[0]
        current_step.result = observation
        current_step.status = "done"

        if current_step.store_key:
            self.memory[current_step.store_key] = observation

        state.history.append(f"{current_step.text} -> done")
        state.plan.popleft()  # advance to next step
        return state

    # 5. EVALUATE -------------------------------------------------------
    def evaluate(self, state: ReasonerState) -> bool:
        return not state.plan  # all steps completed

    # 6. REFLECT (optional) --------------------------------------------
    def reflect(self, state: ReasonerState):
        # Simple no‑op; could implement failed‑step repair here.
        return state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_fenced_code(text: str) -> str:
        """Return the first triple‑backtick‑fenced block, else raise."""
        m = re.search(r"```[\s\S]+?```", text)
        if not m:
            raise RuntimeError("No fenced plan in LLM response")
        fenced = m.group(0)
        # strip the first and last 
