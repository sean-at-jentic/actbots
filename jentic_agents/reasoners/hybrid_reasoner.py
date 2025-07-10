"""Hybrid Reasoner Design.

A reasoner that dynamically chooses between Freeform and BulletPlan reasoners
based on task complexity, as detailed in the project documentation. This approach
allows the system to balance speed and accuracy by routing simple, single-step
tasks to a fast, conversational reasoner, while reserving a more structured,
plan-first reasoner for complex, multi-step goals.

The selection is performed by a lightweight classifier that uses an LLM to
determine if a goal is 'SINGLE-STEP' or 'MULTI-STEP'.

- **FreeformReasoner**: Handles simple tasks with a direct, conversational loop.
- **BulletPlanReasoner**: Manages complex tasks with a structured plan-first
  approach.
"""

from __future__ import annotations

from typing import Any, Optional

from ..communication.hitl.base_intervention_hub import BaseInterventionHub
from ..memory.scratch_pad import ScratchPadMemory
from ..platform.jentic_client import JenticClient
from ..utils.llm import BaseLLM, LiteLLMChatLLM
from ..utils.logger import get_logger
from .base_reasoner import ReasoningResult
from .bullet_list_reasoner import BulletPlanReasoner
from .freeform_reasoner import FreeformReasoner
from ..utils.config import get_config

config = get_config()
logger = get_logger(__name__)


class HybridReasoner:
    """A reasoner that dynamically selects a strategy based on goal complexity."""

    def __init__(
        self,
        jentic: JenticClient,
        memory: ScratchPadMemory,
        llm: Optional[BaseLLM] = None,
        model: Optional[str] = None,
        intervention_hub: Optional[BaseInterventionHub] = None,
        **kwargs: Any,
    ):
        """Initializes the HybridReasoner and its underlying reasoners."""
        llm_config = config.get("llm", {})
        primary_model = model or llm_config.get("model", "gpt-4o")

        # Use the same LLM instance for both main reasoning and classification.
        self.llm = llm or LiteLLMChatLLM(model=primary_model)
        self.classification_llm = self.llm

        self.freeform = FreeformReasoner(
            jentic, memory, self.llm, intervention_hub=intervention_hub, **kwargs
        )
        self.bullet = BulletPlanReasoner(
            jentic, memory, self.llm, intervention_hub=intervention_hub, **kwargs
        )

        logger.info(
            f"HybridReasoner initialized with model '{primary_model}' for all reasoning."
        )

    def _is_simple_task(self, goal: str) -> bool:
        """Classifies a task as simple (single-step) or complex (multi-step).

        Uses a lightweight LLM call with a model optimized for speed.

        Args:
            goal: The user's goal to classify.

        Returns:
            True if the task is classified as simple, False otherwise.
        """
        logger.info(f"Classifying task complexity for goal: '{goal}'")

        prompt = (
            "Please classify the following user request as 'SINGLE-STEP' or 'MULTI-STEP'.\n\n"
            "SINGLE-STEP tasks are direct commands that can likely be completed with one or 2 tool calls (e.g., 'send a message', 'get the weather and then send a message').\n"
            "MULTI-STEP tasks require planning, information gathering, or several actions (e.g., 'research X and write a summary in a document', 'book a flight and then a hotel and then send a message').\n\n"
            f'Request: "{goal}"\n\n'
            "Classification:"
        )

        try:
            response = (
                self.classification_llm.chat([{"role": "user", "content": prompt}])
                .strip()
                .upper()
            )
            logger.info(f"Task complexity classification response: {response}")
            is_simple = "SINGLE-STEP" in response
            # Basic validation in case the model returns both keywords
            if "SINGLE-STEP" in response and "MULTI-STEP" in response:
                logger.warning(
                    "Classifier returned ambiguous response. Defaulting to MULTI-STEP."
                )
                return False
            return is_simple
        except Exception as e:
            logger.warning(
                f"Complexity classification failed: {e}. Defaulting to complex task (BulletPlan)."
            )
            return False

    def run(self, goal: str, **kwargs: Any) -> ReasoningResult:
        """Executes the reasoning loop by selecting the appropriate strategy.

        Args:
            goal: The goal for the agent to achieve.
            **kwargs: Runtime arguments to pass to the selected reasoner's run method.

        Returns:
            The result of the reasoning process.
        """
        if self._is_simple_task(goal):
            logger.info(
                f"Task classified as SIMPLE. Routing to FreeformReasoner for goal: '{goal}'"
            )
            return self.freeform.run(goal, **kwargs)
        else:
            logger.info(
                f"Task classified as COMPLEX. Routing to BulletPlanReasoner for goal: '{goal}'"
            )
            return self.bullet.run(goal, **kwargs) 