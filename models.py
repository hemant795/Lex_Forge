"""
models.py
=========
Typed OpenEnv models for LexForge.
All classes extend real openenv.core base types — NOT custom Pydantic classes.

LexAction    → extends openenv.core.Action
LexObservation → extends openenv.core.Observation  (done + reward built in)
LexState     → extends openenv.core.State           (episode_id + step_count built in)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import Field, model_validator
from openenv.core import Action, Observation, State


# ─── Action ───────────────────────────────────────────────────────────────────

class LexAction(Action):
    """
    A single agent action in LexForge.

    The agent picks one action_type from available_actions in the observation
    and fills in the relevant parameters for that action type only.

    Action Types
    ------------
    flag_clause              Mark a clause as risky — provide risks[] labels
    clear_clause             Mark a clause as safe/benign
    classify_risk            Provide risk labels + regulation citation for a clause
    rewrite_clause           Submit a rewritten version of a risky clause
    generate_report          Produce structured due diligence audit report
    detect_adversarial_clauses  Identify obfuscated violation clauses (Expert task)
    submit_multi_party_sign_off  Confirm both parties accept redlined contract (Expert task)
    """

    action_type: str = Field(
        ...,
        description="One of the 7 available action types listed in the observation"
    )

    # ── flag_clause / clear_clause / classify_risk / rewrite_clause ──────────
    clause_id: Optional[str] = Field(
        default=None,
        description="Clause ID to act on — e.g. 'C001', 'C004'. From pending_clauses in context."
    )

    # ── flag_clause / classify_risk ───────────────────────────────────────────
    risks: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of risk label strings from RISK_TAXONOMY. "
            "e.g. ['gdpr_purpose_limitation_violation', 'aml_risk']"
        )
    )

    # ── classify_risk ─────────────────────────────────────────────────────────
    citation: Optional[str] = Field(
        default=None,
        description="Regulation article cited. e.g. 'GDPR Art.5(1)(b)' or 'PMLA S.3'"
    )

    # ── rewrite_clause ────────────────────────────────────────────────────────
    rewritten_text: Optional[str] = Field(
        default=None,
        description="Full rewritten clause text that addresses the identified risk"
    )

    # ── generate_report ───────────────────────────────────────────────────────
    report: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Audit report dict. Required keys: "
            "executive_summary, flagged_clauses, severity_matrix, "
            "recommendations, deal_breakers"
        )
    )

    # ── detect_adversarial_clauses ────────────────────────────────────────────
    clause_ids: Optional[List[str]] = Field(
        default=None,
        description="List of clause IDs identified as adversarially obfuscated. e.g. ['C019', 'C020']"
    )

    # ── submit_multi_party_sign_off ───────────────────────────────────────────
    party_a_satisfied: Optional[bool] = Field(
        default=None,
        description="True if the client party accepts all redlined clauses"
    )
    party_b_satisfied: Optional[bool] = Field(
        default=None,
        description="True if the counterparty accepts all redlined clauses"
    )
    balance_justification: Optional[str] = Field(
        default=None,
        description="Written explanation of how the balance between parties was achieved (min 50 chars)"
    )

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


# ─── Observation ─────────────────────────────────────────────────────────────

class LexObservation(Observation):
    """
    Full observation returned by reset() and step().

    Inherits from openenv.core.Observation which already provides:
      done: bool       — True when episode ends
      reward: float    — reward from last action (None on reset)
      metadata: dict   — additional metadata

    Context dict keys
    -----------------
    pending_clauses      {clause_id: clause_spec}  — up to 3 unseen clauses
    reviewed_count       int  — clauses already reviewed
    total_clauses        int  — total in this task
    flagged_so_far       list — clause IDs flagged as risky so far
    completed_stages     list — pipeline stages done (easy tasks have 1 stage)
    risk_taxonomy        dict — available risk label strings with descriptions
    jurisdiction_note    str  — governing law hint (Expert task only)
    """

    task_id: str = Field(
        ...,
        description="Current task: easy | medium | hard | expert"
    )
    step: int = Field(
        default=0,
        description="Current step number (1-indexed after first step)"
    )
    max_steps: int = Field(
        default=10,
        description="Maximum steps allowed for this task"
    )
    cascade_multiplier: float = Field(
        default=0.99,
        description="Current cascade reward multiplier [0.5, 1.5]. Starts at 1.0."
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Valid action_type strings for the current state"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full environment context — clauses, progress, taxonomy"
    )
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-grader score breakdown for interpretability"
    )
    partial_progress: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Fraction of objectives completed [0.0, 1.0]"
    )
    reward_explanation: str = Field(
        default="",
        description="Human-readable explanation for rewards"
    )

    @model_validator(mode="before")
    @classmethod
    def clip_floats(cls, data):
        if isinstance(data, dict):
            def clip(obj):
                if isinstance(obj, dict): return {k: clip(v) for k,v in obj.items()}
                if isinstance(obj, list): return [clip(v) for v in obj]
                if isinstance(obj, float):
                    if obj <= 0.0: return 0.01
                    if obj >= 1.0: return 0.99
                return obj
            return clip(data)
        return data

    @classmethod
    def _clip_all_floats(cls, data):
        """Clip all float values to (0.01, 0.99) to satisfy validator."""
        if isinstance(data, dict):
            return {k: cls._clip_all_floats(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._clip_all_floats(v) for v in data]
        elif isinstance(data, float):
            if data <= 0.0: return 0.01
            if data >= 1.0: return 0.99
            return data
        return data

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


# ─── State ────────────────────────────────────────────────────────────────────

class LexState(State):
    """
    Internal environment state — returned by the /state endpoint.

    Inherits from openenv.core.State which already provides:
      episode_id: Optional[str]
      step_count: int

    extra="allow" means additional fields can be added without validation errors.
    """

    task_id: str = Field(default="easy")
    reviewed_clauses: List[str] = Field(default_factory=list)
    flagged_clauses: List[str] = Field(default_factory=list)
    cleared_clauses: List[str] = Field(default_factory=list)
    rewritten_clauses: List[Dict[str, Any]] = Field(default_factory=list)
    false_positives: int = Field(default=0)
    false_negatives: int = Field(default=0)
    completed_stages: List[str] = Field(default_factory=list)
    cascade_multiplier: float = Field(default=0.49)
    episode_rewards: List[float] = Field(default_factory=list)
    audit_report: Dict[str, Any] = Field(default_factory=dict)
    adversarial_detected: List[str] = Field(default_factory=list)
    multi_party_scores: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }
# ── Ensure Pydantic v2 validators are fully built ─────────────────────────────
# Must be called at module level so openenv-core's deserialize_action() works.
LexAction.model_rebuild()
LexObservation.model_rebuild()
LexState.model_rebuild()
