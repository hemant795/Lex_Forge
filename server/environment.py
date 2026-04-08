"""
server/environment.py
=====================
LexForgeEnvironment — the core OpenEnv environment.

Extends openenv.core.Environment.
Implements reset(), step(), and the state @property.
"""

from __future__ import annotations
import sys, os, threading
sys.path.insert(0, os.path.dirname(__file__))

from typing import Any, Dict, List, Optional
from openenv.core import Environment

from fixtures import (
    CLAUSES, TASK_CLAUSE_SETS, TASK_MAX_STEPS,
    TASK_AVAILABLE_ACTIONS, TASK_DESCRIPTIONS,
    RISK_TAXONOMY, BENIGN_CLAUSE_IDS, ADVERSARIAL_CLAUSE_IDS,
)
from graders import (
    grade_identification, grade_classification,
    grade_rewrite, grade_report, grade_adversarial,
    grade_signoff, update_cascade, apply_cascade,
)

# Import models from parent package
import importlib.util, pathlib
_models_path = pathlib.Path(__file__).parent.parent / "models.py"
_spec = importlib.util.spec_from_file_location("models", _models_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LexAction      = _mod.LexAction
LexObservation = _mod.LexObservation
LexState       = _mod.LexState



def _clip_reward(r: float) -> float:
    """Rewards must be strictly between 0 and 1 (exclusive)."""
    if r is None: return None
    return max(0.001, min(0.999, float(r)))

# ─── Clause window size ───────────────────────────────────────────────────────
WINDOW = 3   # How many pending clauses are visible at once


# Module-level store: survives openenv-core's per-request instantiation
_STORE_LOCK = threading.Lock()
_STORE: dict = {'state': None, 'clauses': None}

class LexForgeEnvironment(Environment):
    """
    OpenEnv environment for legal document review.

    Supports concurrent sessions — each reset() creates fresh state.
    SUPPORTS_CONCURRENT_SESSIONS = True (openenv-core requirement).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Internal state ────────────────────────────────────────────────────────
    _state: Optional[LexState] = None

    # ── reset() ───────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        injected_clauses: Optional[List[Dict[str, Any]]] = None,
        **kwargs,           # forward-compat: ignore unknown params
    ) -> LexObservation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id          : "easy" | "medium" | "hard" | "expert". Defaults to "easy".
        injected_clauses : optional list of clause dicts from parser_agent.py.
                           If provided, they are merged into the CLAUSES registry
                           for this episode only.
        """
        task_id = task_id or "easy"
        if task_id not in TASK_CLAUSE_SETS:
            task_id = "easy"

        # Inject parsed clauses if provided (parser_agent pipeline)
        if injected_clauses:
            _episode_clauses = dict(CLAUSES)
            for c in injected_clauses:
                cid = c.get("id", f"P{len(_episode_clauses)+1:03d}")
                _episode_clauses[cid] = {
                    "id": cid,
                    "domain": c.get("domain", "unknown"),
                    "contract_type": c.get("contract_type", "unknown"),
                    "severity": c.get("severity", "UNSCORED"),
                    "text": c.get("text", ""),
                    "risks": c.get("risks", []),
                    "is_benign": c.get("is_benign", False),
                    "is_adversarial": False,
                    "indian_law": "",
                    "global_law": "",
                    "rewrite_hint": "",
                }
                if cid not in TASK_CLAUSE_SETS[task_id]:
                    TASK_CLAUSE_SETS[task_id].append(cid)
            self._episode_clauses = _episode_clauses
        else:
            self._episode_clauses = dict(CLAUSES)

        self._state = LexState(
            task_id=task_id,
            episode_id=episode_id,
            step_count=0,
        )

        with _STORE_LOCK:
            _STORE['state']   = self._state
            _STORE['clauses'] = self._episode_clauses
        return self._build_observation(reward=None, done=False)

    # ── step() ────────────────────────────────────────────────────────────────

    def step(self, action: LexAction, **kwargs) -> LexObservation:
        """
        Process one agent action and return updated observation.

        Returns LexObservation directly — NOT wrapped in StepResult.
        done and reward are fields ON the observation.
        """
        with _STORE_LOCK:
            if self._state is None and _STORE['state'] is not None:
                self._state           = _STORE['state']
                self._episode_clauses = _STORE['clauses']
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        st = self._state
        st.step_count += 1
        clause_set = TASK_CLAUSE_SETS[st.task_id]
        max_steps  = TASK_MAX_STEPS[st.task_id]

        reward           = 0.0
        reward_breakdown = {}
        correct_action   = False

        atype = (action.action_type or "").strip()

        # ── flag_clause ───────────────────────────────────────────────────────
        if atype == "flag_clause":
            cid = action.clause_id
            if cid and cid in self._episode_clauses and cid not in st.reviewed_clauses:
                st.reviewed_clauses.append(cid)
                clause = self._episode_clauses[cid]
                if clause["is_benign"]:
                    st.false_positives += 1
                    reward = -0.1
                    correct_action = False
                else:
                    st.flagged_clauses.append(cid)
                    reward = 0.5
                    correct_action = True
                reward_breakdown["flag_accuracy"] = max(0.0, reward)

        # ── clear_clause ──────────────────────────────────────────────────────
        elif atype == "clear_clause":
            cid = action.clause_id
            if cid and cid in self._episode_clauses and cid not in st.reviewed_clauses:
                st.reviewed_clauses.append(cid)
                clause = self._episode_clauses[cid]
                if clause["is_benign"]:
                    st.cleared_clauses.append(cid)
                    reward = 0.5
                    correct_action = True
                else:
                    st.false_negatives += 1
                    reward = -0.2
                    correct_action = False
                reward_breakdown["clear_accuracy"] = max(0.0, reward)

        # ── classify_risk ─────────────────────────────────────────────────────
        elif atype == "classify_risk":
            cid = action.clause_id
            if cid and cid in self._episode_clauses:
                result = grade_classification(
                    cid,
                    action.risks or [],
                    action.citation,
                )
                reward = result["score"]
                correct_action = reward > 0.5
                reward_breakdown["classification_score"] = reward

        # ── rewrite_clause ────────────────────────────────────────────────────
        elif atype == "rewrite_clause":
            cid = action.clause_id
            if cid and cid in self._episode_clauses and action.rewritten_text:
                result = grade_rewrite(cid, action.rewritten_text)
                reward = result["score"]
                correct_action = reward > 0.5
                reward_breakdown["rewrite_quality"] = reward
                if reward > 0:
                    st.rewritten_clauses.append({
                        "clause_id": cid,
                        "rewritten_text": action.rewritten_text,
                        "score": reward,
                    })

        # ── generate_report ───────────────────────────────────────────────────
        elif atype == "generate_report":
            if action.report:
                result = grade_report(
                    action.report,
                    st.flagged_clauses,
                    clause_set,
                )
                reward = result["score"]
                correct_action = reward > 0.5
                reward_breakdown["report_completeness"] = reward
                st.audit_report = action.report
                if "generate_report" not in st.completed_stages:
                    st.completed_stages.append("generate_report")

        # ── detect_adversarial_clauses ────────────────────────────────────────
        elif atype == "detect_adversarial_clauses":
            result = grade_adversarial(action.clause_ids or [], clause_set)
            reward = result["score"]
            correct_action = reward > 0.5
            reward_breakdown["adversarial_f1"] = reward
            st.adversarial_detected = action.clause_ids or []
            if "detect_adversarial" not in st.completed_stages:
                st.completed_stages.append("detect_adversarial")

        # ── submit_multi_party_sign_off ───────────────────────────────────────
        elif atype == "submit_multi_party_sign_off":
            result = grade_signoff(
                action.party_a_satisfied,
                action.party_b_satisfied,
                action.balance_justification,
            )
            reward = result["score"]
            correct_action = reward >= 0.8
            reward_breakdown["balance_score"] = reward
            st.multi_party_scores = result
            if "sign_off" not in st.completed_stages:
                st.completed_stages.append("sign_off")

        # ── Update cascade ────────────────────────────────────────────────────
        st.cascade_multiplier = update_cascade(st.cascade_multiplier, correct_action)
        final_reward = apply_cascade(max(0.0, reward), st.cascade_multiplier)
        st.episode_rewards.append(final_reward)

        # ── Completion bonus ──────────────────────────────────────────────────
        all_reviewed = set(clause_set) <= set(st.reviewed_clauses)
        if all_reviewed and "complete" not in st.completed_stages:
            final_reward = min(1.0, final_reward + 0.1)
            st.completed_stages.append("complete")

        # ── Done condition ────────────────────────────────────────────────────
        done = (st.step_count >= max_steps) or all_reviewed

        with _STORE_LOCK:
            _STORE['state']   = self._state
            _STORE['clauses'] = self._episode_clauses
        final_reward = _clip_reward(final_reward)
        return self._build_observation(
            reward=final_reward,
            done=done,
            reward_breakdown=reward_breakdown,
        )

    # ── state @property ───────────────────────────────────────────────────────

    @property
    def state(self) -> LexState:
        """Return current episode state (required by openenv-core)."""
        if self._state is None:
            return LexState(task_id="easy")
        return self._state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        reward_breakdown: Optional[Dict[str, float]] = None,
    ) -> LexObservation:
        st         = self._state
        clause_set = TASK_CLAUSE_SETS[st.task_id]

        # Pending clauses window (unseen clauses, up to WINDOW)
        unseen = [
            cid for cid in clause_set
            if cid not in st.reviewed_clauses
        ]
        window_ids    = unseen[:WINDOW]
        pending_clauses = {
            cid: {
                "id":            cid,
                "text":          self._episode_clauses[cid]["text"],
                "domain":        self._episode_clauses[cid]["domain"],
                "severity_hint": self._episode_clauses[cid]["severity"],
            }
            for cid in window_ids
        }

        partial_progress = (
            len(st.reviewed_clauses) / len(clause_set)
            if clause_set else 1.0
        )

        context = {
            "pending_clauses":   pending_clauses,
            "reviewed_count":    len(st.reviewed_clauses),
            "total_clauses":     len(clause_set),
            "flagged_so_far":    list(st.flagged_clauses),
            "completed_stages":  list(st.completed_stages),
            "risk_taxonomy":     RISK_TAXONOMY,
            "task_description":  TASK_DESCRIPTIONS.get(st.task_id, ""),
        }
        if st.task_id == "expert":
            context["jurisdiction_note"] = (
                "This contract contains both UK and US governed clauses. "
                "Read the governing law clause first before auditing others."
            )

        reward_explanation = self._explain_reward(
            reward, reward_breakdown or {}, st.cascade_multiplier
        )

        return LexObservation(
            task_id=st.task_id,
            step=st.step_count,
            max_steps=TASK_MAX_STEPS[st.task_id],
            cascade_multiplier=st.cascade_multiplier,
            available_actions=TASK_AVAILABLE_ACTIONS[st.task_id],
            context=context,
            reward_breakdown=reward_breakdown or {},
            partial_progress=round(partial_progress, 4),
            reward_explanation=reward_explanation,
            done=done,
            reward=reward,
        )

    @staticmethod
    def _explain_reward(
        reward: Optional[float],
        breakdown: Dict[str, float],
        cascade: float,
    ) -> str:
        if reward is None:
            return "Episode started — no reward yet."
        parts = [f"{k}={v:.2f}" for k, v in breakdown.items()]
        components = ", ".join(parts) if parts else "no graders triggered"
        return (
            f"Reward={reward:.4f} | cascade={cascade:.1f}x | "
            f"components: {components}"
        )