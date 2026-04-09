"""
server/graders.py
=================
Six deterministic graders for LexForge.
Pure functions — no state, no side effects, always reproducible.

Grader 1 — grade_identification   → clause flag/clear F1
Grader 2 — grade_classification   → risk label + citation accuracy
Grader 3 — grade_rewrite          → rewrite quality (keyword + structure)
Grader 4 — grade_report           → audit report completeness
Grader 5 — grade_adversarial      → adversarial clause detection F1
Grader 6 — grade_signoff          → multi-party balance score
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple

from fixtures import (
    CLAUSES,
    RISK_TAXONOMY,
    IMPROVEMENT_KEYWORDS,
    BENIGN_CLAUSE_IDS,
    ADVERSARIAL_CLAUSE_IDS,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clamp(value: float) -> float:
    """Clamp score to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.01
    return 2 * precision * recall / (precision + recall)


def _precision_recall(
    predicted: Set[str], relevant: Set[str]
) -> Tuple[float, float]:
    if not predicted:
        return 0.01, 0.0 if relevant else 1.0
    tp = len(predicted & relevant)
    precision = tp / len(predicted)
    recall = tp / len(relevant) if relevant else 1.0
    return precision, recall


# ─── Grader 1 — Clause Identification F1 ────────────────────────────────────

def grade_identification(
    flagged_ids: List[str],
    cleared_ids: List[str],
    clause_set: List[str],
) -> Dict[str, float]:
    """
    Score clause identification (flag_clause + clear_clause actions).

    Parameters
    ----------
    flagged_ids : clauses the agent flagged as RISKY
    cleared_ids : clauses the agent marked as BENIGN
    clause_set  : all clause IDs active in this task

    Returns
    -------
    dict with keys: precision, recall, f1, false_positives, false_negatives, score
    """
    risky_in_task  = {
        cid for cid in clause_set
        if not CLAUSES[cid]["is_benign"]
    }
    benign_in_task = {
        cid for cid in clause_set
        if CLAUSES[cid]["is_benign"]
    }

    flagged = set(flagged_ids) & set(clause_set)
    cleared = set(cleared_ids) & set(clause_set)

    # True positives: correctly flagged risky clauses
    tp = flagged & risky_in_task
    # False positives: benign clauses incorrectly flagged
    fp = flagged & benign_in_task
    # False negatives: risky clauses missed (not flagged)
    fn = risky_in_task - flagged

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall    = len(tp) / len(risky_in_task)   if risky_in_task else 1.0
    f1        = _f1(precision, recall)

    return {
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "score":           round(f1, 4),
    }


# ─── Grader 2 — Risk Classification Accuracy ────────────────────────────────

def grade_classification(
    clause_id: str,
    stated_risks: List[str],
    citation: Optional[str] = None,
) -> Dict[str, float]:
    """
    Score risk label classification for a single clause.

    Parameters
    ----------
    clause_id    : the clause being classified
    stated_risks : risk labels the agent provided
    citation     : regulation article cited (e.g. 'GDPR Art.5(1)(b)')

    Returns
    -------
    dict with keys: label_score, citation_bonus, score
    """
    if clause_id not in CLAUSES:
        return {"label_score": 0.0, "citation_bonus": 0.0, "score": 0.0}

    clause = CLAUSES[clause_id]
    correct_risks = set(clause["risks"])
    stated        = set(stated_risks or [])

    # Benign clause: agent should NOT flag any risks
    if clause["is_benign"]:
        label_score = 1.0 if not stated else 0.0
        return {
            "label_score":   round(label_score, 4),
            "citation_bonus": 0.0,
            "score":          round(label_score, 4),
        }

    # Risky clause: measure overlap
    if not correct_risks:
        label_score = 1.0
    elif not stated:
        label_score = 0.0
    else:
        overlap     = len(correct_risks & stated)
        false_pos   = len(stated - correct_risks)
        label_score = overlap / len(correct_risks) - (false_pos * 0.1)
        label_score = _clamp(label_score)

    # Citation bonus: up to +0.2 if valid regulation cited
    citation_bonus = 0.0
    if citation:
        citation_lower = citation.lower()
        # Check against known law references in the clause
        indian_law = clause.get("indian_law", "").lower()
        global_law = clause.get("global_law", "").lower()
        # Simple check: key terms from citation appear in law references
        terms = [t.strip() for t in citation_lower.replace(".", " ").split() if len(t) > 2]
        law_text = indian_law + " " + global_law
        matches  = sum(1 for t in terms if t in law_text)
        if matches >= 2:
            citation_bonus = 0.2
        elif matches == 1:
            citation_bonus = 0.1

    score = _clamp(label_score * 0.8 + citation_bonus)

    return {
        "label_score":    round(label_score, 4),
        "citation_bonus": round(citation_bonus, 4),
        "score":          round(score, 4),
    }


# ─── Grader 3 — Rewrite Quality ──────────────────────────────────────────────

def grade_rewrite(
    clause_id: str,
    rewritten_text: str,
) -> Dict[str, float]:
    """
    Score rewrite quality for a risky clause.

    Parameters
    ----------
    clause_id      : clause being rewritten
    rewritten_text : agent's proposed replacement text

    Scoring
    -------
    +0.30  structural change (different from original)
    +0.70  improvement keywords present (split across risk labels)
    -0.30  dangerous patterns still present in rewrite
    = max 1.0, min 0.0

    Returns
    -------
    dict with keys: structural_change, keyword_score, danger_penalty, score
    """
    if clause_id not in CLAUSES:
        return {"structural_change": 0.0, "keyword_score": 0.0,
                "danger_penalty": 0.0, "score": 0.0}

    clause        = CLAUSES[clause_id]
    original_text = clause["text"].lower().strip()
    rewrite_lower = (rewritten_text or "").lower().strip()

    if not rewrite_lower:
        return {"structural_change": 0.0, "keyword_score": 0.0,
                "danger_penalty": 0.0, "score": 0.0}

    # Structural change: rewrite is meaningfully different from original
    structural_change = 0.30 if rewrite_lower != original_text else 0.0

    # Keyword score: check improvement keywords per risk label
    risks = clause.get("risks", [])
    keyword_score = 0.0
    if risks:
        per_risk_score = 0.70 / len(risks)
        for risk in risks:
            keywords = IMPROVEMENT_KEYWORDS.get(risk, [])
            if any(kw in rewrite_lower for kw in keywords):
                keyword_score += per_risk_score

    # Danger penalty: original dangerous patterns still present
    danger_patterns = {
        "unreasonable_duration":     ["unlimited", "perpetual", "forever", "in perpetuity"],
        "aml_risk":                  ["bearer", "without verification", "no kyc", "cash equivalent"],
        "gdpr_purpose_limitation_violation": ["any purpose", "commercially reasonable", "any use"],
        "gdpr_storage_limitation_violation": ["indefinitely", "permanently", "forever", "no limit"],
        "irrevocable_license":       ["irrevocable", "perpetual license"],
        "unilateral_disclosure_authority": ["sole discretion", "deems appropriate", "at its discretion"],
        "unfair_arbitration":        ["provider's standard", "sole arbitrator", "provider appoints"],
    }
    danger_penalty = 0.0
    for risk in risks:
        patterns = danger_patterns.get(risk, [])
        if any(pat in rewrite_lower for pat in patterns):
            danger_penalty = 0.30
            break

    score = _clamp(structural_change + keyword_score - danger_penalty)

    return {
        "structural_change": round(structural_change, 4),
        "keyword_score":     round(keyword_score, 4),
        "danger_penalty":    round(danger_penalty, 4),
        "score":             round(score, 4),
    }


# ─── Grader 4 — Audit Report Completeness ───────────────────────────────────

REQUIRED_REPORT_SECTIONS = [
    "executive_summary",
    "flagged_clauses",
    "severity_matrix",
    "recommendations",
    "deal_breakers",
]

def grade_report(
    report: Dict[str, Any],
    flagged_clause_ids: List[str],
    clause_set: List[str],
) -> Dict[str, float]:
    """
    Score audit report completeness and clause coverage.

    Parameters
    ----------
    report             : the report dict submitted by the agent
    flagged_clause_ids : clause IDs the agent flagged during the episode
    clause_set         : all clause IDs active in this task

    Scoring (50/50 split)
    ---------------------
    50% — section coverage: fraction of 5 required sections present and non-empty
    50% — clause coverage: fraction of risky clauses mentioned in flagged_clauses section

    Returns
    -------
    dict with keys: section_score, clause_coverage, score
    """
    if not report:
        return {"section_score": 0.0, "clause_coverage": 0.0, "score": 0.0}

    # Section coverage
    sections_present = 0
    for section in REQUIRED_REPORT_SECTIONS:
        val = report.get(section)
        # Non-empty: string > 10 chars, list with items, dict with keys
        if isinstance(val, str) and len(val.strip()) > 10:
            sections_present += 1
        elif isinstance(val, (list, dict)) and len(val) > 0:
            sections_present += 1

    section_score = sections_present / len(REQUIRED_REPORT_SECTIONS)

    # Clause coverage: risky clauses in flagged_clauses section of report
    risky_in_task = {
        cid for cid in clause_set
        if not CLAUSES[cid]["is_benign"]
    }
    report_flagged = set(report.get("flagged_clauses", []))
    if risky_in_task:
        clause_coverage = len(report_flagged & risky_in_task) / len(risky_in_task)
    else:
        clause_coverage = 1.0

    score = _clamp((section_score * 0.5) + (clause_coverage * 0.5))

    return {
        "section_score":   round(section_score, 4),
        "clause_coverage": round(clause_coverage, 4),
        "score":           round(score, 4),
    }


# ─── Grader 5 — Adversarial Detection F1 ────────────────────────────────────

def grade_adversarial(
    detected_ids: List[str],
    clause_set: List[str],
) -> Dict[str, float]:
    """
    Score adversarial clause detection (Expert task only).

    Parameters
    ----------
    detected_ids : clause IDs the agent claims are adversarially obfuscated
    clause_set   : all clause IDs active in this task (expert = all 20)

    Scoring
    -------
    Precision: fraction of detected clauses that are actually adversarial
    Recall:    fraction of adversarial clauses that were detected
    F1:        harmonic mean — penalises both false positives AND misses

    False positive trap: C016, C017, C018 are benign.
    Flagging them hurts precision significantly.

    Returns
    -------
    dict with keys: precision, recall, f1, false_alarms, missed, score
    """
    adversarial_in_task = set(ADVERSARIAL_CLAUSE_IDS) & set(clause_set)
    detected            = set(detected_ids or []) & set(clause_set)

    if not adversarial_in_task:
        # No adversarial clauses in this task — detection not applicable
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "false_alarms": 0, "missed": 0, "score": 1.0}

    tp           = detected & adversarial_in_task
    false_alarms = detected - adversarial_in_task
    missed       = adversarial_in_task - detected

    precision = len(tp) / len(detected) if detected else 0.0
    recall    = len(tp) / len(adversarial_in_task)
    f1        = _f1(precision, recall)

    return {
        "precision":    round(precision, 4),
        "recall":       round(recall, 4),
        "f1":           round(f1, 4),
        "false_alarms": len(false_alarms),
        "missed":       len(missed),
        "score":        round(f1, 4),
    }


# ─── Grader 6 — Multi-Party Sign-Off Balance ────────────────────────────────

def grade_signoff(
    party_a_satisfied: Optional[bool],
    party_b_satisfied: Optional[bool],
    balance_justification: Optional[str],
) -> Dict[str, float]:
    """
    Score multi-party sign-off (Expert task only).

    Parameters
    ----------
    party_a_satisfied      : True if client party accepts all redlined clauses
    party_b_satisfied      : True if counterparty accepts all redlined clauses
    balance_justification  : written explanation of balance (min 50 chars)

    Scoring
    -------
    0.40 — party_a_satisfied is True
    0.40 — party_b_satisfied is True
    0.20 — balance_justification is present and >= 50 characters
    = max 1.0

    Returns
    -------
    dict with keys: party_a, party_b, justification, score
    """
    a_score = 0.40 if party_a_satisfied is True else 0.0
    b_score = 0.40 if party_b_satisfied is True else 0.0

    justification = balance_justification or ""
    j_score = 0.20 if len(justification.strip()) >= 50 else 0.0

    score = _clamp(a_score + b_score + j_score)

    return {
        "party_a":       round(a_score, 4),
        "party_b":       round(b_score, 4),
        "justification": round(j_score, 4),
        "score":         round(score, 4),
    }


# ─── Cascade Mechanic ─────────────────────────────────────────────────────────

def update_cascade(current: float, correct: bool) -> float:
    """
    Update cascade multiplier after an action.

    correct action → +0.1 (max 1.5)
    wrong action   → -0.1 (min 0.5)
    """
    delta = 0.1 if correct else -0.1
    return round(max(0.5, min(1.5, current + delta)), 2)


def apply_cascade(base_score: float, cascade: float) -> float:
    """Apply cascade multiplier to base score, clamped to [0.0, 1.0]."""
    return round(_clamp(base_score * cascade), 4)