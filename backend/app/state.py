"""
Learner-state manager for the learner-state-aware pedagogical RAG tutor.

This module creates and updates compact learner-state records for simulated
multi-turn algebra tutoring episodes.

The state layer is research evidence. It supports:
- learner-state continuity
- misconception tracking
- hint counting
- support-level transitions
- reproducible state-transition logs

This module does not call an LLM.
This module does not call retrieval.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4


LEARNER_STATE_SCHEMA_VERSION = "v1.0"
DEFAULT_TASK_BANK_VERSION = "v1.0"
DEFAULT_POLICY_VERSION = "pedagogical_policy_v1.0"


SUPPORT_LEVELS = {
    "NONE": "none",
    "CONCEPTUAL_PROMPT": "conceptual_prompt",
    "LIGHT_HINT": "light_hint",
    "PROCEDURAL_HINT": "procedural_hint",
    "MISCONCEPTION_CORRECTION": "misconception_correction",
    "WORKED_STEP_EXPLANATION": "worked_step_explanation",
    "FULL_SOLUTION": "full_solution",
}


KNOWN_ERROR_PATTERNS = {
    "6x": "incorrect_coefficient_addition",
    "4a + 10": "distribution_or_combining_error",
    "12y - 10": "negative_distribution_sign_error",
    "5m and 9": "combining_unlike_terms",
    "added 7": "wrong_inverse_operation",
    "divided 11 by 2": "incorrect_order_of_inverse_operations",
    "y = 17": "failure_to_isolate_variable",
    "3x - 2 = 12": "incomplete_distribution",
    "23 + 5": "concatenation_instead_of_multiplication",
    "5 - 4 first": "order_of_operations_error",
    "i got -5": "subtraction_sign_error",
    "2x + y": "parentheses_or_partial_distribution_error",
    "moving 5 across": "move_across_sign_error",
    "sign changes": "move_across_sign_error"
}

@dataclass(frozen=True)
class StateTransition:
    transition_id: str
    session_id: str
    task_id: str
    turn_number: int
    learner_state_before: Optional[Dict[str, Any]]
    learner_state_after: Dict[str, Any]
    transition_reason: str
    created_at: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.lower().strip().split())


def detect_error_type(
    learner_message: str,
    task_metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Deterministic misconception detector for benchmark simulation.

    For controlled research use, this intentionally uses transparent rules.
    Later, this can be replaced or supplemented by a classifier.
    """
    normalized = normalize_text(learner_message)

    for pattern, error_type in KNOWN_ERROR_PATTERNS.items():
        if pattern.lower() in normalized:
            return error_type

    if task_metadata:
        tag = task_metadata.get("misconception_tag")
        if tag and any(marker in normalized for marker in ["i got", "is the answer", "why", "i think"]):
            return tag

    return None


def build_initial_learner_state(
    session_id: str,
    task: Dict[str, Any],
    turn_number: int = 1,
    learner_message: Optional[str] = None
) -> Dict[str, Any]:
    now = utc_now_iso()

    task_id = task["task_id"]

    return {
        "state_id": f"state_{uuid4().hex}",
        "session_id": session_id,
        "task_id": task_id,
        "turn_number": turn_number,
        "problem_id": task_id,
        "topic": task.get("topic") or task.get("category_label") or task.get("category"),
        "subskill": task.get("subskill") or "",
        "recent_answer_attempt": learner_message,
        "detected_error_type": detect_error_type(learner_message or "", task),
        "hint_count": 0,
        "support_level": SUPPORT_LEVELS["NONE"],
        "interaction_summary": build_interaction_summary(
            task=task,
            learner_message=learner_message,
            detected_error_type=detect_error_type(learner_message or "", task),
            selected_mode=None
        ),
        "last_pedagogical_mode": None,
        "state_created_at": now,
        "state_updated_at": now,
        "versioning": {
            "task_bank_version": DEFAULT_TASK_BANK_VERSION,
            "policy_version": DEFAULT_POLICY_VERSION,
            "learner_state_schema_version": LEARNER_STATE_SCHEMA_VERSION
        }
    }


def build_interaction_summary(
    task: Dict[str, Any],
    learner_message: Optional[str],
    detected_error_type: Optional[str],
    selected_mode: Optional[str]
) -> str:
    parts = [
        f"Task {task.get('task_id')} targets {task.get('topic') or task.get('category_label')} / {task.get('subskill')}."
    ]

    if learner_message:
        parts.append(f"Learner message: {learner_message}")

    if detected_error_type:
        parts.append(f"Detected error type: {detected_error_type}.")

    if selected_mode:
        parts.append(f"Most recent pedagogical mode: {selected_mode}.")

    return " ".join(parts)


def should_increment_hint_count(selected_mode: Optional[str]) -> bool:
    return selected_mode in {
        SUPPORT_LEVELS["CONCEPTUAL_PROMPT"],
        SUPPORT_LEVELS["LIGHT_HINT"],
        SUPPORT_LEVELS["PROCEDURAL_HINT"],
        SUPPORT_LEVELS["MISCONCEPTION_CORRECTION"],
        SUPPORT_LEVELS["WORKED_STEP_EXPLANATION"]
    }


def update_learner_state(
    previous_state: Optional[Dict[str, Any]],
    task: Dict[str, Any],
    learner_message: str,
    selected_pedagogical_mode: Optional[str],
    turn_number: int,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create or update learner state for a tutoring turn.
    """
    if previous_state is None:
        state = build_initial_learner_state(
            session_id=session_id or f"session_{uuid4().hex}",
            task=task,
            turn_number=turn_number,
            learner_message=learner_message
        )
    else:
        state = deepcopy(previous_state)

    detected_error_type = detect_error_type(learner_message, task)

    current_hint_count = int(state.get("hint_count", 0))
    if should_increment_hint_count(selected_pedagogical_mode):
        current_hint_count += 1

    state["turn_number"] = turn_number
    state["recent_answer_attempt"] = learner_message
    state["detected_error_type"] = detected_error_type
    state["hint_count"] = current_hint_count
    state["support_level"] = selected_pedagogical_mode or state.get("support_level") or SUPPORT_LEVELS["NONE"]
    state["last_pedagogical_mode"] = selected_pedagogical_mode
    state["interaction_summary"] = build_interaction_summary(
        task=task,
        learner_message=learner_message,
        detected_error_type=detected_error_type,
        selected_mode=selected_pedagogical_mode
    )
    state["state_updated_at"] = utc_now_iso()

    return state


def create_state_transition(
    previous_state: Optional[Dict[str, Any]],
    updated_state: Dict[str, Any],
    transition_reason: str = "learner_message_processed"
) -> Dict[str, Any]:
    transition = StateTransition(
        transition_id=f"transition_{uuid4().hex}",
        session_id=updated_state["session_id"],
        task_id=updated_state["task_id"],
        turn_number=updated_state["turn_number"],
        learner_state_before=previous_state,
        learner_state_after=updated_state,
        transition_reason=transition_reason,
        created_at=utc_now_iso()
    )

    return asdict(transition)


if __name__ == "__main__":
    sample_task = {
        "task_id": "LE-04",
        "category": "solving_linear_equations",
        "category_label": "Solving linear equations",
        "student_facing_prompt": "Solve for x: 3(x - 2) = 12.",
        "topic": "solving linear equations",
        "subskill": "distribution in equations",
        "difficulty": "medium",
        "misconception_tag": "incomplete_distribution"
    }

    before = None

    after = update_learner_state(
        previous_state=before,
        task=sample_task,
        learner_message="I distributed and got 3x - 2 = 12.",
        selected_pedagogical_mode="misconception_correction",
        turn_number=2,
        session_id="session_demo_LE_04"
    )

    transition = create_state_transition(
        previous_state=before,
        updated_state=after
    )

    import json
    print(json.dumps(transition, indent=2))