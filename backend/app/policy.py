"""
Pedagogical policy engine for the learner-state-aware algebra RAG tutor.

This module selects the pedagogical response mode for each tutoring turn.
It is deterministic so the research prototype can produce auditable and
reproducible policy-decision logs.

This policy mirrors the study's bounded pedagogical policy:

- conceptual_prompt: initial attempt with no prior hint
- light_hint: first incorrect or partial attempt
- procedural_hint: repeated error without forward progress
- misconception_correction: learner response matches a tagged misconception pattern
- worked_step_explanation: after two unsuccessful attempts or explicit request for more help
- full_solution: after three unsuccessful attempts or direct request for the answer

The policy does not call an LLM.
The policy does not perform retrieval.
The policy only decides the instructional support mode based on:
- learner state
- turn number
- hint count / unsuccessful attempt count
- detected misconception/error type
- current learner message
- task metadata

Author: Jude A. Adenuga
Project: Pedagogical RAG Algebra Tutor Study
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


POLICY_VERSION = "pedagogical_policy_v1.1"


PEDAGOGICAL_MODES = {
    "CONCEPTUAL_PROMPT": "conceptual_prompt",
    "LIGHT_HINT": "light_hint",
    "PROCEDURAL_HINT": "procedural_hint",
    "MISCONCEPTION_CORRECTION": "misconception_correction",
    "WORKED_STEP_EXPLANATION": "worked_step_explanation",
    "FULL_SOLUTION": "full_solution",
}


ERROR_TYPES_REQUIRING_CORRECTION = {
    "incorrect_coefficient_addition",
    "distribution_or_combining_error",
    "distribution_over_subtraction_confusion",
    "negative_distribution_sign_error",
    "combining_unlike_terms",
    "wrong_inverse_operation",
    "incorrect_order_of_inverse_operations",
    "failure_to_isolate_variable",
    "incomplete_distribution",
    "balance_principle_confusion",
    "concatenation_instead_of_multiplication",
    "order_of_operations_error",
    "subtraction_sign_error",
    "parentheses_or_partial_distribution_error",
    "move_across_sign_error",
    "multi_step_error_analysis",
    "coefficient_meaning_confusion",
    "equivalence_confusion",
    "expression_structure_confusion",
    "negative_coefficient_error",
    "fixed_cost_confusion",
    "slope_intercept_confusion",
    "equivalent_expression_confusion",
}


DIRECT_ANSWER_REQUEST_PATTERNS = [
    "just give me the answer",
    "give me the answer",
    "what is the answer",
    "tell me the answer",
    "answer only",
    "final answer",
    "solve it for me",
]


MORE_HELP_REQUEST_PATTERNS = [
    "more help",
    "show a step",
    "show me a step",
    "show the steps",
    "can you show",
    "walk me through",
    "explain more",
    "i need more help",
    "i still need help",
    "help me more",
]


FULL_SOLUTION_AFTER_STRUGGLE_PATTERNS = [
    "i give up",
    "i don't know",
    "i do not know",
    "i am stuck",
    "i'm stuck",
    "stuck",
]


CONFUSION_PATTERNS = [
    "why",
    "confused",
    "i don't understand",
    "i do not understand",
    "how",
    "can you explain",
    "what does",
    "what is",
]


ATTEMPT_PATTERNS = [
    "i got",
    "is the answer",
    "my answer is",
    "i think",
    "would it be",
    "is it",
    "i wrote",
    "i did",
]


@dataclass(frozen=True)
class PolicyInput:
    session_id: str
    task_id: str
    turn_number: int
    learner_state: Dict[str, Any]
    current_learner_message: str
    task_metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PolicyDecision:
    policy_decision_id: str
    session_id: str
    task_id: str
    turn_number: int
    selected_mode: str
    reason_codes: List[str]
    input_state_summary: Dict[str, Any]
    policy_version: str
    created_at: str


def utc_now_iso() -> str:
    """Return timezone-aware UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Optional[str]) -> str:
    """Normalize text for deterministic matching."""
    if not value:
        return ""

    return " ".join(value.lower().strip().split())


def contains_any(text: str, patterns: List[str]) -> bool:
    """Return True when any pattern appears in the normalized text."""
    normalized = normalize_text(text)
    return any(pattern in normalized for pattern in patterns)


def safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_unsuccessful_attempt_count(state: Dict[str, Any]) -> int:
    """
    Prefer explicit unsuccessful_attempt_count when available.
    Fall back to hint_count because the current learner-state schema already
    records hint_count as the support progression variable.
    """
    if "unsuccessful_attempt_count" in state:
        return safe_int(state.get("unsuccessful_attempt_count"), 0)

    return safe_int(state.get("hint_count"), 0)


def summarize_policy_input(policy_input: PolicyInput) -> Dict[str, Any]:
    """Create a compact input summary for the policy decision log."""
    state = policy_input.learner_state or {}
    task_metadata = policy_input.task_metadata or {}

    return {
        "problem_id": state.get("problem_id"),
        "topic": state.get("topic") or task_metadata.get("topic"),
        "subskill": state.get("subskill") or task_metadata.get("subskill"),
        "recent_answer_attempt": state.get("recent_answer_attempt"),
        "detected_error_type": state.get("detected_error_type"),
        "hint_count": safe_int(state.get("hint_count"), 0),
        "unsuccessful_attempt_count": get_unsuccessful_attempt_count(state),
        "support_level": state.get("support_level"),
        "last_pedagogical_mode": state.get("last_pedagogical_mode"),
        "message_preview": normalize_text(policy_input.current_learner_message)[:180],
    }


def select_pedagogical_mode(policy_input: PolicyInput) -> PolicyDecision:
    """
    Select the response mode for a tutoring turn.

    Study-aligned policy ladder:
    1. Direct answer request -> full_solution.
    2. Three unsuccessful attempts -> full_solution.
    3. Struggle request after enough prior support -> full_solution.
    4. Explicit request for more help after a prior attempt -> worked_step_explanation.
    5. Two unsuccessful attempts -> worked_step_explanation.
    6. Tagged misconception -> misconception_correction.
    7. Repeated error without forward progress -> procedural_hint.
    8. First incorrect or partial attempt -> light_hint.
    9. Initial attempt with no prior hint -> conceptual_prompt.
    10. Default -> light_hint.
    """

    state = policy_input.learner_state or {}
    message = normalize_text(policy_input.current_learner_message)

    hint_count = safe_int(state.get("hint_count"), 0)
    unsuccessful_attempt_count = get_unsuccessful_attempt_count(state)

    detected_error_type = state.get("detected_error_type")
    last_mode = state.get("last_pedagogical_mode")
    support_level = state.get("support_level")

    reason_codes: List[str] = []

    direct_answer_request = contains_any(message, DIRECT_ANSWER_REQUEST_PATTERNS)
    more_help_request = contains_any(message, MORE_HELP_REQUEST_PATTERNS)
    struggle_request = contains_any(message, FULL_SOLUTION_AFTER_STRUGGLE_PATTERNS)
    shows_confusion = contains_any(message, CONFUSION_PATTERNS)
    gives_attempt = contains_any(message, ATTEMPT_PATTERNS)

    if direct_answer_request:
        selected_mode = PEDAGOGICAL_MODES["FULL_SOLUTION"]
        reason_codes.extend(
            [
                "direct_learner_request_for_answer",
                "study_rule_full_solution_on_direct_answer_request",
            ]
        )

    elif unsuccessful_attempt_count >= 3:
        selected_mode = PEDAGOGICAL_MODES["FULL_SOLUTION"]
        reason_codes.extend(
            [
                "three_unsuccessful_attempts",
                "study_rule_full_solution_after_three_unsuccessful_attempts",
            ]
        )

    elif struggle_request and unsuccessful_attempt_count >= 2:
        selected_mode = PEDAGOGICAL_MODES["FULL_SOLUTION"]
        reason_codes.extend(
            [
                "learner_expressed_sustained_struggle",
                "support_already_escalated",
                "full_solution_allowed_after_sustained_struggle",
            ]
        )

    elif more_help_request and unsuccessful_attempt_count >= 1:
        selected_mode = PEDAGOGICAL_MODES["WORKED_STEP_EXPLANATION"]
        reason_codes.extend(
            [
                "explicit_request_for_more_help_after_prior_attempt",
                "study_rule_worked_step_after_more_help_request",
            ]
        )

    elif unsuccessful_attempt_count >= 2:
        selected_mode = PEDAGOGICAL_MODES["WORKED_STEP_EXPLANATION"]
        reason_codes.extend(
            [
                "two_unsuccessful_attempts",
                "study_rule_worked_step_after_two_unsuccessful_attempts",
            ]
        )

    elif detected_error_type in ERROR_TYPES_REQUIRING_CORRECTION:
        selected_mode = PEDAGOGICAL_MODES["MISCONCEPTION_CORRECTION"]
        reason_codes.extend(
            [
                "tagged_misconception_detected",
                f"error_type:{detected_error_type}",
                "study_rule_misconception_correction_for_tagged_pattern",
            ]
        )

    elif gives_attempt and unsuccessful_attempt_count == 1:
        selected_mode = PEDAGOGICAL_MODES["PROCEDURAL_HINT"]
        reason_codes.extend(
            [
                "repeated_error_without_forward_progress",
                "study_rule_procedural_hint_after_repeated_error",
            ]
        )

    elif gives_attempt and unsuccessful_attempt_count == 0:
        selected_mode = PEDAGOGICAL_MODES["LIGHT_HINT"]
        reason_codes.extend(
            [
                "first_incorrect_or_partial_attempt",
                "study_rule_light_hint_after_first_incorrect_attempt",
            ]
        )

    elif shows_confusion and hint_count > 0:
        selected_mode = PEDAGOGICAL_MODES["PROCEDURAL_HINT"]
        reason_codes.extend(
            [
                "learner_shows_confusion",
                "prior_hint_exists",
                "procedural_support_needed",
            ]
        )

    elif hint_count == 0 and unsuccessful_attempt_count == 0:
        selected_mode = PEDAGOGICAL_MODES["CONCEPTUAL_PROMPT"]
        reason_codes.extend(
            [
                "initial_or_early_attempt",
                "no_prior_hint",
                "study_rule_conceptual_prompt_initial_attempt",
            ]
        )

    else:
        selected_mode = PEDAGOGICAL_MODES["LIGHT_HINT"]
        reason_codes.extend(
            [
                "default_light_hint",
                "preserve_productive_struggle",
            ]
        )

    if last_mode == selected_mode:
        reason_codes.append("same_mode_as_previous_turn")

    if support_level:
        reason_codes.append(f"prior_support_level:{support_level}")

    return PolicyDecision(
        policy_decision_id=f"policy_{uuid4().hex}",
        session_id=policy_input.session_id,
        task_id=policy_input.task_id,
        turn_number=policy_input.turn_number,
        selected_mode=selected_mode,
        reason_codes=reason_codes,
        input_state_summary=summarize_policy_input(policy_input),
        policy_version=POLICY_VERSION,
        created_at=utc_now_iso(),
    )


def policy_decision_to_dict(decision: PolicyDecision) -> Dict[str, Any]:
    """Convert a PolicyDecision dataclass to a JSON-serializable dictionary."""
    return asdict(decision)


def select_mode_from_state(
    session_id: str,
    task_id: str,
    turn_number: int,
    learner_state: Dict[str, Any],
    current_learner_message: str,
    task_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper used by the orchestration layer."""
    policy_input = PolicyInput(
        session_id=session_id,
        task_id=task_id,
        turn_number=turn_number,
        learner_state=learner_state,
        current_learner_message=current_learner_message,
        task_metadata=task_metadata,
    )

    decision = select_pedagogical_mode(policy_input)
    return policy_decision_to_dict(decision)


if __name__ == "__main__":
    import json

    examples = [
        {
            "label": "Initial attempt -> conceptual_prompt",
            "state": {
                "problem_id": "LE-04",
                "topic": "solving linear equations",
                "subskill": "distribution in equations",
                "recent_answer_attempt": None,
                "detected_error_type": None,
                "hint_count": 0,
                "unsuccessful_attempt_count": 0,
                "support_level": "none",
                "interaction_summary": "Initial turn.",
                "last_pedagogical_mode": None,
            },
            "message": "I am ready to start.",
            "expected_mode": "conceptual_prompt",
        },
        {
            "label": "First incorrect attempt -> light_hint",
            "state": {
                "problem_id": "SE-01",
                "topic": "simplifying expressions",
                "subskill": "combining like terms",
                "recent_answer_attempt": "Is the answer 6x?",
                "detected_error_type": None,
                "hint_count": 0,
                "unsuccessful_attempt_count": 0,
                "support_level": "none",
                "interaction_summary": "Learner gave a first answer attempt.",
                "last_pedagogical_mode": None,
            },
            "message": "Is the answer 6x?",
            "expected_mode": "light_hint",
        },
        {
            "label": "Repeated error without tagged misconception -> procedural_hint",
            "state": {
                "problem_id": "LE-02",
                "topic": "solving linear equations",
                "subskill": "two-step equations",
                "recent_answer_attempt": "I tried again but still got it wrong.",
                "detected_error_type": None,
                "hint_count": 1,
                "unsuccessful_attempt_count": 1,
                "support_level": "light_hint",
                "interaction_summary": "Learner has one prior unsuccessful attempt.",
                "last_pedagogical_mode": "light_hint",
            },
            "message": "I got another wrong answer.",
            "expected_mode": "procedural_hint",
        },
        {
            "label": "Tagged misconception -> misconception_correction",
            "state": {
                "problem_id": "LE-04",
                "topic": "solving linear equations",
                "subskill": "distribution in equations",
                "recent_answer_attempt": "I distributed and got 3x - 2 = 12.",
                "detected_error_type": "incomplete_distribution",
                "hint_count": 1,
                "unsuccessful_attempt_count": 1,
                "support_level": "light_hint",
                "interaction_summary": "Learner made an incomplete distribution error.",
                "last_pedagogical_mode": "light_hint",
            },
            "message": "I distributed and got 3x - 2 = 12.",
            "expected_mode": "misconception_correction",
        },
        {
            "label": "Two unsuccessful attempts -> worked_step_explanation",
            "state": {
                "problem_id": "LE-04",
                "topic": "solving linear equations",
                "subskill": "distribution in equations",
                "recent_answer_attempt": "I still got it wrong.",
                "detected_error_type": None,
                "hint_count": 2,
                "unsuccessful_attempt_count": 2,
                "support_level": "procedural_hint",
                "interaction_summary": "Learner has made two unsuccessful attempts.",
                "last_pedagogical_mode": "procedural_hint",
            },
            "message": "Can you show me a step?",
            "expected_mode": "worked_step_explanation",
        },
        {
            "label": "Three unsuccessful attempts -> full_solution",
            "state": {
                "problem_id": "LE-04",
                "topic": "solving linear equations",
                "subskill": "distribution in equations",
                "recent_answer_attempt": "I still cannot solve it.",
                "detected_error_type": None,
                "hint_count": 3,
                "unsuccessful_attempt_count": 3,
                "support_level": "worked_step_explanation",
                "interaction_summary": "Learner has made three unsuccessful attempts.",
                "last_pedagogical_mode": "worked_step_explanation",
            },
            "message": "I am still stuck.",
            "expected_mode": "full_solution",
        },
        {
            "label": "Direct answer request -> full_solution",
            "state": {
                "problem_id": "LE-04",
                "topic": "solving linear equations",
                "subskill": "distribution in equations",
                "recent_answer_attempt": "I am stuck.",
                "detected_error_type": None,
                "hint_count": 1,
                "unsuccessful_attempt_count": 1,
                "support_level": "light_hint",
                "interaction_summary": "Learner requested answer.",
                "last_pedagogical_mode": "light_hint",
            },
            "message": "Just give me the answer.",
            "expected_mode": "full_solution",
        },
    ]

    all_passed = True

    for example in examples:
        print("\n" + example["label"])
        print("-" * 80)

        decision = select_mode_from_state(
            session_id="session_demo_policy",
            task_id=example["state"]["problem_id"],
            turn_number=2,
            learner_state=example["state"],
            current_learner_message=example["message"],
            task_metadata={
                "topic": example["state"]["topic"],
                "subskill": example["state"]["subskill"],
            },
        )

        actual_mode = decision["selected_mode"]
        expected_mode = example["expected_mode"]
        passed = actual_mode == expected_mode

        if not passed:
            all_passed = False

        print(json.dumps(decision, indent=2))
        print(f"EXPECTED: {expected_mode}")
        print(f"ACTUAL:   {actual_mode}")
        print(f"PASS:     {passed}")

    print("\n" + "=" * 80)
    print(f"POLICY TESTS PASSED: {all_passed}")