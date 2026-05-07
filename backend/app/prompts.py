"""
Prompt assembly utilities for the baseline LLM tutor and the learner-state-aware
pedagogical RAG tutor.

This module is intentionally deterministic. It does not call an LLM, does not
perform retrieval, and does not update learner state. It only converts study
objects into auditable prompt text.

Author: Jude A. Adenuga
Project: Pedagogical RAG Algebra Tutor Study
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = PROJECT_ROOT / "backend" / "prompts"

BASELINE_PROMPT_TEMPLATE_PATH = PROMPT_DIR / "baseline_prompt_template.txt"
PEDAGOGICAL_RAG_PROMPT_TEMPLATE_PATH = PROMPT_DIR / "pedagogical_rag_prompt_template.txt"

BASELINE_PROMPT_VERSION = "baseline_prompt_v1.0"
PEDAGOGICAL_RAG_PROMPT_VERSION = "pedagogical_rag_prompt_v1.0"


class PromptAssemblyError(Exception):
    """Raised when a prompt template cannot be loaded or formatted."""


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise PromptAssemblyError(f"Prompt template not found: {path}")

    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise PromptAssemblyError(f"Prompt template is empty: {path}")

    return text


def safe_value(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def format_retrieved_chunks(retrieved_chunks: Optional[Iterable[Dict[str, Any]]]) -> str:
    chunks = list(retrieved_chunks or [])

    if not chunks:
        return "No retrieved instructional evidence was supplied for this turn."

    formatted: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}

        formatted.append(
            "\n".join(
                [
                    f"[Retrieved chunk {index}]",
                    f"Chunk ID: {safe_value(chunk.get('chunk_id'), 'unknown_chunk')}",
                    f"Title: {safe_value(chunk.get('title'), 'Untitled instructional chunk')}",
                    f"Score: {safe_value(chunk.get('score'), 'not_available')}",
                    f"Topic: {safe_value(metadata.get('topic') or chunk.get('topic'), 'not_available')}",
                    f"Subskill: {safe_value(metadata.get('subskill') or chunk.get('subskill'), 'not_available')}",
                    f"Resource type: {safe_value(metadata.get('resource_type') or chunk.get('resource_type'), 'not_available')}",
                    f"Misconception tag: {safe_value(metadata.get('misconception_tag') or chunk.get('misconception_tag'), 'not_available')}",
                    f"Content: {safe_value(chunk.get('content'), '')}",
                ]
            )
        )

    return "\n\n".join(formatted)


def extract_task_prompt(task: Dict[str, Any]) -> str:
    return safe_value(
        task.get("student_facing_prompt")
        or task.get("student_prompt")
        or task.get("prompt")
    )


def extract_algebra_category(task: Dict[str, Any]) -> str:
    return safe_value(
        task.get("category_label")
        or task.get("algebra_category")
        or task.get("category")
    )


def build_baseline_prompt(task: Dict[str, Any]) -> Dict[str, Any]:
    template = load_prompt_template(BASELINE_PROMPT_TEMPLATE_PATH)

    values = {
        "task_id": safe_value(task.get("task_id")),
        "algebra_category": extract_algebra_category(task),
        "student_prompt": extract_task_prompt(task),
        "simulated_follow_up_turn": safe_value(task.get("simulated_follow_up_turn")),
    }

    try:
        prompt_text = template.format(**values)
    except KeyError as exc:
        raise PromptAssemblyError(f"Missing placeholder for baseline prompt: {exc}") from exc

    return {
        "condition": "baseline",
        "prompt_template_version": BASELINE_PROMPT_VERSION,
        "prompt_template_path": str(BASELINE_PROMPT_TEMPLATE_PATH),
        "task_id": values["task_id"],
        "prompt_text": prompt_text,
        "template_values": values,
        "retrieval_enabled": False,
        "learner_state_enabled": False,
        "file_search_enabled": False,
    }


def build_pedagogical_rag_prompt(
    task: Dict[str, Any],
    current_learner_message: str,
    learner_state: Dict[str, Any],
    pedagogical_mode: str,
    retrieved_chunks: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    template = load_prompt_template(PEDAGOGICAL_RAG_PROMPT_TEMPLATE_PATH)
    retrieved_context = format_retrieved_chunks(retrieved_chunks)

    values = {
        "task_id": safe_value(task.get("task_id")),
        "algebra_category": extract_algebra_category(task),
        "student_prompt": extract_task_prompt(task),
        "current_learner_message": safe_value(current_learner_message),
        "retrieved_context_chunks": retrieved_context,
        "problem_id": safe_value(learner_state.get("problem_id") or task.get("task_id")),
        "topic": safe_value(learner_state.get("topic") or task.get("topic")),
        "subskill": safe_value(learner_state.get("subskill") or task.get("subskill")),
        "recent_answer_attempt": safe_value(learner_state.get("recent_answer_attempt"), "None"),
        "detected_error_type": safe_value(learner_state.get("detected_error_type"), "None"),
        "hint_count": safe_value(learner_state.get("hint_count"), "0"),
        "support_level": safe_value(learner_state.get("support_level"), "none"),
        "interaction_summary": safe_value(learner_state.get("interaction_summary"), ""),
        "pedagogical_mode": safe_value(pedagogical_mode),
    }

    try:
        prompt_text = template.format(**values)
    except KeyError as exc:
        raise PromptAssemblyError(f"Missing placeholder for pedagogical RAG prompt: {exc}") from exc

    return {
        "condition": "pedagogical_rag",
        "prompt_template_version": PEDAGOGICAL_RAG_PROMPT_VERSION,
        "prompt_template_path": str(PEDAGOGICAL_RAG_PROMPT_TEMPLATE_PATH),
        "task_id": values["task_id"],
        "prompt_text": prompt_text,
        "template_values": values,
        "retrieval_enabled": True,
        "learner_state_enabled": True,
        "file_search_enabled": False,
    }


if __name__ == "__main__":
    import json

    from tasks import get_task_by_id
    from state import build_initial_learner_state
    from policy import select_mode_from_state

    task = get_task_by_id("LE-04")
    session_id = "session_demo_prompts"
    learner_message = 'Student says: "I distributed and got 3x - 2 = 12."'

    initial_state = build_initial_learner_state(
        session_id=session_id,
        task=task,
        turn_number=1,
        learner_message=learner_message,
    )

    decision = select_mode_from_state(
        session_id=session_id,
        task_id=task["task_id"],
        turn_number=1,
        learner_state=initial_state,
        current_learner_message=learner_message,
        task_metadata=task,
    )

    sample_chunks = [
        {
            "chunk_id": "ALG-LE-005",
            "title": "Distribution in equations",
            "score": 0.91,
            "content": "In 3(x - 2) = 12, distributing means multiplying 3 by every term inside the parentheses: 3x - 6, not 3x - 2.",
            "metadata": {
                "topic": "solving linear equations",
                "subskill": "distribution in equations",
                "resource_type": "worked_example",
                "misconception_tag": "incomplete_distribution"
            }
        }
    ]

    baseline_prompt = build_baseline_prompt(task)
    rag_prompt = build_pedagogical_rag_prompt(
        task=task,
        current_learner_message=learner_message,
        learner_state=initial_state,
        pedagogical_mode=decision["selected_mode"],
        retrieved_chunks=sample_chunks,
    )

    print(json.dumps(
        {
            "baseline_prompt_preview": baseline_prompt["prompt_text"][:800],
            "rag_prompt_preview": rag_prompt["prompt_text"][:1200]
        },
        indent=2
    ))
