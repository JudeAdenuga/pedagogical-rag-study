#!/usr/bin/env python3
"""
Live two-turn Gemini File Search/RAG runner for the pedagogical RAG study.

This file is additive only. It does not modify any existing project file.

Study condition:
- Same foundation model as baseline: Gemini 2.5 Flash.
- Difference from baseline:
  1. Gemini File Search retrieves instructional algebra evidence.
  2. Retrieved chunks are injected into the existing pedagogical RAG prompt.
  3. Existing learner-state tracking is used.
  4. Existing pedagogical policy mode selection is used.

Episode structure:
- Turn 1: original benchmark student-facing algebra task.
- Turn 2: simulated learner follow-up turn.

Outputs:
- Complete two-turn RAG episode log.
- Turn 1 and Turn 2 retrieval logs.
- Matched baseline/RAG JSON export when corresponding baseline evidence exists.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from logging_service import (
    create_retrieval_log,
    save_json_export,
    save_retrieval_log,
)
from policy import select_mode_from_state
from prompts import build_pedagogical_rag_prompt
from rag_file_search_service import run_retrieval_probe
from rag_gemini_service import generate_rag_tutor_response
from state import create_state_transition, update_learner_state
from tasks import get_task_by_id, list_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_EVIDENCE_DIR = (
    PROJECT_ROOT / "manuscript_artifacts" / "evidence_logs" / "baseline"
)

RAG_EVIDENCE_DIR = (
    PROJECT_ROOT / "manuscript_artifacts" / "evidence_logs" / "rag"
)

DEFAULT_MODEL = "gemini-2.5-flash"
RAG_TWO_TURN_LIVE_VERSION = "rag_two_turn_live_v1.0"


class RagTwoTurnLiveError(Exception):
    """Raised when a live RAG two-turn episode cannot complete."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def redact_secret(value: Optional[str]) -> str:
    if not value:
        return "missing"
    return "present_redacted"


def get_model_name() -> str:
    load_project_env()
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def get_task_field(task: Dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = task.get(name)
        if value is not None:
            return str(value)
    return default


def normalize_task_id_for_filename(task_id: str) -> str:
    return str(task_id).replace("-", "_")


def build_task_summary(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": get_task_field(task, "task_id"),
        "algebra_category": get_task_field(
            task,
            "algebra_category",
            "category_label",
            "category",
        ),
        "student_prompt": get_task_field(
            task,
            "student_prompt",
            "student_facing_prompt",
            "prompt",
        ),
        "expected_answer_or_reasoning_target": get_task_field(
            task,
            "expected_answer",
            "expected_answer_or_reasoning_target",
            "expected_reasoning",
            "reasoning_target",
        ),
        "simulated_follow_up_turn": get_task_field(
            task,
            "simulated_follow_up_turn",
            "follow_up",
            "simulated_followup",
        ),
        "scoring_notes": get_task_field(task, "scoring_notes"),
    }


def build_retrieval_query(
    task: Dict[str, Any],
    current_learner_message: str,
    selected_pedagogical_mode: str,
    turn_number: int,
) -> str:
    """
    Build a retrieval query without exposing expected answers or scoring notes.
    """
    task_id = get_task_field(task, "task_id")
    category = get_task_field(
        task,
        "category_label",
        "algebra_category",
        "category",
    )
    topic = get_task_field(task, "topic")
    subskill = get_task_field(task, "subskill")
    student_prompt = get_task_field(
        task,
        "student_facing_prompt",
        "student_prompt",
        "prompt",
    )

    return f"""Retrieve the most relevant algebra instructional evidence for this pedagogical RAG tutoring turn.

Task ID: {task_id}
Turn number: {turn_number}
Algebra category: {category}
Topic: {topic}
Subskill: {subskill}
Original student-facing task: {student_prompt}
Current learner message: {current_learner_message}
Selected pedagogical response mode: {selected_pedagogical_mode}

Prioritize evidence that supports:
- mathematically accurate explanation,
- misconception correction where relevant,
- stepwise tutoring support,
- the selected pedagogical response mode.
"""


def build_response_object(response_text: str) -> Dict[str, Any]:
    return {
        "response_id": f"gemini_rag_response_{uuid.uuid4().hex}",
        "response_text": response_text,
        "is_mock_response": False,
    }


def save_complete_rag_episode_log(log: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def locate_matching_baseline_log(task_id: str) -> Optional[Path]:
    candidate = (
        BASELINE_EVIDENCE_DIR
        / f"baseline_episode_{normalize_task_id_for_filename(task_id)}_two_turn.json"
    )

    if candidate.exists():
        return candidate

    return None


def run_live_rag_two_turn_episode(
    task_id: str,
) -> Dict[str, Any]:
    task = get_task_by_id(task_id)
    model_name = get_model_name()
    session_id = f"rag_two_turn_{task_id}"

    # ------------------------------------------------------------------
    # TURN 1: original benchmark task
    # ------------------------------------------------------------------
    turn_1_message = get_task_field(
        task,
        "student_facing_prompt",
        "student_prompt",
        "prompt",
    )

    turn_1_state_before = None

    turn_1_provisional_state = update_learner_state(
        previous_state=turn_1_state_before,
        task=task,
        learner_message=turn_1_message,
        selected_pedagogical_mode=None,
        turn_number=1,
        session_id=session_id,
    )

    turn_1_policy_decision = select_mode_from_state(
        session_id=session_id,
        task_id=task_id,
        turn_number=1,
        learner_state=turn_1_provisional_state,
        current_learner_message=turn_1_message,
        task_metadata=task,
    )

    # Turn 1 is the initial student-facing benchmark request, not a failed learner attempt.
    # For the evaluation design, the RAG tutor should answer the task directly while
    # preserving teacher-like scaffolding and stepwise explanation.
    turn_1_policy_selected_mode = turn_1_policy_decision["selected_mode"]
    turn_1_selected_mode = "worked_step_explanation"

    turn_1_policy_decision = {
        **turn_1_policy_decision,
        "selected_mode_before_turn_1_evaluation_override": turn_1_policy_selected_mode,
        "selected_mode": turn_1_selected_mode,
        "turn_1_evaluation_override_applied": True,
        "turn_1_evaluation_override_reason": (
            "Turn 1 is the initial benchmark tutoring request. "
            "The tutor should provide an instructionally complete answer "
            "while retaining scaffolded, teacher-like explanation."
        ),
    }

    turn_1_state_after = update_learner_state(
        previous_state=turn_1_state_before,
        task=task,
        learner_message=turn_1_message,
        selected_pedagogical_mode=turn_1_selected_mode,
        turn_number=1,
        session_id=session_id,
    )

    turn_1_state_transition = create_state_transition(
        previous_state=turn_1_state_before,
        updated_state=turn_1_state_after,
        transition_reason="rag_turn_1_original_task_processed",
    )

    turn_1_retrieval_query = build_retrieval_query(
        task=task,
        current_learner_message=turn_1_message,
        selected_pedagogical_mode=turn_1_selected_mode,
        turn_number=1,
    )

    turn_1_retrieval_result = run_retrieval_probe(
        query_text=turn_1_retrieval_query,
        retrieval_model=model_name,
    )

    turn_1_retrieved_chunks = turn_1_retrieval_result["retrieved_chunks"]

    turn_1_retrieval_log = create_retrieval_log(
        session_id=session_id,
        task_id=task_id,
        turn_number=1,
        query_text=turn_1_retrieval_query,
        retrieved_chunks=turn_1_retrieved_chunks,
        retrieval_mode="gemini_file_search_retrieval_probe",
        metadata_filters={},
    )

    turn_1_retrieval_log["retrieval_probe"] = turn_1_retrieval_result

    turn_1_retrieval_log_path = save_retrieval_log(
        turn_1_retrieval_log,
        filename=(
            f"rag_episode_{normalize_task_id_for_filename(task_id)}"
            "_turn_1_retrieval.json"
        ),
    )

    turn_1_rag_prompt = build_pedagogical_rag_prompt(
        task=task,
        current_learner_message=turn_1_message,
        learner_state=turn_1_state_after,
        pedagogical_mode=turn_1_selected_mode,
        retrieved_chunks=turn_1_retrieved_chunks,
    )

    turn_1_response_text = generate_rag_tutor_response(
        prompt_text=turn_1_rag_prompt["prompt_text"],
        model_name=model_name,
    )

    turn_1_response = build_response_object(turn_1_response_text)

    # ------------------------------------------------------------------
    # TURN 2: simulated learner follow-up
    # ------------------------------------------------------------------
    turn_2_message = get_task_field(
        task,
        "simulated_follow_up_turn",
        "follow_up",
        "simulated_followup",
    )

    turn_2_state_before = turn_1_state_after

    turn_2_provisional_state = update_learner_state(
        previous_state=turn_2_state_before,
        task=task,
        learner_message=turn_2_message,
        selected_pedagogical_mode=None,
        turn_number=2,
        session_id=session_id,
    )

    turn_2_policy_decision = select_mode_from_state(
        session_id=session_id,
        task_id=task_id,
        turn_number=2,
        learner_state=turn_2_provisional_state,
        current_learner_message=turn_2_message,
        task_metadata=task,
    )

    turn_2_selected_mode = turn_2_policy_decision["selected_mode"]

    turn_2_state_after = update_learner_state(
        previous_state=turn_2_state_before,
        task=task,
        learner_message=turn_2_message,
        selected_pedagogical_mode=turn_2_selected_mode,
        turn_number=2,
        session_id=session_id,
    )

    turn_2_state_transition = create_state_transition(
        previous_state=turn_2_state_before,
        updated_state=turn_2_state_after,
        transition_reason="rag_turn_2_simulated_follow_up_processed",
    )

    turn_2_retrieval_query = build_retrieval_query(
        task=task,
        current_learner_message=turn_2_message,
        selected_pedagogical_mode=turn_2_selected_mode,
        turn_number=2,
    )

    turn_2_retrieval_result = run_retrieval_probe(
        query_text=turn_2_retrieval_query,
        retrieval_model=model_name,
    )

    turn_2_retrieved_chunks = turn_2_retrieval_result["retrieved_chunks"]

    turn_2_retrieval_log = create_retrieval_log(
        session_id=session_id,
        task_id=task_id,
        turn_number=2,
        query_text=turn_2_retrieval_query,
        retrieved_chunks=turn_2_retrieved_chunks,
        retrieval_mode="gemini_file_search_retrieval_probe",
        metadata_filters={},
    )

    turn_2_retrieval_log["retrieval_probe"] = turn_2_retrieval_result

    turn_2_retrieval_log_path = save_retrieval_log(
        turn_2_retrieval_log,
        filename=(
            f"rag_episode_{normalize_task_id_for_filename(task_id)}"
            "_turn_2_retrieval.json"
        ),
    )

    turn_2_rag_prompt = build_pedagogical_rag_prompt(
        task=task,
        current_learner_message=turn_2_message,
        learner_state=turn_2_state_after,
        pedagogical_mode=turn_2_selected_mode,
        retrieved_chunks=turn_2_retrieved_chunks,
    )

    turn_2_response_text = generate_rag_tutor_response(
        prompt_text=turn_2_rag_prompt["prompt_text"],
        model_name=model_name,
    )

    turn_2_response = build_response_object(turn_2_response_text)

    # ------------------------------------------------------------------
    # COMPLETE TWO-TURN RAG EPISODE LOG
    # ------------------------------------------------------------------
    rag_episode_log = {
        "evaluation_log_id": f"rag_two_turn_{uuid.uuid4().hex}",
        "condition": "pedagogical_rag_file_search",
        "episode_type": "two_turn_simulated_follow_up",
        "task_id": task_id,
        "session_id": session_id,
        "created_at": utc_now_iso(),
        "runner_version": RAG_TWO_TURN_LIVE_VERSION,
        "model": {
            "provider": "Google",
            "model_id": model_name,
            "api_surface": "Gemini Developer API",
        },
        "rag_configuration": {
            "retrieval_used": True,
            "gemini_file_search_used": True,
            "retrieval_probe_then_prompt_enrichment": True,
            "retrieved_context_passed_to_final_prompt": True,
            "learner_state_tracking_used": True,
            "pedagogical_policy_used": True,
            "same_foundation_model_as_baseline": True,
        },
        "task": build_task_summary(task),
        "turns": [
            {
                "turn_number": 1,
                "turn_type": "initial_task_response",
                "current_learner_message": turn_1_message,
                "selected_pedagogical_mode": turn_1_selected_mode,
                "policy_decision": turn_1_policy_decision,
                "learner_state_before": turn_1_state_before,
                "learner_state_after": turn_1_state_after,
                "state_transition": turn_1_state_transition,
                "retrieval_query_text": turn_1_retrieval_query,
                "retrieval_log_path": str(turn_1_retrieval_log_path),
                "retrieved_chunk_count": len(turn_1_retrieved_chunks),
                "retrieved_chunks": turn_1_retrieved_chunks,
                "prompt_template_version": turn_1_rag_prompt[
                    "prompt_template_version"
                ],
                "prompt_text": turn_1_rag_prompt["prompt_text"],
                "response": turn_1_response,
            },
            {
                "turn_number": 2,
                "turn_type": "simulated_follow_up_response",
                "current_learner_message": turn_2_message,
                "selected_pedagogical_mode": turn_2_selected_mode,
                "policy_decision": turn_2_policy_decision,
                "learner_state_before": turn_2_state_before,
                "learner_state_after": turn_2_state_after,
                "state_transition": turn_2_state_transition,
                "retrieval_query_text": turn_2_retrieval_query,
                "retrieval_log_path": str(turn_2_retrieval_log_path),
                "retrieved_chunk_count": len(turn_2_retrieved_chunks),
                "retrieved_chunks": turn_2_retrieved_chunks,
                "prompt_template_version": turn_2_rag_prompt[
                    "prompt_template_version"
                ],
                "prompt_text": turn_2_rag_prompt["prompt_text"],
                "response": turn_2_response,
            },
        ],
        "environment": {
            "gemini_api_key": redact_secret(os.getenv("GEMINI_API_KEY")),
            "model_from_env": os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        },
        "evidence_use": {
            "spreadsheet_capture_candidate": True,
            "supplementary_candidate": True,
            "suggested_caption": (
                "Two-turn pedagogical RAG episode showing Gemini File Search "
                "retrieval, learner-state-aware pedagogical orchestration, and "
                "Gemini 2.5 Flash tutoring responses."
            ),
            "notes": (
                "This log supports the RAG condition. Retrieved instructional "
                "chunks were obtained through Gemini File Search and injected "
                "into the pedagogical RAG prompt before final response generation."
            ),
        },
    }

    rag_episode_log_path = (
        RAG_EVIDENCE_DIR
        / f"rag_episode_{normalize_task_id_for_filename(task_id)}_two_turn.json"
    )

    save_complete_rag_episode_log(
        rag_episode_log,
        rag_episode_log_path,
    )

    # ------------------------------------------------------------------
    # MATCHED BASELINE/RAG EXPORT
    # ------------------------------------------------------------------
    matched_export_path: Optional[str] = None
    matching_baseline_log_path = locate_matching_baseline_log(task_id)

    if matching_baseline_log_path:
        baseline_log = json.loads(
            matching_baseline_log_path.read_text(encoding="utf-8")
        )

        matched_export = {
            "matched_output_id": (
                f"matched_baseline_rag_"
                f"{normalize_task_id_for_filename(task_id)}"
            ),
            "task_id": task_id,
            "created_at": utc_now_iso(),
            "baseline_log_path": str(matching_baseline_log_path),
            "rag_log_path": str(rag_episode_log_path),
            "baseline": baseline_log,
            "pedagogical_rag": rag_episode_log,
        }

        path = save_json_export(
            matched_export,
            filename=(
                f"matched_baseline_rag_"
                f"{normalize_task_id_for_filename(task_id)}_two_turn.json"
            ),
        )

        matched_export_path = str(path)

    return {
        "status": "rag_two_turn_live_complete",
        "task_id": task_id,
        "session_id": session_id,
        "rag_log_path": str(rag_episode_log_path),
        "turn_1_retrieval_log_path": str(turn_1_retrieval_log_path),
        "turn_2_retrieval_log_path": str(turn_2_retrieval_log_path),
        "matched_export_path": matched_export_path,
        "turn_1_selected_pedagogical_mode": turn_1_selected_mode,
        "turn_2_selected_pedagogical_mode": turn_2_selected_mode,
        "turn_1_retrieved_chunk_count": len(turn_1_retrieved_chunks),
        "turn_2_retrieved_chunk_count": len(turn_2_retrieved_chunks),
        "retrieval_probe_calls_made": 2,
        "final_tutor_generation_calls_made": 2,
        "total_model_calls_made": 4,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live two-turn Gemini File Search/RAG tutoring episodes."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--task-id",
        help="Benchmark task ID, for example LE-02.",
    )

    group.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all 24 benchmark tasks.",
    )

    parser.add_argument(
        "--sleep-seconds-between-tasks",
        type=float,
        default=1.0,
        help="Pause between tasks during --all-tasks runs. Default: 1.0 second.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.task_id:
        result = run_live_rag_two_turn_episode(args.task_id)

        rag_log_path = Path(result["rag_log_path"])
        rag_log = json.loads(rag_log_path.read_text(encoding="utf-8"))

        turn_1_response = rag_log["turns"][0]["response"]["response_text"]
        turn_2_response = rag_log["turns"][1]["response"]["response_text"]

        print("")
        print("=" * 100)
        print(f'{result["task_id"]} — GEMINI FILE SEARCH/RAG INITIAL RESPONSE')
        print("=" * 100)
        print(turn_1_response)

        print("")
        print("=" * 100)
        print(f'{result["task_id"]} — GEMINI FILE SEARCH/RAG FOLLOW-UP RESPONSE')
        print("=" * 100)
        print(turn_2_response)

        print("")
        print("=" * 100)
        print(f'{result["task_id"]} — RAG RUN SUMMARY')
        print("=" * 100)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.all_tasks:
        results: List[Dict[str, Any]] = []

        for task in list_tasks():
            task_id = str(task["task_id"])
            result = run_live_rag_two_turn_episode(task_id)
            results.append(result)

            if args.sleep_seconds_between_tasks > 0:
                time.sleep(args.sleep_seconds_between_tasks)

        print(
            json.dumps(
                {
                    "status": "all_live_rag_two_turn_tasks_complete",
                    "total_tasks": len(results),
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    raise RagTwoTurnLiveError("No task-selection argument was provided.")


if __name__ == "__main__":
    main()
