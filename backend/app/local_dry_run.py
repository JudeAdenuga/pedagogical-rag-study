"""
Local matched dry-run runner for the pedagogical RAG algebra tutor study.

This module runs one matched baseline/RAG evaluation episode without calling:
- Gemini
- Claude
- Bedrock
- AWS
- external retrieval services

Purpose:
- prove local orchestration works
- assemble baseline prompt
- assemble pedagogical RAG prompt
- simulate retrieval from the local corpus seed
- update learner state
- select pedagogical policy mode
- write structured baseline, RAG, retrieval, and matched-output logs

This is Step 9 in the project charter:
Run local dry-run with mock outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from logging_service import (
    build_mock_model_response,
    create_evaluation_log,
    create_retrieval_log,
    save_evaluation_log,
    save_json_export,
    save_retrieval_log,
    utc_now_iso,
)
from policy import select_mode_from_state
from prompts import build_baseline_prompt, build_pedagogical_rag_prompt
from state import create_state_transition, update_learner_state
from tasks import get_task_by_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "backend" / "data" / "algebra_corpus_seed.json"

LOCAL_DRY_RUN_VERSION = "local_dry_run_v1.0"


class LocalDryRunError(Exception):
    """Raised when the local dry-run cannot complete."""


def load_corpus_chunks() -> List[Dict[str, Any]]:
    """Load local instructional corpus chunks from algebra_corpus_seed.json."""
    if not CORPUS_PATH.exists():
        raise LocalDryRunError(f"Corpus file not found: {CORPUS_PATH}")

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        corpus = json.load(file)

    chunks = corpus.get("chunks")

    if not isinstance(chunks, list):
        raise LocalDryRunError("Corpus file must contain a top-level 'chunks' array.")

    return chunks


def score_chunk_for_task(
    chunk: Dict[str, Any],
    task: Dict[str, Any],
    pedagogical_mode: Optional[str] = None,
) -> float:
    """
    Deterministically score a local corpus chunk for a task.

    This is mock retrieval for local development only.
    It is intentionally transparent and reproducible.
    """
    score = 0.0

    task_id = task.get("task_id")
    topic = task.get("topic")
    subskill = task.get("subskill")
    misconception_tag = task.get("misconception_tag")

    if task_id in chunk.get("supports_tasks", []):
        score += 0.55

    if topic and chunk.get("topic") == topic:
        score += 0.15

    if subskill and chunk.get("subskill") == subskill:
        score += 0.15

    if misconception_tag and chunk.get("misconception_tag") == misconception_tag:
        score += 0.10

    if pedagogical_mode and pedagogical_mode in chunk.get("pedagogical_use", []):
        score += 0.05

    return round(score, 4)


def mock_retrieve_chunks(
    task: Dict[str, Any],
    pedagogical_mode: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return top local corpus chunks for a task.

    Output shape is compatible with prompt assembly and retrieval logging.
    """
    chunks = load_corpus_chunks()

    scored_chunks = []

    for chunk in chunks:
        score = score_chunk_for_task(
            chunk=chunk,
            task=task,
            pedagogical_mode=pedagogical_mode,
        )

        if score <= 0:
            continue

        scored_chunks.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title"),
                "score": score,
                "content": chunk.get("content"),
                "metadata": {
                    "topic": chunk.get("topic"),
                    "subskill": chunk.get("subskill"),
                    "difficulty": chunk.get("difficulty"),
                    "resource_type": chunk.get("resource_type"),
                    "misconception_tag": chunk.get("misconception_tag"),
                    "supports_tasks": chunk.get("supports_tasks", []),
                    "pedagogical_use": chunk.get("pedagogical_use", []),
                    "corpus_version": "algebra_corpus_v1.0",
                },
            }
        )

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]


def build_request(
    condition: str,
    session_id: str,
    task_id: str,
    turn_number: int,
    current_learner_message: str,
) -> Dict[str, Any]:
    """Build a local tutor request object."""
    return {
        "request_id": f"local_request_{condition}_{task_id}_turn_{turn_number}",
        "condition": condition,
        "session_id": session_id,
        "task_id": task_id,
        "turn_number": turn_number,
        "current_learner_message": current_learner_message,
        "dry_run": True,
        "created_at": utc_now_iso(),
        "local_dry_run_version": LOCAL_DRY_RUN_VERSION,
    }


def run_local_dry_run(task_id: str, turn_number: int = 1, top_k: int = 5) -> Dict[str, Any]:
    """
    Run one matched local baseline/RAG dry-run episode.
    """
    task = get_task_by_id(task_id)
    session_id = f"local_dry_run_{task_id}"

    current_learner_message = task.get("simulated_follow_up_turn") or task.get("student_facing_prompt")

    baseline_request = build_request(
        condition="baseline",
        session_id=session_id,
        task_id=task_id,
        turn_number=turn_number,
        current_learner_message=current_learner_message,
    )

    baseline_prompt = build_baseline_prompt(task)
    baseline_response = build_mock_model_response(
        condition="baseline",
        task_id=task_id,
    )

    baseline_log = create_evaluation_log(
        condition="baseline",
        session_id=session_id,
        task_id=task_id,
        turn_number=turn_number,
        request=baseline_request,
        response=baseline_response,
        prompt_text=baseline_prompt["prompt_text"],
        model_id="local_mock_no_model_call",
        learner_state_before=None,
        learner_state_after=None,
        policy_decision=None,
        retrieval_log=None,
    )

    baseline_log_path = save_evaluation_log(
        baseline_log,
        filename=f"local_dry_run_{task_id}_baseline.json",
    )

    learner_state_before = None

    provisional_state = update_learner_state(
        previous_state=learner_state_before,
        task=task,
        learner_message=current_learner_message,
        selected_pedagogical_mode=None,
        turn_number=turn_number,
        session_id=session_id,
    )

    policy_decision = select_mode_from_state(
        session_id=session_id,
        task_id=task_id,
        turn_number=turn_number,
        learner_state=provisional_state,
        current_learner_message=current_learner_message,
        task_metadata=task,
    )

    selected_mode = policy_decision["selected_mode"]

    learner_state_after = update_learner_state(
        previous_state=learner_state_before,
        task=task,
        learner_message=current_learner_message,
        selected_pedagogical_mode=selected_mode,
        turn_number=turn_number,
        session_id=session_id,
    )

    state_transition = create_state_transition(
        previous_state=learner_state_before,
        updated_state=learner_state_after,
        transition_reason="local_dry_run_learner_message_processed",
    )

    retrieved_chunks = mock_retrieve_chunks(
        task=task,
        pedagogical_mode=selected_mode,
        top_k=top_k,
    )

    retrieval_log = create_retrieval_log(
        session_id=session_id,
        task_id=task_id,
        turn_number=turn_number,
        query_text=current_learner_message,
        retrieved_chunks=retrieved_chunks,
        retrieval_mode="mock",
        top_k=top_k,
        metadata_filters={
            "topic": task.get("topic"),
            "subskill": task.get("subskill"),
            "misconception_tag": task.get("misconception_tag"),
        },
    )

    retrieval_log_path = save_retrieval_log(
        retrieval_log,
        filename=f"local_dry_run_{task_id}_retrieval.json",
    )

    rag_request = build_request(
        condition="pedagogical_rag",
        session_id=session_id,
        task_id=task_id,
        turn_number=turn_number,
        current_learner_message=current_learner_message,
    )

    rag_prompt = build_pedagogical_rag_prompt(
        task=task,
        current_learner_message=current_learner_message,
        learner_state=learner_state_after,
        pedagogical_mode=selected_mode,
        retrieved_chunks=retrieved_chunks,
    )

    rag_response = build_mock_model_response(
        condition="pedagogical_rag",
        task_id=task_id,
        selected_pedagogical_mode=selected_mode,
    )

    rag_log = create_evaluation_log(
        condition="pedagogical_rag",
        session_id=session_id,
        task_id=task_id,
        turn_number=turn_number,
        request=rag_request,
        response=rag_response,
        prompt_text=rag_prompt["prompt_text"],
        model_id="local_mock_no_model_call",
        learner_state_before=learner_state_before,
        learner_state_after=learner_state_after,
        policy_decision=policy_decision,
        retrieval_log=retrieval_log,
    )

    rag_log_path = save_evaluation_log(
        rag_log,
        filename=f"local_dry_run_{task_id}_rag.json",
    )

    matched_output = {
        "matched_output_id": f"matched_local_dry_run_{task_id}",
        "local_dry_run_version": LOCAL_DRY_RUN_VERSION,
        "session_id": session_id,
        "task_id": task_id,
        "turn_number": turn_number,
        "dry_run": True,
        "model_calls_made": False,
        "external_services_called": False,
        "task": task,
        "baseline": {
            "request": baseline_request,
            "prompt_template_version": baseline_prompt["prompt_template_version"],
            "prompt_text": baseline_prompt["prompt_text"],
            "response": baseline_response,
            "evaluation_log_path": str(baseline_log_path),
        },
        "pedagogical_rag": {
            "request": rag_request,
            "prompt_template_version": rag_prompt["prompt_template_version"],
            "prompt_text": rag_prompt["prompt_text"],
            "response": rag_response,
            "learner_state_before": learner_state_before,
            "learner_state_after": learner_state_after,
            "state_transition": state_transition,
            "policy_decision": policy_decision,
            "retrieval_log": retrieval_log,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_log_path": str(retrieval_log_path),
            "evaluation_log_path": str(rag_log_path),
        },
        "created_at": utc_now_iso(),
    }

    matched_export_path = save_json_export(
        matched_output,
        filename=f"local_dry_run_{task_id}_matched.json",
    )

    matched_output["matched_export_path"] = str(matched_export_path)

    return matched_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one matched local dry-run for baseline and pedagogical RAG conditions."
    )
    parser.add_argument(
        "--task-id",
        default="LE-02",
        help="Benchmark task ID to run. Default: LE-02",
    )
    parser.add_argument(
        "--turn-number",
        type=int,
        default=1,
        help="Turn number for the dry-run episode. Default: 1",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of mock retrieved chunks. Default: 5",
    )

    args = parser.parse_args()

    matched_output = run_local_dry_run(
        task_id=args.task_id,
        turn_number=args.turn_number,
        top_k=args.top_k,
    )

    print(json.dumps(
        {
            "status": "local_dry_run_complete",
            "task_id": matched_output["task_id"],
            "session_id": matched_output["session_id"],
            "baseline_log_path": matched_output["baseline"]["evaluation_log_path"],
            "rag_log_path": matched_output["pedagogical_rag"]["evaluation_log_path"],
            "retrieval_log_path": matched_output["pedagogical_rag"]["retrieval_log_path"],
            "matched_export_path": matched_output["matched_export_path"],
            "selected_pedagogical_mode": matched_output["pedagogical_rag"]["policy_decision"]["selected_mode"],
            "retrieved_chunk_count": len(matched_output["pedagogical_rag"]["retrieved_chunks"]),
            "model_calls_made": matched_output["model_calls_made"],
            "external_services_called": matched_output["external_services_called"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()