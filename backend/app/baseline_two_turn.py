#!/usr/bin/env python3
"""
Two-turn Gemini baseline runner for the pedagogical RAG study.

Purpose:
- Run the baseline Gemini condition in the same two-turn structure used by the benchmark.
- Turn 1: Gemini answers the original algebra task.
- Turn 2: Gemini answers the simulated student follow-up using only the immediate episode transcript.
- Preserve the baseline condition as prompt-only and non-retrieval.
- Save structured evidence for response capture and later comparison with the RAG/Bedrock condition.

Important baseline constraints:
- No Gemini File Search.
- No uploaded files.
- No uploaded images.
- No external curriculum retrieval.
- No metadata filters.
- No page-level citations.
- No knowledge base.
- No learner-state tracking.
- No persistent memory across episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai

import baseline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_EVIDENCE_DIR = (
    PROJECT_ROOT / "manuscript_artifacts" / "evidence_logs" / "baseline"
)

DEFAULT_MODEL = "gemini-2.5-flash"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_secret(value: Optional[str]) -> str:
    if not value:
        return "missing"
    return "present_redacted"


def get_task_field(task: Dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = task.get(name)
        if value is not None:
            return str(value)
    return default


def call_gemini(prompt: str, model_name: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError("Gemini response did not include response.text")

    return text


def build_turn_1_prompt(task: Dict[str, Any]) -> str:
    task_id = get_task_field(task, "task_id")
    algebra_category = get_task_field(
        task,
        "algebra_category",
        "category_label",
        "category",
    )
    student_prompt = get_task_field(
        task,
        "student_prompt",
        "student_facing_prompt",
        "prompt",
    )

    return f"""System role:
You are an algebra tutoring assistant. Your task is to help a student understand and solve algebra problems clearly and accurately.

Tutoring goal:
Provide instructionally useful support rather than only a final answer. Explain the mathematical reasoning in a way that supports student understanding.

Student-facing task:
Task ID: {task_id}
Algebra category: {algebra_category}
Student prompt: {student_prompt}

Expected tutoring behavior:
1. Solve the problem accurately.
2. Explain each important step using clear algebra language.
3. Use student-friendly wording appropriate for an algebra learner.
4. Encourage mathematical reasoning and conceptual understanding.
5. Keep the response concise, focused, and instructionally supportive.

Response constraints:
Do not use retrieval-augmented generation.
Do not use Gemini File Search.
Do not create or query a File Search store.
Do not use uploaded files or uploaded images.
Do not use metadata filters, page-level citations, or retrieved context.
Do not use external curriculum documents.
Do not use a knowledge base.
Do not use learner-state tracking.
Do not use prior-session memory.
Respond only from the current prompt context and the model's general parametric knowledge.
Do not invent unsupported facts or claims.
Do not mention that this is an evaluation unless explicitly asked.

Output:
Provide the baseline tutor response for Turn 1 only. Answer the original student task and explain the steps.
"""


def build_turn_2_prompt(task: Dict[str, Any], turn_1_response: str) -> str:
    task_id = get_task_field(task, "task_id")
    algebra_category = get_task_field(
        task,
        "algebra_category",
        "category_label",
        "category",
    )
    student_prompt = get_task_field(
        task,
        "student_prompt",
        "student_facing_prompt",
        "prompt",
    )
    simulated_follow_up_turn = get_task_field(
        task,
        "simulated_follow_up_turn",
        "follow_up",
        "simulated_followup",
    )

    return f"""System role:
You are an algebra tutoring assistant. Your task is to help a student understand and solve algebra problems clearly and accurately.

Tutoring goal:
Respond to the student's follow-up in a way that supports understanding. Use only the immediate episode transcript below.

Original student task:
Task ID: {task_id}
Algebra category: {algebra_category}
Student prompt: {student_prompt}

Turn 1 baseline tutor response:
{turn_1_response}

Turn 2 simulated student follow-up:
{simulated_follow_up_turn}

Expected tutoring behavior:
1. Respond directly to the student's follow-up.
2. If the student gives an incorrect answer or expresses a misconception, identify the error and explain why it is incorrect.
3. Ask a guiding question when appropriate rather than immediately over-explaining.
4. Keep the response concise, focused, and instructionally supportive.
5. Encourage mathematical reasoning and conceptual understanding.

Response constraints:
Do not use retrieval-augmented generation.
Do not use Gemini File Search.
Do not create or query a File Search store.
Do not use uploaded files or uploaded images.
Do not use metadata filters, page-level citations, or retrieved context.
Do not use external curriculum documents.
Do not use a knowledge base.
Do not use learner-state tracking.
Do not use persistent memory or prior-session memory.
Use only the immediate Turn 1 and Turn 2 transcript supplied in this prompt.
Do not invent unsupported facts or claims.
Do not mention that this is an evaluation unless explicitly asked.

Output:
Provide the baseline tutor response for Turn 2 only. Respond to the simulated student follow-up.
"""


def create_two_turn_log(
    task: Dict[str, Any],
    model_name: str,
    turn_1_prompt: str,
    turn_1_response: str,
    turn_2_prompt: str,
    turn_2_response: str,
) -> Dict[str, Any]:
    task_id = get_task_field(task, "task_id")

    return {
        "evaluation_log_id": f"baseline_two_turn_{uuid.uuid4().hex}",
        "condition": "baseline_prompt_only",
        "episode_type": "two_turn_simulated_follow_up",
        "task_id": task_id,
        "session_id": f"baseline_two_turn_{task_id}",
        "created_at": utc_now_iso(),
        "model": {
            "provider": "Google",
            "model_id": model_name,
            "api_surface": "Gemini Developer API",
        },
        "baseline_constraints": {
            "retrieval_used": False,
            "gemini_file_search_used": False,
            "file_search_store_created_or_queried": False,
            "uploaded_files_used": False,
            "uploaded_images_used": False,
            "external_curriculum_documents_used": False,
            "knowledge_base_used": False,
            "metadata_filters_used": False,
            "page_level_citations_requested": False,
            "retrieved_context_passed_to_model": False,
            "learner_state_tracking_used": False,
            "prior_session_memory_used": False,
            "persistent_memory_used": False,
        },
        "task": {
            "task_id": task_id,
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
        },
        "turns": [
            {
                "turn_number": 1,
                "turn_type": "initial_task_response",
                "prompt_text": turn_1_prompt,
                "response_id": f"gemini_turn_1_{uuid.uuid4().hex}",
                "response_text": turn_1_response,
                "is_mock_response": False,
            },
            {
                "turn_number": 2,
                "turn_type": "simulated_follow_up_response",
                "prompt_text": turn_2_prompt,
                "response_id": f"gemini_turn_2_{uuid.uuid4().hex}",
                "response_text": turn_2_response,
                "is_mock_response": False,
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
                "Two-turn logged baseline episode showing the prompt-only Gemini "
                "baseline response to the original task and simulated learner follow-up."
            ),
            "notes": (
                "This two-turn log supports baseline response capture. It does not use "
                "retrieval, learner-state tracking, external curriculum grounding, Gemini "
                "File Search, uploaded files, uploaded images, metadata filters, or page-level citations."
            ),
        },
    }


def save_log(log: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-turn real Gemini prompt-only baseline episode."
    )

    parser.add_argument(
        "--task-id",
        required=True,
        help="Benchmark task ID to run, for example SE-01 or LE-02.",
    )

    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Required safety flag confirming that real Gemini model calls should be made.",
    )

    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()

    if not args.real_model:
        print(
            json.dumps(
                {
                    "status": "refused_without_real_model_flag",
                    "message": "Add --real-model to make real Gemini baseline calls.",
                },
                indent=2,
            )
        )
        return 2

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if not api_key or api_key == "PASTE_YOUR_REAL_GEMINI_API_KEY_HERE":
        print(
            json.dumps(
                {
                    "status": "missing_gemini_api_key",
                    "message": "Set GEMINI_API_KEY in .env before running the baseline.",
                },
                indent=2,
            )
        )
        return 1

    task = baseline.find_task(args.task_id)

    turn_1_prompt = build_turn_1_prompt(task)
    turn_1_response = call_gemini(
        prompt=turn_1_prompt,
        model_name=model_name,
        api_key=api_key,
    )

    turn_2_prompt = build_turn_2_prompt(
        task=task,
        turn_1_response=turn_1_response,
    )
    turn_2_response = call_gemini(
        prompt=turn_2_prompt,
        model_name=model_name,
        api_key=api_key,
    )

    log = create_two_turn_log(
        task=task,
        model_name=model_name,
        turn_1_prompt=turn_1_prompt,
        turn_1_response=turn_1_response,
        turn_2_prompt=turn_2_prompt,
        turn_2_response=turn_2_response,
    )

    output_path = (
        BASELINE_EVIDENCE_DIR
        / f"baseline_episode_{args.task_id.replace('-', '_')}_two_turn.json"
    )

    save_log(log, output_path)

    print(
        json.dumps(
            {
                "status": "baseline_two_turn_episode_complete",
                "task_id": args.task_id,
                "condition": "baseline_prompt_only",
                "episode_type": "two_turn_simulated_follow_up",
                "model": model_name,
                "retrieval_used": False,
                "file_search_used": False,
                "learner_state_used": False,
                "model_calls_made": 2,
                "external_services_called": True,
                "log_path": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())