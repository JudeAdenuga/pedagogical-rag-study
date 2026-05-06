"""
Task loader for the algebra tutoring benchmark task bank.

This module loads benchmark_tasks.json and provides helper functions for:
- listing tasks
- retrieving one task by task_id
- validating task structure
- preparing task metadata for prompt, state, retrieval, and policy modules
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_BANK_PATH = PROJECT_ROOT / "backend" / "data" / "benchmark_tasks.json"


class TaskBankError(Exception):
    """Raised when the benchmark task bank cannot be loaded or queried."""


def load_task_bank(path: Optional[Path] = None) -> Dict[str, Any]:
    task_path = path or TASK_BANK_PATH

    if not task_path.exists():
        raise TaskBankError(f"Task bank file not found: {task_path}")

    try:
        with task_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise TaskBankError(f"Invalid JSON in task bank: {task_path}") from exc

    if "tasks" not in data or not isinstance(data["tasks"], list):
        raise TaskBankError("Task bank must contain a top-level 'tasks' array.")

    return data


def list_tasks() -> List[Dict[str, Any]]:
    task_bank = load_task_bank()
    return task_bank["tasks"]


def get_task_by_id(task_id: str) -> Dict[str, Any]:
    normalized_id = task_id.strip()

    for task in list_tasks():
        if task.get("task_id") == normalized_id:
            return task

    raise TaskBankError(f"Task not found: {task_id}")


def get_task_metadata(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "task_order": task.get("task_order"),
        "category": task.get("category"),
        "category_label": task.get("category_label"),
        "topic": task.get("topic"),
        "subskill": task.get("subskill"),
        "difficulty": task.get("difficulty"),
        "resource_type": task.get("resource_type"),
        "misconception_tag": task.get("misconception_tag"),
        "student_facing_prompt": task.get("student_facing_prompt"),
        "expected_answer_or_reasoning_target": task.get("expected_answer_or_reasoning_target"),
        "simulated_follow_up_turn": task.get("simulated_follow_up_turn"),
        "scoring_notes": task.get("scoring_notes"),
    }


def validate_task(task: Dict[str, Any]) -> None:
    required_fields = [
        "task_id",
        "category",
        "category_label",
        "student_facing_prompt",
        "expected_answer_or_reasoning_target",
        "simulated_follow_up_turn",
        "scoring_notes",
        "topic",
        "subskill",
        "difficulty",
        "resource_type",
        "misconception_tag",
    ]

    missing = [field for field in required_fields if field not in task]

    if missing:
        task_id = task.get("task_id", "UNKNOWN_TASK")
        raise TaskBankError(f"Task {task_id} is missing required fields: {missing}")


def validate_all_tasks() -> Dict[str, Any]:
    task_bank = load_task_bank()
    tasks = task_bank["tasks"]

    errors = []

    for task in tasks:
        try:
            validate_task(task)
        except TaskBankError as exc:
            errors.append(str(exc))

    return {
        "task_bank_name": task_bank.get("task_bank_name"),
        "task_bank_version": task_bank.get("task_bank_version"),
        "declared_total_tasks": task_bank.get("total_tasks"),
        "actual_total_tasks": len(tasks),
        "valid": len(errors) == 0,
        "errors": errors,
    }


if __name__ == "__main__":
    result = validate_all_tasks()
    print(json.dumps(result, indent=2))

    sample = get_task_by_id("LE-04")
    print("\nSample task LE-04:")
    print(json.dumps(get_task_metadata(sample), indent=2))