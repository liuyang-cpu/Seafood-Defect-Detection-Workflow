from __future__ import annotations

import argparse
from pathlib import Path

from planner_common import (
    build_planner_payload,
    choose_baseline_run,
    default_runs_train_dir,
    default_search_space_json,
    default_task_context,
    find_repo_root,
    load_history_from_run_summaries,
    load_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local mock payload for the LLM training planner.")
    parser.add_argument("--search-space", type=str, default=None, help="Optional search-space JSON path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--baseline-version", type=str, default=None, help="Optional baseline version override.")
    parser.add_argument("--max-history", type=int, default=8, help="Maximum number of history rows to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    search_space_path = Path(args.search_space).resolve() if args.search_space else default_search_space_json(repo_root)
    search_space_payload = load_json(search_space_path)
    history_payloads = load_history_from_run_summaries(default_runs_train_dir(repo_root), max_history=args.max_history)

    payload = build_planner_payload(
        task_context=default_task_context(),
        baseline_run=choose_baseline_run(history_payloads, args.baseline_version),
        search_space_payload=search_space_payload,
        history_payloads=history_payloads,
    )

    output_path = Path(args.output).resolve() if args.output else script_path.parent.parent / "artifacts" / "llm_planner_mock_payload.json"
    write_json(output_path, payload)
    print(f"Mock payload written to: {output_path}")


if __name__ == "__main__":
    main()
