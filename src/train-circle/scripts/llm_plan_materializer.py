from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from planner_common import (
    default_base_config_json,
    default_search_space_json,
    find_repo_root,
    load_history_from_run_summaries,
    load_json,
    materialize_plan,
    validate_planner_response,
    write_json,
)


def default_materialized_plans_root(script_path: Path) -> Path:
    return script_path.parent.parent / "plans" / "materialized_plans"


def format_run_stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H_%M_%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an LLM planner response and convert it into a runnable search plan."
    )
    parser.add_argument("--planner-response", type=str, required=True, help="Path to planner_response.json")
    parser.add_argument("--payload", type=str, default=None, help="Optional planner_payload.json path")
    parser.add_argument("--search-space", type=str, default=None, help="Optional search-space JSON path")
    parser.add_argument("--base-config", type=str, default=None, help="Optional base config JSON path")
    parser.add_argument("--output-root", type=str, default=None, help="Optional output root for generated plans")
    return parser.parse_args()


def materialize_from_paths(
    *,
    planner_response_path: Path,
    payload_path: Path | None = None,
    search_space_path: Path | None = None,
    base_config_path: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)

    planner_response_path = planner_response_path.resolve()
    payload_path = payload_path.resolve() if payload_path else planner_response_path.with_name("planner_payload.json")
    search_space_path = search_space_path.resolve() if search_space_path else default_search_space_json(repo_root)
    base_config_path = base_config_path.resolve() if base_config_path else default_base_config_json(repo_root)
    output_root = output_root.resolve() if output_root else default_materialized_plans_root(script_path)

    planner_response = load_json(planner_response_path)
    _payload = load_json(payload_path) if payload_path.exists() else {}
    search_space_payload = load_json(search_space_path)
    base_config_payload = load_json(base_config_path)
    schema_payload = load_json(script_path.parent.parent / "docs" / "llm_planner_response_schema.json")
    history_payloads = load_history_from_run_summaries(repo_root / "src" / "train" / "youge" / "runs" / "train")

    errors = validate_planner_response(
        planner_response=planner_response,
        schema_payload=schema_payload,
        search_space_payload=search_space_payload,
        history_payloads=history_payloads,
    )
    if errors:
        validation_path = planner_response_path.with_name("planner_validation_errors.json")
        write_json(validation_path, {"errors": errors})
        raise RuntimeError(f"Planner response validation failed. See: {validation_path}")

    plan_dir = output_root / f"materialized_{format_run_stamp(datetime.now())}"
    manifest = materialize_plan(
        planner_response=planner_response,
        plan_root=plan_dir,
        base_config=base_config_payload,
        metadata={
            "base_config_path": str(base_config_path),
            "search_space_path": str(search_space_path),
            "planner_response_path": str(planner_response_path),
            "payload_path": str(payload_path) if payload_path.exists() else None,
        },
    )
    write_json(
        plan_dir / "materializer_meta.json", {"validation": "passed", "manifest_path": str(plan_dir / "manifest.json")}
    )
    return {
        "plan_dir": str(plan_dir),
        "manifest_path": str(plan_dir / "manifest.json"),
        "experiments_count": len(manifest.get("experiments") or []),
        "planner_response_path": str(planner_response_path),
        "payload_path": str(payload_path) if payload_path.exists() else None,
    }


def main() -> None:
    args = parse_args()
    result = materialize_from_paths(
        planner_response_path=Path(args.planner_response),
        payload_path=Path(args.payload) if args.payload else None,
        search_space_path=Path(args.search_space) if args.search_space else None,
        base_config_path=Path(args.base_config) if args.base_config else None,
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(f"Materialized plan directory: {result['plan_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Experiments: {result['experiments_count']}")


if __name__ == "__main__":
    main()
