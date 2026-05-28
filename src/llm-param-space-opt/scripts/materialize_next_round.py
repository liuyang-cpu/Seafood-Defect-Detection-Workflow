from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from common import (
    default_materialized_rounds_root,
    default_optuna_config_path,
    default_search_space_path,
    find_repo_root,
    format_run_stamp,
    load_json,
    materialize_next_round,
    validate_refiner_response,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one LLM search-space refinement response and materialize next-round configs.")
    parser.add_argument("--response", type=str, required=True, help="Path to refiner_response.json")
    parser.add_argument("--meta", type=str, default=None, help="Optional refiner_meta.json path")
    parser.add_argument("--optuna-config", type=str, default=None, help="Optional current optuna config JSON path")
    parser.add_argument("--search-space", type=str, default=None, help="Optional current search-space JSON path")
    parser.add_argument("--output-root", type=str, default=None, help="Optional output root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    module_root = script_path.parent.parent
    repo_root = find_repo_root(script_path.parent)

    response_path = Path(args.response).resolve()
    if not response_path.exists():
        raise FileNotFoundError(f"Response file not found: {response_path}")

    meta_path = Path(args.meta).resolve() if args.meta else response_path.with_name("refiner_meta.json")
    optuna_config_path = Path(args.optuna_config).resolve() if args.optuna_config else default_optuna_config_path(repo_root)
    search_space_path = Path(args.search_space).resolve() if args.search_space else default_search_space_path(repo_root)
    output_root = Path(args.output_root).resolve() if args.output_root else default_materialized_rounds_root(module_root)
    schema_path = module_root / "docs" / "llm_space_refiner_response_schema.json"

    response_payload = load_json(response_path)
    schema_payload = load_json(schema_path)
    current_optuna_config = load_json(optuna_config_path)
    current_search_space_payload = load_json(search_space_path)

    errors = validate_refiner_response(
        response=response_payload,
        schema_payload=schema_payload,
        base_optuna_config=current_optuna_config,
        search_space_payload=current_search_space_payload,
    )
    if errors:
        validation_path = response_path.with_name("refiner_validation_errors.json")
        write_json(validation_path, {"errors": errors})
        raise RuntimeError(f"Refiner response validation failed. See: {validation_path}")

    next_optuna_config, next_search_space_payload, manifest_meta = materialize_next_round(
        response=response_payload,
        current_optuna_config=current_optuna_config,
        current_search_space_payload=current_search_space_payload,
    )

    round_dir = output_root / f"round_{format_run_stamp(datetime.now())}"
    round_dir.mkdir(parents=True, exist_ok=True)

    next_optuna_config_path = round_dir / "youge_optuna_config.json"
    next_search_space_path = round_dir / "train_youge_search_space.json"
    manifest_path = round_dir / "manifest.json"

    write_json(next_optuna_config_path, next_optuna_config)
    write_json(next_search_space_path, next_search_space_payload)
    write_json(
        manifest_path,
        {
            "source_response_path": str(response_path),
            "source_meta_path": str(meta_path) if meta_path.exists() else None,
            "source_optuna_config_path": str(optuna_config_path),
            "source_search_space_path": str(search_space_path),
            "generated_optuna_config_path": str(next_optuna_config_path),
            "generated_search_space_path": str(next_search_space_path),
            **manifest_meta,
        },
    )

    print(f"Materialized next round directory: {round_dir}")
    print(f"Next optuna config: {next_optuna_config_path}")
    print(f"Next search space: {next_search_space_path}")


if __name__ == "__main__":
    main()
