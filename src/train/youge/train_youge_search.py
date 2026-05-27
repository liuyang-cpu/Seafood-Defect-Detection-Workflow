from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "ultralytics").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate repo root from: {start}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and optionally run youge training search experiments.")
    parser.add_argument("--search-space", type=str, default=None, help="Path to the search-space JSON file.")
    parser.add_argument("--base-config", type=str, default=None, help="Path to the base training config JSON file.")
    parser.add_argument(
        "--plan",
        type=str,
        choices=("stage1-oat", "stage1-grid"),
        default="stage1-oat",
        help="Experiment generation strategy. stage1-oat = one-at-a-time around baseline. stage1-grid = Cartesian product.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on generated experiments.")
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When true, immediately run all generated experiments.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_stage1_oat_experiments(search_space: dict, base_config: dict) -> list[dict]:
    baseline = dict(search_space.get("baseline_observed_args") or {})
    grid = dict(search_space.get("stage1_recommended_grid") or {})
    grid.pop("_comment", None)
    priority_order = list(search_space.get("priority_order") or [])
    ordered_keys = [key for key in priority_order if key in grid]
    ordered_keys.extend(key for key in grid if key not in ordered_keys)

    experiments: list[dict] = []
    experiments.append(
        {
            "label": "baseline_refresh",
            "description": "Baseline-like rerun using current base config.",
            "overrides": {},
        }
    )

    for key in ordered_keys:
        values = grid[key]
        baseline_value = base_config.get(key, baseline.get(key))
        for value in values:
            if value == baseline_value:
                continue
            experiments.append(
                {
                    "label": f"{key}_{str(value).replace('.', '_')}",
                    "description": f"One-at-a-time override for {key}.",
                    "overrides": {key: value},
                }
            )
    return experiments


def build_stage1_grid_experiments(search_space: dict) -> list[dict]:
    grid = dict(search_space.get("stage1_recommended_grid") or {})
    grid.pop("_comment", None)
    keys = list(grid.keys())
    value_lists = [list(grid[key]) for key in keys]
    experiments: list[dict] = []
    for combo in itertools.product(*value_lists):
        overrides = {key: value for key, value in zip(keys, combo)}
        label_parts = [f"{key}_{str(value).replace('.', '_')}" for key, value in overrides.items()]
        experiments.append(
            {
                "label": "__".join(label_parts),
                "description": "Cartesian product experiment from stage1 grid.",
                "overrides": overrides,
            }
        )
    return experiments


def split_overrides(overrides: dict) -> tuple[dict, dict]:
    direct_keys = {"epochs", "imgsz", "batch", "device", "workers", "project", "name", "version", "patience", "mosaic", "cache", "exist_ok", "amp"}
    direct_config: dict = {}
    extra_train_args: dict = {}
    for key, value in overrides.items():
        if key in direct_keys:
            direct_config[key] = value
        else:
            extra_train_args[key] = value
    return direct_config, extra_train_args


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    search_space_path = Path(args.search_space).resolve() if args.search_space else script_path.with_name("train_youge_search_space.json")
    base_config_path = Path(args.base_config).resolve() if args.base_config else script_path.with_name("train_youge.json")
    train_script_path = script_path.with_name("train_youge.py")

    if not search_space_path.exists():
        raise FileNotFoundError(f"Search-space JSON not found: {search_space_path}")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config JSON not found: {base_config_path}")
    if not train_script_path.exists():
        raise FileNotFoundError(f"Train script not found: {train_script_path}")

    search_space = load_json(search_space_path)
    base_config = load_json(base_config_path)

    if args.plan == "stage1-oat":
        experiments = build_stage1_oat_experiments(search_space, base_config)
    else:
        experiments = build_stage1_grid_experiments(search_space)

    if args.max_runs is not None:
        experiments = experiments[: max(0, args.max_runs)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_root = script_path.parent / "runs" / "search_plans" / f"youge_search_{timestamp}_{args.plan}"
    configs_dir = plan_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    manifest_runs: list[dict] = []
    for index, experiment in enumerate(experiments, start=1):
        direct_overrides, extra_train_args = split_overrides(dict(experiment["overrides"]))
        config_payload = dict(base_config)
        config_payload.update(direct_overrides)
        config_payload["version"] = None
        config_payload["exist_ok"] = True
        config_payload["report_group"] = plan_root.name
        config_payload["report_label"] = experiment["label"]
        config_payload["report_description"] = experiment["description"]
        if extra_train_args:
            merged_extra = dict(config_payload.get("extra_train_args") or {})
            merged_extra.update(extra_train_args)
            config_payload["extra_train_args"] = merged_extra

        config_name = f"exp_{index:03d}.json"
        config_path = configs_dir / config_name
        config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest_runs.append(
            {
                "index": index,
                "label": experiment["label"],
                "description": experiment["description"],
                "config_path": str(config_path),
                "overrides": experiment["overrides"],
            }
        )

    manifest = {
        "created_at": timestamp,
        "plan": args.plan,
        "search_space_path": str(search_space_path),
        "base_config_path": str(base_config_path),
        "execute": bool(args.execute),
        "experiments": manifest_runs,
    }
    manifest_path = plan_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated {len(manifest_runs)} experiment configs.")
    print(f"Plan directory: {plan_root}")
    print(f"Manifest: {manifest_path}")

    if not args.execute:
        return

    for run in manifest_runs:
        print(f"[{run['index']}/{len(manifest_runs)}] Running {run['label']}")
        command = [sys.executable, str(train_script_path), "--config", str(run["config_path"])]
        subprocess.run(command, cwd=str(repo_root), check=True)

    report_script_path = script_path.with_name("report_youge_experiments.py")
    if report_script_path.exists():
        subprocess.run(
            [sys.executable, str(report_script_path), "--group", plan_root.name],
            cwd=str(repo_root),
            check=True,
        )


if __name__ == "__main__":
    main()
