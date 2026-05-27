from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "ultralytics").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate repo root from: {start}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one existing youge search plan from runs/search_plans.")
    parser.add_argument("--plan", type=str, default=None, help="Plan directory name or full path.")
    parser.add_argument("--list", action="store_true", help="List available search plans and exit.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based experiment start index.")
    parser.add_argument("--end-index", type=int, default=None, help="1-based experiment end index.")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum number of experiments to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected runs without executing.")
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip aggregate report generation after training.",
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch for every selected run.")
    parser.add_argument("--workers", type=int, default=None, help="Override workers for every selected run.")
    return parser.parse_args()


def get_search_plans_root(script_path: Path) -> Path:
    return script_path.parent / "runs" / "search_plans"


def get_train_circle_materialized_root(script_path: Path) -> Path:
    repo_root = find_repo_root(script_path.parent)
    preferred = repo_root / "src" / "train-circle" / "plans" / "materialized_plans"
    legacy = repo_root / "src" / "train-circle" / "plans"
    return preferred if preferred.exists() else legacy


def list_plan_sources(script_path: Path) -> list[tuple[str, Path]]:
    search_plans_root = get_search_plans_root(script_path)
    materialized_root = get_train_circle_materialized_root(script_path)
    sources: list[tuple[str, Path]] = []

    if search_plans_root.exists():
        for path in sorted((item for item in search_plans_root.iterdir() if item.is_dir()), key=lambda item: item.name):
            sources.append((f"搜索计划 | {path.name}", path.resolve()))

    if materialized_root.exists():
        candidates = [
            path for path in materialized_root.iterdir() if path.is_dir() and path.name.startswith("materialized_")
        ]
        for path in sorted(candidates, key=lambda item: item.name):
            sources.append((f"LLM计划 | {path.name}", path.resolve()))

    return sources


def list_plan_dirs(search_plans_root: Path) -> list[Path]:
    if not search_plans_root.exists():
        return []
    return sorted((path for path in search_plans_root.iterdir() if path.is_dir()), key=lambda path: path.name)


def resolve_plan_dir(plan_arg: str | None, search_plans_root: Path) -> Path:
    if not plan_arg:
        raise ValueError("Missing --plan. Use --list to inspect available plans.")

    candidate = Path(plan_arg)
    if candidate.is_dir():
        return candidate.resolve()

    candidate = search_plans_root / plan_arg
    if candidate.is_dir():
        return candidate.resolve()

    raise FileNotFoundError(f"Search plan directory not found: {plan_arg}")


def load_manifest(plan_dir: Path) -> dict:
    manifest_path = plan_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def select_runs(experiments: list[dict], start_index: int, end_index: int | None, max_runs: int | None) -> list[dict]:
    if start_index < 1:
        raise ValueError("--start-index must be >= 1")

    selected = [exp for exp in experiments if int(exp["index"]) >= start_index]
    if end_index is not None:
        if end_index < start_index:
            raise ValueError("--end-index must be >= --start-index")
        selected = [exp for exp in selected if int(exp["index"]) <= end_index]
    if max_runs is not None:
        selected = selected[: max(0, max_runs)]
    return selected


def resolve_experiment_config_path(plan_dir: Path, run: dict) -> Path:
    raw_path = Path(str(run["config_path"]))
    if raw_path.exists():
        return raw_path.resolve()

    fallback = (plan_dir / "configs" / raw_path.name).resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Experiment config not found. Manifest path={raw_path}, fallback path={fallback}")


def get_plan_selection(
    *,
    script_path: Path,
    plan: str,
    start_index: int = 1,
    end_index: int | None = None,
    max_runs: int | None = None,
) -> tuple[Path, dict, list[dict]]:
    search_plans_root = get_search_plans_root(script_path)
    plan_dir = resolve_plan_dir(plan, search_plans_root)
    manifest = load_manifest(plan_dir)
    experiments = list(manifest.get("experiments") or [])
    if not experiments:
        raise RuntimeError(f"No experiments found in manifest: {plan_dir / 'manifest.json'}")

    selected_runs = select_runs(experiments, start_index, end_index, max_runs)
    if not selected_runs:
        raise RuntimeError("No experiments matched the requested range.")
    return plan_dir, manifest, selected_runs


def run_plan(
    *,
    script_path: Path,
    plan: str,
    start_index: int = 1,
    end_index: int | None = None,
    max_runs: int | None = None,
    dry_run: bool = False,
    skip_report: bool = False,
    batch: int | None = None,
    workers: int | None = None,
    python_executable: str | None = None,
    log: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[Path, list[dict]]:
    log_fn = log or print
    repo_root = find_repo_root(script_path.parent)
    plan_dir, _manifest, selected_runs = get_plan_selection(
        script_path=script_path,
        plan=plan,
        start_index=start_index,
        end_index=end_index,
        max_runs=max_runs,
    )

    train_script_path = script_path.with_name("train_youge.py")
    report_script_path = script_path.with_name("report_youge_experiments.py")
    python_cmd = python_executable or sys.executable
    stop_check = stop_requested or (lambda: False)

    log_fn(f"Plan: {plan_dir.name}")
    log_fn(f"Selected runs: {len(selected_runs)}")
    if batch is not None or workers is not None:
        log_fn(
            "Global overrides: "
            + ", ".join(
                part
                for part in (
                    f"batch={batch}" if batch is not None else "",
                    f"workers={workers}" if workers is not None else "",
                )
                if part
            )
        )
    for run in selected_runs:
        config_path = resolve_experiment_config_path(plan_dir, run)
        log_fn(f"  [{run['index']}] {run['label']} -> {config_path}")

    if dry_run:
        return plan_dir, selected_runs

    for offset, run in enumerate(selected_runs, start=1):
        if stop_check():
            raise RuntimeError("执行已被用户终止。")
        log_fn(f"[{offset}/{len(selected_runs)}] Running plan item {run['index']}: {run['label']}")
        config_path = resolve_experiment_config_path(plan_dir, run)
        command = [
            python_cmd,
            str(train_script_path),
            "--config",
            str(config_path),
            "--report-group",
            plan_dir.name,
            "--report-label",
            str(run["label"]),
            "--report-description",
            str(run.get("description") or ""),
        ]
        if batch is not None:
            command.extend(["--batch", str(batch)])
        if workers is not None:
            command.extend(["--workers", str(workers)])
        process = subprocess.Popen(command, cwd=str(repo_root))
        try:
            while True:
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        raise subprocess.CalledProcessError(return_code, command)
                    break
                if stop_check():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError("执行已被用户终止。")
                time.sleep(1.0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    if not skip_report and report_script_path.exists():
        if stop_check():
            raise RuntimeError("执行已被用户终止。")
        log_fn("Generating aggregate report...")
        command = [
            python_cmd,
            str(report_script_path),
            "--group",
            plan_dir.name,
        ]
        process = subprocess.Popen(command, cwd=str(repo_root))
        try:
            while True:
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        raise subprocess.CalledProcessError(return_code, command)
                    break
                if stop_check():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError("执行已被用户终止。")
                time.sleep(1.0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    return plan_dir, selected_runs


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    search_plans_root = get_search_plans_root(script_path)

    if args.list:
        sources = list_plan_sources(script_path)
        if not sources:
            print(f"No search plans found under: {search_plans_root}")
            return
        print("Available plans:")
        for display_name, plan_dir in sources:
            print(f"- {display_name} -> {plan_dir}")
        return

    plan_dir, _manifest, _selected_runs = get_plan_selection(
        script_path=script_path,
        plan=args.plan,
        start_index=args.start_index,
        end_index=args.end_index,
        max_runs=args.max_runs,
    )
    run_plan(
        script_path=script_path,
        plan=str(plan_dir),
        start_index=args.start_index,
        end_index=args.end_index,
        max_runs=args.max_runs,
        dry_run=args.dry_run,
        skip_report=args.skip_report,
        batch=args.batch,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
