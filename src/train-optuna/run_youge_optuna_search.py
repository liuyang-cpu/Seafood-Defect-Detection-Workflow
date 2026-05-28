from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    apply_overrides,
    default_base_config_json,
    default_runs_train_dir,
    default_search_space_json,
    default_studies_root,
    default_study_name,
    default_train_script,
    find_repo_root,
    load_json,
    resolve_metric_value,
    sanitize_name,
    write_json,
)
from report_youge_optuna_study import generate_study_report

try:
    import optuna
except ImportError:
    optuna = None


RESULTS_DIR_PATTERN = re.compile(r"Results saved to:\s*(.+)")
VERSION_PATTERN = re.compile(r"Version=(version\d+)")
DEFAULT_CONFIG = {
    "study_name": None,
    "metric": "map50_95",
    "direction": "maximize",
    "n_trials": 20,
    "timeout_seconds": None,
    "search_space_key": "full_search_space",
    "enabled_params": None,
    "sampler": {
        "name": "TPESampler",
        "seed": 42,
        "n_startup_trials": 5,
        "multivariate": False,
    },
    "pruner": {
        "name": "NopPruner",
    },
    "fixed_overrides": {
        "device": "0",
        "workers": 4,
        "exist_ok": True,
    },
    "use_fixed_recommendations": True,
    "continue_on_trial_failure": True,
    "subprocess_timeout_seconds": None,
}


def require_optuna() -> Any:
    if optuna is None:
        raise RuntimeError(
            f"Optuna is not installed for Python interpreter: {sys.executable}\n"
            "Activate the environment used for YOLO training, then install and rerun with:\n"
            "  python -m pip install optuna"
        )
    return optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run youge hyperparameter search with Optuna.")
    parser.add_argument("--config", type=str, default=None, help="Optional Optuna config JSON path.")
    parser.add_argument("--study-name", type=str, default=None, help="Override study name.")
    parser.add_argument("--base-config", type=str, default=None, help="Override base train_youge.json path.")
    parser.add_argument("--search-space", type=str, default=None, help="Override search space JSON path.")
    parser.add_argument("--train-script", type=str, default=None, help="Override train_youge.py path.")
    parser.add_argument("--storage", type=str, default=None, help="Override Optuna storage URL.")
    parser.add_argument("--n-trials", type=int, default=None, help="Override total number of trials.")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Override study timeout in seconds.")
    parser.add_argument("--metric", type=str, default=None, help="Override objective metric.")
    parser.add_argument("--sampler", type=str, default=None, choices=("tpe", "random"), help="Override sampler type.")
    parser.add_argument("--enqueue-baseline", action="store_true", help="Enqueue one baseline-like trial before search.")
    parser.add_argument("--top-k-report", type=int, default=20, help="Number of completed trials to include in report charts.")
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return dict(DEFAULT_CONFIG)
    payload = dict(DEFAULT_CONFIG)
    payload.update(load_json(path))
    return payload


def resolve_study_name(args: argparse.Namespace, config: dict[str, Any]) -> str:
    raw = args.study_name or config.get("study_name") or default_study_name()
    return sanitize_name(raw, fallback="youge_optuna")


def resolve_paths(
    repo_root: Path,
    module_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    study_name: str,
) -> dict[str, Path | str]:
    study_dir = default_studies_root(module_dir) / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    base_config_path = Path(args.base_config).resolve() if args.base_config else default_base_config_json(repo_root)
    search_space_path = Path(args.search_space).resolve() if args.search_space else default_search_space_json(repo_root)
    train_script_path = Path(args.train_script).resolve() if args.train_script else default_train_script(repo_root)
    storage = args.storage or config.get("storage") or f"sqlite:///{(study_dir / 'study.db').resolve().as_posix()}"
    return {
        "study_dir": study_dir.resolve(),
        "base_config_path": base_config_path.resolve(),
        "search_space_path": search_space_path.resolve(),
        "train_script_path": train_script_path.resolve(),
        "storage": str(storage),
    }


def resolve_sampler(optuna_mod: Any, sampler_config: dict[str, Any], sampler_override: str | None) -> Any:
    name = (sampler_override or sampler_config.get("name") or "TPESampler").lower()
    seed = sampler_config.get("seed")
    if name == "random" or name == "randomsampler":
        return optuna_mod.samplers.RandomSampler(seed=seed)
    return optuna_mod.samplers.TPESampler(
        seed=seed,
        n_startup_trials=int(sampler_config.get("n_startup_trials", 5)),
        multivariate=bool(sampler_config.get("multivariate", False)),
    )


def resolve_pruner(optuna_mod: Any, pruner_config: dict[str, Any]) -> Any:
    name = str(pruner_config.get("name") or "NopPruner").lower()
    if name == "medianpruner":
        return optuna_mod.pruners.MedianPruner(
            n_startup_trials=int(pruner_config.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_config.get("n_warmup_steps", 0)),
        )
    return optuna_mod.pruners.NopPruner()


def resolve_search_space(config: dict[str, Any], search_space_payload: dict[str, Any]) -> dict[str, Any]:
    key = str(config.get("search_space_key") or "full_search_space")
    raw_space = dict(search_space_payload.get(key) or {})
    raw_space.pop("_comment", None)
    enabled_params = config.get("enabled_params")
    if enabled_params:
        return {name: raw_space[name] for name in enabled_params if name in raw_space}
    return raw_space


def suggest_value(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    spec_type = spec.get("type")
    if spec_type == "categorical":
        return trial.suggest_categorical(name, list(spec.get("values") or []))
    if spec_type == "int":
        minimum = int(spec["min"])
        maximum = int(spec["max"])
        step = int(spec.get("step", 1))
        if spec.get("scale") == "log":
            return trial.suggest_int(name, minimum, maximum, log=True)
        return trial.suggest_int(name, minimum, maximum, step=step)
    if spec_type == "float":
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        if spec.get("scale") == "log":
            return trial.suggest_float(name, minimum, maximum, log=True)
        if "step" in spec:
            return trial.suggest_float(name, minimum, maximum, step=float(spec["step"]))
        return trial.suggest_float(name, minimum, maximum)
    raise ValueError(f"Unsupported search space spec for {name}: {spec}")


def build_trial_overrides(
    trial: Any,
    *,
    base_config: dict[str, Any],
    search_space: dict[str, Any],
    fixed_overrides: dict[str, Any],
    fixed_recommendations: dict[str, Any],
    study_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sampled_overrides = {name: suggest_value(trial, name, spec) for name, spec in search_space.items()}
    all_overrides = {}
    all_overrides.update(fixed_recommendations)
    all_overrides.update(fixed_overrides)
    all_overrides.update(sampled_overrides)

    base_name = str(fixed_overrides.get("name") or base_config.get("name") or "youge_yolo11n_optuna")
    all_overrides["name"] = f"{sanitize_name(base_name, fallback='youge_optuna')}_trial{trial.number:04d}"
    all_overrides["version"] = None
    all_overrides["exist_ok"] = True
    all_overrides["report_group"] = study_name
    all_overrides["report_label"] = f"optuna_trial_{trial.number:04d}"
    all_overrides["report_description"] = json.dumps(sampled_overrides, ensure_ascii=False, sort_keys=True)
    return all_overrides, sampled_overrides


def build_trial_config(base_config: dict[str, Any], trial_overrides: dict[str, Any]) -> dict[str, Any]:
    return apply_overrides(base_config, trial_overrides)


def print_trial_banner(
    *,
    trial_number: int,
    total_trials: int,
    metric_name: str,
    study_name: str,
    trial_dir: Path,
    sampled_overrides: dict[str, Any],
) -> None:
    print(f"[Optuna] Trial {trial_number + 1}/{total_trials} started for study={study_name} (trial_{trial_number:04d})")
    print(f"[Optuna] Objective metric: {metric_name}")
    print(f"[Optuna] Trial directory: {trial_dir}")
    print(f"[Optuna] Sampled params: {json.dumps(sampled_overrides, ensure_ascii=False, sort_keys=True)}")


def stream_subprocess_output(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int | None,
    display_prefix: str,
) -> str:
    output_parts: list[str] = []
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("Unable to capture subprocess stdout.")
            while True:
                if deadline is not None and time.monotonic() > deadline:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(command, timeout_seconds, output="".join(output_parts))

                line = process.stdout.readline()
                if line:
                    print(f"{display_prefix}{line}", end="")
                    log_file.write(line)
                    log_file.flush()
                    output_parts.append(line)
                    continue

                if process.poll() is not None:
                    break

            remainder = process.stdout.read()
            if remainder:
                for line in remainder.splitlines(keepends=True):
                    print(f"{display_prefix}{line}", end="")
                log_file.write(remainder)
                log_file.flush()
                output_parts.append(remainder)

            returncode = process.wait()
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise

    output_text = "".join(output_parts)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output=output_text)
    return output_text


def parse_train_output(output_text: str) -> tuple[Path | None, str | None]:
    run_dir = None
    version = None
    for line in output_text.splitlines():
        match = RESULTS_DIR_PATTERN.search(line)
        if match:
            run_dir = Path(match.group(1).strip()).resolve()
        version_match = VERSION_PATTERN.search(line)
        if version_match:
            version = version_match.group(1)
    return run_dir, version


def find_matching_run_summary(
    runs_train_dir: Path,
    *,
    report_group: str,
    report_label: str,
    started_at: datetime,
) -> Path | None:
    if not runs_train_dir.exists():
        return None
    candidates = sorted(
        (path for path in runs_train_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        if datetime.fromtimestamp(run_dir.stat().st_mtime) < started_at:
            continue
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if summary.get("report_group") == report_group and summary.get("report_label") == report_label:
            return summary_path
    return None


def load_run_summary_from_result(
    *,
    repo_root: Path,
    run_dir: Path | None,
    report_group: str,
    report_label: str,
    started_at: datetime,
) -> dict[str, Any]:
    if run_dir is not None:
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            return load_json(summary_path)

    summary_path = find_matching_run_summary(
        default_runs_train_dir(repo_root),
        report_group=report_group,
        report_label=report_label,
        started_at=started_at,
    )
    if summary_path is None:
        raise FileNotFoundError("Unable to locate run_summary.json for the finished Optuna trial.")
    return load_json(summary_path)


def export_best_trial(study: Any, study_dir: Path, base_config: dict[str, Any]) -> None:
    completed_trials = [trial for trial in study.trials if getattr(trial.state, "name", "") == "COMPLETE" and trial.value is not None]
    if not completed_trials:
        return
    best_trial = study.best_trial
    best_payload = {
        "study_name": study.study_name,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "trial_number": best_trial.number,
        "value": best_trial.value,
        "params": dict(best_trial.params),
        "user_attrs": dict(best_trial.user_attrs),
    }
    write_json(study_dir / "best_trial.json", best_payload)
    best_config = apply_overrides(base_config, dict(best_trial.user_attrs.get("resolved_overrides") or {}))
    write_json(study_dir / "best_trial_config.json", best_config)


def build_enqueued_baseline_params(
    *,
    search_space: dict[str, Any],
    base_config: dict[str, Any],
    fixed_recommendations: dict[str, Any],
) -> dict[str, Any]:
    extra_train_args = dict(base_config.get("extra_train_args") or {})
    params: dict[str, Any] = {}
    for name in search_space:
        if name in extra_train_args:
            params[name] = extra_train_args[name]
        elif name in base_config:
            params[name] = base_config[name]
        elif name in fixed_recommendations:
            params[name] = fixed_recommendations[name]
    return params


def main() -> None:
    args = parse_args()
    optuna_mod = require_optuna()
    module_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(module_dir)

    config_path = Path(args.config).resolve() if args.config else module_dir / "youge_optuna_config.json"
    config = load_config(config_path if config_path.exists() else None)
    study_name = resolve_study_name(args, config)
    paths = resolve_paths(repo_root, module_dir, args, config, study_name)

    base_config_path = Path(paths["base_config_path"])
    search_space_path = Path(paths["search_space_path"])
    train_script_path = Path(paths["train_script_path"])
    study_dir = Path(paths["study_dir"])
    storage = str(paths["storage"])

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if not search_space_path.exists():
        raise FileNotFoundError(f"Search space not found: {search_space_path}")
    if not train_script_path.exists():
        raise FileNotFoundError(f"Train script not found: {train_script_path}")

    base_config = load_json(base_config_path)
    search_space_payload = load_json(search_space_path)
    search_space = resolve_search_space(config, search_space_payload)
    fixed_recommendations = dict(search_space_payload.get("fixed_recommendations") or {}) if config.get("use_fixed_recommendations", True) else {}
    fixed_recommendations.pop("_comment", None)
    fixed_overrides = dict(config.get("fixed_overrides") or {})
    metric_name = str(args.metric or config.get("metric") or "map50_95")
    direction = str(config.get("direction") or "maximize")
    n_trials = int(args.n_trials if args.n_trials is not None else config.get("n_trials", 20))
    timeout_seconds = args.timeout_seconds if args.timeout_seconds is not None else config.get("timeout_seconds")
    timeout_seconds = int(timeout_seconds) if timeout_seconds not in (None, "") else None
    sampler = resolve_sampler(optuna_mod, dict(config.get("sampler") or {}), args.sampler)
    pruner = resolve_pruner(optuna_mod, dict(config.get("pruner") or {}))

    study = optuna_mod.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )

    if args.enqueue_baseline:
        baseline_params = build_enqueued_baseline_params(
            search_space=search_space,
            base_config=base_config,
            fixed_recommendations=fixed_recommendations,
        )
        if baseline_params:
            study.enqueue_trial(baseline_params)

    write_json(
        study_dir / "resolved_config.json",
        {
            "study_name": study_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path) if config_path.exists() else None,
            "base_config_path": str(base_config_path),
            "search_space_path": str(search_space_path),
            "train_script_path": str(train_script_path),
            "storage": storage,
            "metric": metric_name,
            "direction": direction,
            "n_trials": n_trials,
            "timeout_seconds": timeout_seconds,
            "search_space": search_space,
            "fixed_recommendations": fixed_recommendations,
            "fixed_overrides": fixed_overrides,
        },
    )

    subprocess_timeout = config.get("subprocess_timeout_seconds")
    subprocess_timeout = int(subprocess_timeout) if subprocess_timeout not in (None, "") else None
    continue_on_failure = bool(config.get("continue_on_trial_failure", True))

    def objective(trial: Any) -> float:
        trial_overrides, sampled_overrides = build_trial_overrides(
            trial,
            base_config=base_config,
            search_space=search_space,
            fixed_overrides=fixed_overrides,
            fixed_recommendations=fixed_recommendations,
            study_name=study_name,
        )
        config_payload = build_trial_config(base_config, trial_overrides)

        trial_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_output_path = trial_dir / "config.json"
        write_json(config_output_path, config_payload)
        write_json(
            trial_dir / "trial_request.json",
            {
                "trial_number": trial.number,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "metric": metric_name,
                "sampled_overrides": sampled_overrides,
                "resolved_overrides": trial_overrides,
            },
        )

        command = [sys.executable, str(train_script_path), "--config", str(config_output_path)]
        started_at = datetime.now()
        print_trial_banner(
            trial_number=trial.number,
            total_trials=n_trials,
            metric_name=metric_name,
            study_name=study_name,
            trial_dir=trial_dir,
            sampled_overrides=sampled_overrides,
        )
        try:
            output_text = stream_subprocess_output(
                command,
                cwd=repo_root,
                log_path=trial_dir / "train.log",
                timeout_seconds=subprocess_timeout,
                display_prefix=f"[Trial {trial.number + 1}/{n_trials}] ",
            )
        except subprocess.CalledProcessError as exc:
            print(f"[Optuna] Trial {trial.number:04d} failed with returncode={exc.returncode}. Log: {trial_dir / 'train.log'}")
            write_json(
                trial_dir / "trial_result.json",
                {
                    "trial_number": trial.number,
                    "status": "failed",
                    "command": command,
                    "returncode": exc.returncode,
                    "started_at": started_at.isoformat(timespec="seconds"),
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            raise
        except subprocess.TimeoutExpired as exc:
            log_text = str(exc.output or exc.stdout or "")
            if log_text:
                (trial_dir / "train.log").write_text(log_text, encoding="utf-8")
            print(
                f"[Optuna] Trial {trial.number:04d} timed out after {subprocess_timeout}s. "
                f"Log: {trial_dir / 'train.log'}"
            )
            write_json(
                trial_dir / "trial_result.json",
                {
                    "trial_number": trial.number,
                    "status": "timeout",
                    "command": command,
                    "timeout_seconds": subprocess_timeout,
                    "started_at": started_at.isoformat(timespec="seconds"),
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            raise

        run_dir, version = parse_train_output(output_text)
        run_summary = load_run_summary_from_result(
            repo_root=repo_root,
            run_dir=run_dir,
            report_group=str(trial_overrides["report_group"]),
            report_label=str(trial_overrides["report_label"]),
            started_at=started_at,
        )
        score = resolve_metric_value(run_summary, metric_name)
        if score is None:
            write_json(
                trial_dir / "trial_result.json",
                {
                    "trial_number": trial.number,
                    "status": "missing_metric",
                    "metric": metric_name,
                    "run_summary": run_summary,
                },
            )
            raise RuntimeError(f"Metric {metric_name} is missing in run_summary.json for trial {trial.number}.")

        trial.set_user_attr("metric_name", metric_name)
        trial.set_user_attr("report_label", trial_overrides["report_label"])
        trial.set_user_attr("version", version or run_summary.get("version"))
        trial.set_user_attr("train_run_dir", run_summary.get("train_run_dir"))
        trial.set_user_attr("weights", run_summary.get("weights"))
        trial.set_user_attr("resolved_overrides", trial_overrides)
        trial.set_user_attr("sampled_overrides", sampled_overrides)

        write_json(
            trial_dir / "trial_result.json",
            {
                "trial_number": trial.number,
                "status": "complete",
                "metric": metric_name,
                "value": score,
                "version": version or run_summary.get("version"),
                "train_run_dir": run_summary.get("train_run_dir"),
                "weights": run_summary.get("weights"),
                "sampled_overrides": sampled_overrides,
                "resolved_overrides": trial_overrides,
                "run_summary": run_summary,
                "started_at": started_at.isoformat(timespec="seconds"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(
            f"[Optuna] Trial {trial.number:04d} complete: {metric_name}={score:.6f}, "
            f"version={version or run_summary.get('version')}, run_dir={run_summary.get('train_run_dir')}"
        )
        return score

    catch_types = (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) if continue_on_failure else ()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, catch=catch_types)

    export_best_trial(study, study_dir, base_config)
    report_result = generate_study_report(study_dir, storage=storage, top_k=args.top_k_report)
    completed_trials = [trial for trial in study.trials if getattr(trial.state, "name", "") == "COMPLETE" and trial.value is not None]
    best_value_text = str(study.best_value) if completed_trials else "n/a"
    print(f"Study complete: {study.study_name}")
    print(f"Best value: {best_value_text}")
    print(f"Study directory: {study_dir}")
    print(f"Report directory: {report_result['report_dir']}")


if __name__ == "__main__":
    main()
