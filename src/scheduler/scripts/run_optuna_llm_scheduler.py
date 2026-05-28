from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "cycle_name": "youge_optuna_llm_cycle",
    "python_executable": sys.executable,
    "max_rounds": 3,
    "target_metric_threshold": None,
    "stop_direction": None,
    "initial_optuna_config_path": "src/train-optuna/youge_optuna_config.json",
    "initial_search_space_path": "src/train/youge/train_youge_search_space.json",
    "base_train_config_path": None,
    "train_script_path": None,
    "optuna_top_k_report": 20,
    "enqueue_baseline": False,
    "llm_refiner": {
        "enabled": True,
        "config_path": "src/llm-param-space-opt/config/llm_space_refiner.config.json",
        "provider": None,
        "model": None,
        "base_url": None,
        "top_k": 10,
        "failed_k": 10,
    },
}

STUDY_RESPONSE_PATTERN = "Refiner response written to:"
ROUND_DIR_PATTERN = "Materialized next round directory:"
NEXT_OPTUNA_PATTERN = "Next optuna config:"
NEXT_SPACE_PATTERN = "Next search space:"


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "ultralytics").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate repo root from: {start}")


def sanitize_name(value: str, fallback: str = "artifact") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value).strip()).strip("._-")
    return cleaned or fallback


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-round Optuna -> LLM -> next-round scheduling.")
    parser.add_argument("--config", type=str, default=None, help="Scheduler config JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and planned paths without executing them.")
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    payload = json.loads(json.dumps(DEFAULT_CONFIG))
    incoming = load_json(path)
    payload.update({key: value for key, value in incoming.items() if key != "llm_refiner"})
    payload["llm_refiner"].update(dict(incoming.get("llm_refiner") or {}))
    return payload


def resolve_path(repo_root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def format_stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H_%M_%S")


def stream_command(command: list[str], *, cwd: Path, log_path: Path, prefix: str) -> str:
    output_parts: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
                raise RuntimeError("Unable to read subprocess stdout.")
            for line in process.stdout:
                print(f"{prefix}{line}", end="")
                log_file.write(line)
                log_file.flush()
                output_parts.append(line)
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


def parse_prefixed_path(output_text: str, marker: str) -> Path | None:
    for line in output_text.splitlines():
        if marker in line:
            _, value = line.split(marker, 1)
            candidate = value.strip()
            if candidate:
                return Path(candidate).resolve()
    return None


def should_stop(*, best_value: float | None, threshold: float | None, direction: str) -> bool:
    if threshold is None or best_value is None:
        return False
    direction_normalized = direction.lower()
    if direction_normalized == "minimize":
        return best_value <= threshold
    return best_value >= threshold


def build_optuna_command(
    *,
    python_executable: str,
    repo_root: Path,
    study_name: str,
    optuna_config_path: Path,
    search_space_path: Path,
    base_train_config_path: Path | None,
    train_script_path: Path | None,
    top_k_report: int,
    enqueue_baseline: bool,
) -> list[str]:
    command = [
        python_executable,
        str((repo_root / "src" / "train-optuna" / "run_youge_optuna_search.py").resolve()),
        "--study-name",
        study_name,
        "--config",
        str(optuna_config_path),
        "--search-space",
        str(search_space_path),
        "--top-k-report",
        str(top_k_report),
    ]
    if base_train_config_path is not None:
        command.extend(["--base-config", str(base_train_config_path)])
    if train_script_path is not None:
        command.extend(["--train-script", str(train_script_path)])
    if enqueue_baseline:
        command.append("--enqueue-baseline")
    return command


def build_llm_command(
    *,
    python_executable: str,
    repo_root: Path,
    study_name: str,
    llm_config_path: Path | None,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    optuna_config_path: Path,
    search_space_path: Path,
    top_k: int,
    failed_k: int,
) -> list[str]:
    command = [
        python_executable,
        str((repo_root / "src" / "llm-param-space-opt" / "scripts" / "llm_refine_optuna_space.py").resolve()),
        "--study-name",
        study_name,
        "--optuna-config",
        str(optuna_config_path),
        "--search-space",
        str(search_space_path),
        "--top-k",
        str(top_k),
        "--failed-k",
        str(failed_k),
    ]
    if llm_config_path is not None:
        command.extend(["--config", str(llm_config_path)])
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["--model", model])
    if base_url:
        command.extend(["--base-url", base_url])
    return command


def build_materialize_command(
    *,
    python_executable: str,
    repo_root: Path,
    response_path: Path,
    optuna_config_path: Path,
    search_space_path: Path,
    output_root: Path,
) -> list[str]:
    return [
        python_executable,
        str((repo_root / "src" / "llm-param-space-opt" / "scripts" / "materialize_next_round.py").resolve()),
        "--response",
        str(response_path),
        "--optuna-config",
        str(optuna_config_path),
        "--search-space",
        str(search_space_path),
        "--output-root",
        str(output_root),
    ]


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)

    config_path = Path(args.config).resolve() if args.config else (repo_root / "src" / "scheduler" / "config" / "optuna_llm_cycle.config.json")
    config = load_config(config_path if config_path.exists() else None)

    cycle_name = sanitize_name(str(config.get("cycle_name") or "optuna_llm_cycle"), fallback="optuna_llm_cycle")
    python_executable = str(config.get("python_executable") or sys.executable)
    max_rounds = int(config.get("max_rounds") or 1)
    target_metric_threshold = config.get("target_metric_threshold")
    target_metric_threshold = float(target_metric_threshold) if target_metric_threshold not in (None, "") else None

    initial_optuna_config_path = resolve_path(repo_root, str(config.get("initial_optuna_config_path") or ""))
    initial_search_space_path = resolve_path(repo_root, str(config.get("initial_search_space_path") or ""))
    base_train_config_path = resolve_path(repo_root, config.get("base_train_config_path"))
    train_script_path = resolve_path(repo_root, config.get("train_script_path"))
    llm_config_path = resolve_path(repo_root, (config.get("llm_refiner") or {}).get("config_path"))

    if initial_optuna_config_path is None or not initial_optuna_config_path.exists():
        raise FileNotFoundError(f"Initial optuna config not found: {initial_optuna_config_path}")
    if initial_search_space_path is None or not initial_search_space_path.exists():
        raise FileNotFoundError(f"Initial search space not found: {initial_search_space_path}")

    initial_optuna_config = load_json(initial_optuna_config_path)
    stop_direction = str(config.get("stop_direction") or initial_optuna_config.get("direction") or "maximize")

    cycle_root = (repo_root / "src" / "scheduler" / "runs" / f"{cycle_name}_{format_stamp(datetime.now())}").resolve()
    cycle_root.mkdir(parents=True, exist_ok=True)
    write_json(
        cycle_root / "resolved_scheduler_config.json",
        {
            **config,
            "config_path": str(config_path) if config_path.exists() else None,
            "initial_optuna_config_path": str(initial_optuna_config_path),
            "initial_search_space_path": str(initial_search_space_path),
            "base_train_config_path": str(base_train_config_path) if base_train_config_path else None,
            "train_script_path": str(train_script_path) if train_script_path else None,
            "llm_refiner": {
                **dict(config.get("llm_refiner") or {}),
                "config_path": str(llm_config_path) if llm_config_path else None,
            },
        },
    )

    current_optuna_config_path = initial_optuna_config_path
    current_search_space_path = initial_search_space_path
    rounds_summary: list[dict[str, Any]] = []

    print(f"[Scheduler] Cycle root: {cycle_root}")
    print(f"[Scheduler] Max rounds: {max_rounds}")
    print(f"[Scheduler] Stop threshold: {target_metric_threshold} ({stop_direction})")

    for round_index in range(1, max_rounds + 1):
        round_dir = cycle_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        study_name = f"{cycle_name}_round{round_index:02d}"

        round_meta: dict[str, Any] = {
            "round_index": round_index,
            "study_name": study_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "input_optuna_config_path": str(current_optuna_config_path),
            "input_search_space_path": str(current_search_space_path),
        }
        write_json(round_dir / "round_meta.json", round_meta)

        optuna_command = build_optuna_command(
            python_executable=python_executable,
            repo_root=repo_root,
            study_name=study_name,
            optuna_config_path=current_optuna_config_path,
            search_space_path=current_search_space_path,
            base_train_config_path=base_train_config_path,
            train_script_path=train_script_path,
            top_k_report=int(config.get("optuna_top_k_report") or 20),
            enqueue_baseline=bool(config.get("enqueue_baseline", False)),
        )
        round_meta["optuna_command"] = optuna_command

        if args.dry_run:
            print(f"[Scheduler] Round {round_index}: {' '.join(optuna_command)}")
            if round_index == 1:
                llm_refiner = dict(config.get("llm_refiner") or {})
                if bool(llm_refiner.get("enabled", True)):
                    llm_command = build_llm_command(
                        python_executable=python_executable,
                        repo_root=repo_root,
                        study_name=study_name,
                        llm_config_path=llm_config_path,
                        provider=llm_refiner.get("provider"),
                        model=llm_refiner.get("model"),
                        base_url=llm_refiner.get("base_url"),
                        optuna_config_path=current_optuna_config_path,
                        search_space_path=current_search_space_path,
                        top_k=int(llm_refiner.get("top_k") or 10),
                        failed_k=int(llm_refiner.get("failed_k") or 10),
                    )
                    print(f"[Scheduler] Round {round_index} LLM: {' '.join(llm_command)}")
                    print(
                        "[Scheduler] Dry-run stops after round 1 planning because "
                        "later rounds depend on materialized outputs from the previous round."
                    )
            rounds_summary.append(round_meta)
            break

        optuna_output = stream_command(
            optuna_command,
            cwd=repo_root,
            log_path=round_dir / "optuna_round.log",
            prefix=f"[Scheduler][Round {round_index}][Optuna] ",
        )

        study_dir = (repo_root / "src" / "train-optuna" / "runs" / "studies" / study_name).resolve()
        best_trial_path = study_dir / "best_trial.json"
        if not best_trial_path.exists():
            raise FileNotFoundError(f"Best trial file not found after Optuna round: {best_trial_path}")
        best_trial = load_json(best_trial_path)
        best_value = best_trial.get("value")
        if isinstance(best_value, (int, float)):
            best_value = float(best_value)
        else:
            best_value = None

        round_meta.update(
            {
                "study_dir": str(study_dir),
                "best_trial_path": str(best_trial_path),
                "best_trial_value": best_value,
                "optuna_finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_json(round_dir / "round_meta.json", round_meta)

        if should_stop(best_value=best_value, threshold=target_metric_threshold, direction=stop_direction):
            round_meta["stop_reason"] = "target_metric_threshold_reached"
            round_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(round_dir / "round_meta.json", round_meta)
            rounds_summary.append(round_meta)
            break

        llm_refiner = dict(config.get("llm_refiner") or {})
        if not bool(llm_refiner.get("enabled", True)):
            round_meta["stop_reason"] = "llm_refiner_disabled"
            round_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(round_dir / "round_meta.json", round_meta)
            rounds_summary.append(round_meta)
            break

        llm_command = build_llm_command(
            python_executable=python_executable,
            repo_root=repo_root,
            study_name=study_name,
            llm_config_path=llm_config_path,
            provider=llm_refiner.get("provider"),
            model=llm_refiner.get("model"),
            base_url=llm_refiner.get("base_url"),
            optuna_config_path=current_optuna_config_path,
            search_space_path=current_search_space_path,
            top_k=int(llm_refiner.get("top_k") or 10),
            failed_k=int(llm_refiner.get("failed_k") or 10),
        )
        round_meta["llm_command"] = llm_command
        write_json(round_dir / "round_meta.json", round_meta)

        llm_output = stream_command(
            llm_command,
            cwd=repo_root,
            log_path=round_dir / "llm_refiner.log",
            prefix=f"[Scheduler][Round {round_index}][LLM] ",
        )
        response_path = parse_prefixed_path(llm_output, STUDY_RESPONSE_PATTERN)
        if response_path is None or not response_path.exists():
            raise FileNotFoundError("Unable to locate refiner_response.json from LLM refiner output.")

        materialize_output_root = round_dir / "materialized_rounds"
        materialize_command = build_materialize_command(
            python_executable=python_executable,
            repo_root=repo_root,
            response_path=response_path,
            optuna_config_path=current_optuna_config_path,
            search_space_path=current_search_space_path,
            output_root=materialize_output_root,
        )
        round_meta["materialize_command"] = materialize_command
        round_meta["refiner_response_path"] = str(response_path)
        write_json(round_dir / "round_meta.json", round_meta)

        materialize_output = stream_command(
            materialize_command,
            cwd=repo_root,
            log_path=round_dir / "materialize.log",
            prefix=f"[Scheduler][Round {round_index}][Materialize] ",
        )
        materialized_round_dir = parse_prefixed_path(materialize_output, ROUND_DIR_PATTERN)
        next_optuna_config_path = parse_prefixed_path(materialize_output, NEXT_OPTUNA_PATTERN)
        next_search_space_path = parse_prefixed_path(materialize_output, NEXT_SPACE_PATTERN)
        if materialized_round_dir is None or next_optuna_config_path is None or next_search_space_path is None:
            raise FileNotFoundError("Unable to parse materialized round output paths.")

        round_meta.update(
            {
                "materialized_round_dir": str(materialized_round_dir),
                "next_optuna_config_path": str(next_optuna_config_path),
                "next_search_space_path": str(next_search_space_path),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_json(round_dir / "round_meta.json", round_meta)
        rounds_summary.append(round_meta)

        current_optuna_config_path = next_optuna_config_path
        current_search_space_path = next_search_space_path

    cycle_summary = {
        "cycle_name": cycle_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "max_rounds": max_rounds,
        "target_metric_threshold": target_metric_threshold,
        "stop_direction": stop_direction,
        "rounds": rounds_summary,
        "latest_optuna_config_path": str(current_optuna_config_path),
        "latest_search_space_path": str(current_search_space_path),
    }
    write_json(cycle_root / "cycle_summary.json", cycle_summary)
    print(f"[Scheduler] Cycle summary written to: {cycle_root / 'cycle_summary.json'}")


if __name__ == "__main__":
    main()
