from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ALLOWED_PLANNER_PARAMS = ("imgsz", "epochs", "mosaic", "translate", "scale", "erasing", "amp")
DEFAULT_SCHEMA_NAME = "youge_training_plan"
SAFE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "ultralytics").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate repo root from: {start}")


def train_youge_dir(repo_root: Path) -> Path:
    return repo_root / "src" / "train" / "youge"


def default_summary_csv(repo_root: Path) -> Path:
    return train_youge_dir(repo_root) / "runs" / "reports" / "all_runs_latest" / "summary.csv"


def default_search_space_json(repo_root: Path) -> Path:
    return train_youge_dir(repo_root) / "train_youge_search_space.json"


def default_base_config_json(repo_root: Path) -> Path:
    return train_youge_dir(repo_root) / "train_youge.json"


def default_runs_train_dir(repo_root: Path) -> Path:
    return train_youge_dir(repo_root) / "runs" / "train"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_summary_rows(summary_csv: Path, max_history: int | None = None) -> list[dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if max_history is None:
        return rows
    return rows[: max(0, max_history)]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def load_history_from_run_summaries(runs_train_dir: Path, max_history: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not runs_train_dir.exists():
        return rows

    run_dirs = sorted(
        (path for path in runs_train_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        payload = load_json(summary_path)
        rows.append(payload)
        if max_history is not None and len(rows) >= max_history:
            break
    return rows


def build_history_summary(history_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for payload in history_payloads:
        train_config = dict(payload.get("train_config") or {})
        extra = dict(train_config.get("extra_train_args") or {})
        result.append(
            {
                "label": payload.get("report_label") or payload.get("version") or payload.get("train_run_name"),
                "version": payload.get("version"),
                "run_name": payload.get("train_run_name"),
                "metrics": dict(payload.get("best_metrics") or {}),
                "epochs_completed": payload.get("epochs_completed"),
                "best_epoch": payload.get("best_epoch"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "train_config": {
                    "epochs": train_config.get("epochs"),
                    "imgsz": train_config.get("imgsz"),
                    "batch": train_config.get("batch"),
                    "workers": train_config.get("workers"),
                    "patience": train_config.get("patience"),
                    "mosaic": train_config.get("mosaic"),
                    "amp": train_config.get("amp"),
                    "extra_train_args": {
                        key: extra.get(key)
                        for key in (
                            "translate",
                            "scale",
                            "erasing",
                            "mixup",
                            "copy_paste",
                            "degrees",
                            "shear",
                            "perspective",
                        )
                        if key in extra
                    },
                },
            }
        )
    return result


def build_search_space_payload(search_space_payload: dict[str, Any]) -> dict[str, Any]:
    full = dict(search_space_payload.get("full_search_space") or {})
    allowed_params = [key for key in ALLOWED_PLANNER_PARAMS if key in full or key == "epochs"]
    param_space: dict[str, Any] = {}
    for key in allowed_params:
        if key in full:
            param_space[key] = full[key]
            continue
        if key == "epochs":
            param_space[key] = {"type": "categorical", "values": [30, 60, 90, 120]}
    return {
        "allowed_params": allowed_params,
        "param_space": param_space,
        "frozen_defaults": dict(search_space_payload.get("fixed_recommendations") or {}),
    }


def _same_history_run(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_version = left.get("version")
    right_version = right.get("version")
    if left_version and right_version and left_version == right_version:
        return True

    left_run_name = left.get("train_run_name") or left.get("run_name")
    right_run_name = right.get("train_run_name") or right.get("run_name")
    if left_run_name and right_run_name and left_run_name == right_run_name:
        return True
    return False


def build_planner_payload(
    *,
    task_context: dict[str, Any],
    baseline_run: dict[str, Any],
    search_space_payload: dict[str, Any],
    history_payloads: list[dict[str, Any]],
    budget_constraints: dict[str, Any] | None = None,
    planner_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filtered_history_payloads = [
        payload for payload in history_payloads if not _same_history_run(payload, baseline_run)
    ]
    return {
        "task_context": task_context,
        "baseline_run": baseline_run,
        "search_space": build_search_space_payload(search_space_payload),
        "history_summary": build_history_summary(filtered_history_payloads),
        "budget_constraints": budget_constraints
        or {
            "max_recommended_runs": 5,
            "max_epochs_per_run": 120,
            "max_imgsz": 896,
            "max_batch": 8,
            "allow_new_params": False,
            "require_non_duplicate_runs": True,
        },
        "planner_rules": planner_rules
        or {
            "must_stay_within_search_space": True,
            "must_not_repeat_history": True,
            "prefer_conservative_narrowing": True,
            "focus_on_business_problem_over_raw_map": True,
            "max_observations": 5,
            "max_recommended_runs": 5,
        },
    }


def default_task_context() -> dict[str, Any]:
    return {
        "project": "youge",
        "model_family": "YOLO11n",
        "dataset_name": "youge",
        "problem_summary": [
            "相邻帧截断框误分类",
            "边缘目标稳定性不足",
            "小目标更敏感",
        ],
        "primary_objective": "优先改善截断目标和边缘目标的检测稳定性，同时关注总体 mAP50-95",
        "notes": [
            "相机固定",
            "场景几何变化较小",
            "增强不宜过强",
        ],
    }


def choose_baseline_run(history_payloads: list[dict[str, Any]], baseline_version: str | None = None) -> dict[str, Any]:
    if not history_payloads:
        return {}
    if baseline_version:
        for payload in history_payloads:
            if payload.get("version") == baseline_version:
                return build_history_summary([payload])[0]

    def baseline_score(payload: dict[str, Any]) -> float:
        metrics = dict(payload.get("best_metrics") or {})
        score = _to_float(metrics.get("map50_95"))
        return score if score is not None else float("-inf")

    best_payload = max(history_payloads, key=baseline_score)
    if baseline_score(best_payload) == float("-inf"):
        best_payload = history_payloads[0]
    return build_history_summary([best_payload])[0]


def canonical_overrides(overrides: dict[str, Any]) -> str:
    normalized = {key: overrides[key] for key in sorted(overrides)}
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_history_override_fingerprints(history_payloads: list[dict[str, Any]]) -> set[str]:
    fingerprints: set[str] = set()
    for payload in history_payloads:
        config = dict(payload.get("train_config") or {})
        extra = dict(config.get("extra_train_args") or {})
        reduced: dict[str, Any] = {}
        for key in ALLOWED_PLANNER_PARAMS:
            if key in config:
                reduced[key] = config[key]
            elif key in extra:
                reduced[key] = extra[key]
        if reduced:
            fingerprints.add(canonical_overrides(reduced))
    return fingerprints


def validate_planner_response(
    *,
    planner_response: dict[str, Any],
    schema_payload: dict[str, Any],
    search_space_payload: dict[str, Any],
    history_payloads: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(planner_response, dict):
        return ["Planner response must be a JSON object."]

    for key in schema_payload.get("required", []):
        if key not in planner_response:
            errors.append(f"Missing required field: {key}")

    allowed_params = set(build_search_space_payload(search_space_payload).get("allowed_params") or [])
    historical_fingerprints = extract_history_override_fingerprints(history_payloads)
    current_fingerprints: set[str] = set()

    runs = planner_response.get("recommended_runs")
    if not isinstance(runs, list) or not runs:
        errors.append("recommended_runs must be a non-empty array.")
        return errors

    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            errors.append(f"Run #{index} must be an object.")
            continue
        label = run.get("label")
        if not isinstance(label, str) or not SAFE_LABEL_PATTERN.fullmatch(label):
            errors.append(f"Run #{index} has invalid label: {label}")

        overrides = run.get("overrides")
        if not isinstance(overrides, dict) or not overrides:
            errors.append(f"Run #{index} must include non-empty overrides.")
            continue

        reduced_overrides: dict[str, Any] = {}
        for key, value in overrides.items():
            if key not in allowed_params:
                errors.append(f"Run #{index} uses unsupported param: {key}")
                continue
            reduced_overrides[key] = value
            full_space = (search_space_payload.get("full_search_space") or {}).get(key)
            if full_space is None and key == "epochs":
                full_space = {"type": "categorical", "values": [30, 60, 90, 120]}
            if full_space is None:
                continue

            if full_space.get("type") == "categorical":
                if value not in full_space.get("values", []):
                    errors.append(f"Run #{index} value for {key} not in categorical values: {value}")
            elif full_space.get("type") in {"int", "float"}:
                numeric = _to_float(value)
                if numeric is None:
                    errors.append(f"Run #{index} value for {key} must be numeric.")
                    continue
                minimum = _to_float(full_space.get("min"))
                maximum = _to_float(full_space.get("max"))
                if minimum is not None and numeric < minimum:
                    errors.append(f"Run #{index} value for {key} below minimum: {value}")
                if maximum is not None and numeric > maximum:
                    errors.append(f"Run #{index} value for {key} above maximum: {value}")

        fingerprint = canonical_overrides(reduced_overrides)
        if fingerprint in historical_fingerprints:
            errors.append(f"Run #{index} duplicates a historical experiment.")
        if fingerprint in current_fingerprints:
            errors.append(f"Run #{index} duplicates another recommended run.")
        current_fingerprints.add(fingerprint)
    return errors


def materialize_plan(
    *,
    planner_response: dict[str, Any],
    plan_root: Path,
    base_config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    configs_dir = plan_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    direct_keys = {
        "epochs",
        "imgsz",
        "batch",
        "device",
        "workers",
        "project",
        "name",
        "version",
        "patience",
        "mosaic",
        "cache",
        "exist_ok",
        "amp",
        "report_group",
        "report_label",
        "report_description",
    }

    manifest_runs: list[dict[str, Any]] = []
    for index, run in enumerate(planner_response.get("recommended_runs") or [], start=1):
        config_payload = dict(base_config)
        config_payload["version"] = None
        config_payload["exist_ok"] = True
        config_payload["report_group"] = plan_root.name
        config_payload["report_label"] = run["label"]
        config_payload["report_description"] = run.get("reason") or ""

        extra_train_args = dict(config_payload.get("extra_train_args") or {})
        for key, value in dict(run.get("overrides") or {}).items():
            if key in direct_keys:
                config_payload[key] = value
            else:
                extra_train_args[key] = value
        if extra_train_args:
            config_payload["extra_train_args"] = extra_train_args

        config_path = configs_dir / f"exp_{index:03d}.json"
        write_json(config_path, config_payload)
        manifest_runs.append(
            {
                "index": index,
                "label": run["label"],
                "description": run.get("reason") or "",
                "config_path": str(config_path),
                "overrides": run.get("overrides") or {},
            }
        )

    manifest = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "plan": "llm-guided",
        "source": "train-circle",
        "base_config_path": metadata["base_config_path"],
        "search_space_path": metadata["search_space_path"],
        "planner_response_path": metadata["planner_response_path"],
        "payload_path": metadata["payload_path"],
        "observations": planner_response.get("observations") or [],
        "range_updates": planner_response.get("range_updates") or {},
        "frozen_params": planner_response.get("frozen_params") or {},
        "experiments": manifest_runs,
    }
    write_json(plan_root / "manifest.json", manifest)
    return manifest
