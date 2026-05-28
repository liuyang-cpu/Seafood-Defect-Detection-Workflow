from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "ultralytics").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate repo root from: {start}")


def sanitize_name(value: str, fallback: str = "artifact") -> str:
    text = SAFE_NAME_PATTERN.sub("_", str(value).strip()).strip("._-")
    return text or fallback


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def default_optuna_studies_root(repo_root: Path) -> Path:
    return repo_root / "src" / "train-optuna" / "runs" / "studies"


def default_optuna_config_path(repo_root: Path) -> Path:
    return repo_root / "src" / "train-optuna" / "youge_optuna_config.json"


def default_search_space_path(repo_root: Path) -> Path:
    return repo_root / "src" / "train" / "youge" / "train_youge_search_space.json"


def default_refiner_runs_root(module_root: Path) -> Path:
    return module_root / "runs" / "refine_runs"


def default_materialized_rounds_root(module_root: Path) -> Path:
    return module_root / "runs" / "materialized_rounds"


def format_run_stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H_%M_%S")


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


def build_trial_digest(row: dict[str, Any]) -> dict[str, Any]:
    params = dict(row.get("params") or {})
    return {
        "trial_number": row.get("trial_number"),
        "value": _to_float(row.get("value")),
        "state": row.get("state"),
        "version": row.get("version"),
        "metric_name": row.get("metric_name"),
        "precision": _to_float(row.get("precision")),
        "recall": _to_float(row.get("recall")),
        "map50": _to_float(row.get("map50")),
        "map50_95": _to_float(row.get("map50_95")),
        "final_map50_95": _to_float(row.get("final_map50_95")),
        "best_epoch": _to_float(row.get("best_epoch")),
        "epochs_completed": _to_float(row.get("epochs_completed")),
        "elapsed_seconds": _to_float(row.get("elapsed_seconds")),
        "params": params,
    }


def strip_payload_paths(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: strip_payload_paths(value)
            for key, value in payload.items()
            if key not in {"weights", "run_dir"}
        }
    if isinstance(payload, list):
        return [strip_payload_paths(item) for item in payload]
    return payload


def build_search_space_payload(search_space_payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "_comment",
        "baseline_version",
        "baseline_observed_args",
        "full_search_space",
        "fixed_recommendations",
        "priority_order",
        "notes",
    }
    notes = [
        note
        for note in list(search_space_payload.get("notes") or [])
        if "stage1_recommended_grid" not in str(note)
    ]
    payload = {
        key: strip_payload_paths(value)
        for key, value in search_space_payload.items()
        if key in allowed_keys and key != "notes"
    }
    if notes:
        payload["notes"] = notes
    return payload


def infer_param_trends(rows: list[dict[str, Any]], *, top_k: int = 10) -> dict[str, Any]:
    completed = [row for row in rows if str(row.get("state")) == "COMPLETE" and _to_float(row.get("value")) is not None]
    completed.sort(key=lambda row: _to_float(row.get("value")) or float("-inf"), reverse=True)
    top_rows = completed[: max(1, top_k)]
    param_values: dict[str, list[Any]] = {}
    for row in top_rows:
        for key, value in dict(row.get("params") or {}).items():
            param_values.setdefault(key, []).append(value)

    trends: dict[str, Any] = {}
    for key, values in param_values.items():
        unique_values = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        numeric_values = [_to_float(value) for value in values]
        numeric_values = [value for value in numeric_values if value is not None]
        trends[key] = {
            "top_k_values": values,
            "unique_values": unique_values,
            "numeric_min": min(numeric_values) if numeric_values else None,
            "numeric_max": max(numeric_values) if numeric_values else None,
            "true_count": sum(1 for value in values if _to_bool(value) is True),
            "false_count": sum(1 for value in values if _to_bool(value) is False),
        }
    return trends


def allowed_param_names(search_space_payload: dict[str, Any]) -> set[str]:
    full_space = dict(search_space_payload.get("full_search_space") or {})
    return {key for key in full_space.keys() if not key.startswith("_")}


def _validate_search_space_spec(param_name: str, spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    spec_type = spec.get("type")
    if spec_type == "categorical":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            errors.append(f"search_space_updates[{param_name}].values must be a non-empty array.")
    elif spec_type == "int":
        min_value = spec.get("min")
        max_value = spec.get("max")
        step = spec.get("step")
        if not isinstance(min_value, int) or not isinstance(max_value, int):
            errors.append(f"search_space_updates[{param_name}] int spec must use integer min/max.")
        elif min_value > max_value:
            errors.append(f"search_space_updates[{param_name}] min must not exceed max.")
        if step is not None and (not isinstance(step, int) or step <= 0):
            errors.append(f"search_space_updates[{param_name}].step must be a positive integer.")
    elif spec_type == "float":
        min_value = spec.get("min")
        max_value = spec.get("max")
        step = spec.get("step")
        scale = spec.get("scale")
        if not isinstance(min_value, (int, float)) or not isinstance(max_value, (int, float)):
            errors.append(f"search_space_updates[{param_name}] float spec must use numeric min/max.")
        elif float(min_value) > float(max_value):
            errors.append(f"search_space_updates[{param_name}] min must not exceed max.")
        if step is not None and (not isinstance(step, (int, float)) or float(step) <= 0):
            errors.append(f"search_space_updates[{param_name}].step must be a positive number.")
        if scale is not None and scale not in {"linear", "log"}:
            errors.append(f"search_space_updates[{param_name}].scale must be 'linear' or 'log'.")
    else:
        errors.append(f"search_space_updates[{param_name}] has invalid type: {spec_type}")
    return errors


def validate_refiner_response(
    *,
    response: dict[str, Any],
    schema_payload: dict[str, Any],
    base_optuna_config: dict[str, Any],
    search_space_payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(response, dict):
        return ["Refiner response must be a JSON object."]

    for key in schema_payload.get("required", []):
        if key not in response:
            errors.append(f"Missing required field: {key}")

    allowed_params = allowed_param_names(search_space_payload)
    enabled_params = response.get("enabled_params_next")
    if isinstance(enabled_params, list):
        for name in enabled_params:
            if name not in allowed_params:
                errors.append(f"enabled_params_next contains unsupported param: {name}")
    else:
        errors.append("enabled_params_next must be an array.")

    fixed_overrides = response.get("fixed_overrides_next")
    if isinstance(fixed_overrides, dict):
        for key in fixed_overrides:
            if key == "search_space_key":
                errors.append("fixed_overrides_next must not modify search_space_key.")
    else:
        errors.append("fixed_overrides_next must be an object.")

    search_space_updates = response.get("search_space_updates")
    if isinstance(search_space_updates, dict):
        for key, value in search_space_updates.items():
            if key not in allowed_params:
                errors.append(f"search_space_updates contains unsupported param: {key}")
                continue
            if not isinstance(value, dict):
                errors.append(f"search_space_updates[{key}] must be an object.")
                continue
            errors.extend(_validate_search_space_spec(key, value))
    else:
        errors.append("search_space_updates must be an object.")

    frozen_params = response.get("frozen_params")
    if isinstance(frozen_params, dict):
        for key in frozen_params:
            if key not in allowed_params:
                errors.append(f"frozen_params contains unsupported param: {key}")
    else:
        errors.append("frozen_params must be an object.")

    n_trials_next = response.get("n_trials_next")
    if not isinstance(n_trials_next, int) or n_trials_next <= 0:
        errors.append("n_trials_next must be a positive integer.")

    n_startup_trials_next = response.get("n_startup_trials_next")
    if not isinstance(n_startup_trials_next, int) or n_startup_trials_next <= 0:
        errors.append("n_startup_trials_next must be a positive integer.")
    elif isinstance(n_trials_next, int) and n_startup_trials_next > n_trials_next:
        errors.append("n_startup_trials_next must not exceed n_trials_next.")

    base_enabled = list(base_optuna_config.get("enabled_params") or [])
    if isinstance(enabled_params, list) and not enabled_params:
        errors.append("enabled_params_next must not be empty.")
    if isinstance(base_enabled, list) and isinstance(enabled_params, list):
        for name in enabled_params:
            if name not in allowed_params:
                errors.append(f"enabled param outside full_search_space: {name}")

    return errors


def materialize_next_round(
    *,
    response: dict[str, Any],
    current_optuna_config: dict[str, Any],
    current_search_space_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    next_optuna_config = json.loads(json.dumps(current_optuna_config))
    next_search_space_payload = json.loads(json.dumps(current_search_space_payload))

    next_optuna_config["enabled_params"] = list(response.get("enabled_params_next") or [])
    next_optuna_config["n_trials"] = int(response["n_trials_next"])
    sampler = dict(next_optuna_config.get("sampler") or {})
    sampler["n_startup_trials"] = int(response["n_startup_trials_next"])
    next_optuna_config["sampler"] = sampler

    fixed_overrides = dict(next_optuna_config.get("fixed_overrides") or {})
    fixed_overrides.update(dict(response.get("fixed_overrides_next") or {}))
    frozen_params = dict(response.get("frozen_params") or {})
    for key, value in frozen_params.items():
        if key not in fixed_overrides:
            fixed_overrides[key] = value
    next_optuna_config["fixed_overrides"] = fixed_overrides

    full_space = dict(next_search_space_payload.get("full_search_space") or {})
    for key, spec in dict(response.get("search_space_updates") or {}).items():
        full_space[key] = spec
    next_search_space_payload["full_search_space"] = full_space

    meta = {
        "observations": list(response.get("observations") or []),
        "search_space_updates": dict(response.get("search_space_updates") or {}),
        "enabled_params_next": list(response.get("enabled_params_next") or []),
        "n_trials_next": int(response["n_trials_next"]),
        "n_startup_trials_next": int(response["n_startup_trials_next"]),
    }
    return next_optuna_config, next_search_space_payload, meta
