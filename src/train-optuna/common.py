from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

DIRECT_CONFIG_KEYS = {
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
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "ultralytics").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate repo root from: {start}")


def train_youge_dir(repo_root: Path) -> Path:
    return repo_root / "src" / "train" / "youge"


def default_base_config_json(repo_root: Path) -> Path:
    return train_youge_dir(repo_root) / "train_youge.json"


def default_search_space_json(repo_root: Path) -> Path:
    return train_youge_dir(repo_root) / "train_youge_search_space.json"


def default_train_script(repo_root: Path) -> Path:
    return train_youge_dir(repo_root) / "train_youge.py"


def default_runs_train_dir(repo_root: Path) -> Path:
    return train_youge_dir(repo_root) / "runs" / "train"


def default_studies_root(module_dir: Path) -> Path:
    return module_dir / "runs" / "studies"


def default_study_name(prefix: str = "youge_optuna") -> str:
    return f"{sanitize_name(prefix)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def sanitize_name(value: str, fallback: str = "study") -> str:
    text = SAFE_NAME_PATTERN.sub("_", str(value).strip()).strip("._-")
    return text or fallback


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    direct_config: dict[str, Any] = {}
    extra_train_args: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in DIRECT_CONFIG_KEYS:
            direct_config[key] = value
        else:
            extra_train_args[key] = value
    return direct_config, extra_train_args


def apply_overrides(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    config_payload = dict(base_config)
    direct_overrides, extra_overrides = split_overrides(overrides)
    config_payload.update(direct_overrides)
    merged_extra = dict(config_payload.get("extra_train_args") or {})
    merged_extra.update(extra_overrides)
    if merged_extra:
        config_payload["extra_train_args"] = merged_extra
    return config_payload


def resolve_metric_value(run_summary: dict[str, Any], metric_name: str) -> float | None:
    metric_map = {
        "map50_95": ("best_metrics", "map50_95"),
        "map50": ("best_metrics", "map50"),
        "precision": ("best_metrics", "precision"),
        "recall": ("best_metrics", "recall"),
        "final_map50_95": ("final_metrics", "map50_95"),
    }
    if metric_name not in metric_map:
        raise KeyError(f"Unsupported metric: {metric_name}")
    group_key, metric_key = metric_map[metric_name]
    raw = ((run_summary.get(group_key) or {}) if isinstance(run_summary, dict) else {}).get(metric_key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
