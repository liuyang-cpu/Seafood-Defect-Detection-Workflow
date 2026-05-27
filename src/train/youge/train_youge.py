from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "ultralytics").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate repo root from: {start}")


def resolve_output_dir(base_dir: Path, value: str) -> str:
    output_dir = Path(value)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    return str(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11n on the youge dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted training run from a checkpoint.")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a checkpoint such as weights/last.pt.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size. Use -1 for auto-batch.")
    parser.add_argument("--device", type=str, default=None, help="Training device, e.g. 0, 0,1 or cpu.")
    parser.add_argument("--workers", type=int, default=None, help="Number of dataloader workers.")
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name.")
    parser.add_argument("--version", type=str, default=None, help="Model version token, e.g. version001.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability.")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory.")
    parser.add_argument("--report-group", type=str, default=None, help="Optional report group id for aggregated comparison.")
    parser.add_argument("--report-label", type=str, default=None, help="Optional short label for this experiment.")
    parser.add_argument("--report-description", type=str, default=None, help="Optional description for this experiment.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Automatic Mixed Precision.",
    )
    return parser.parse_args()


def load_train_config(config_path: Path | None) -> dict:
    defaults = {
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "device": "0",
        "workers": 8,
        "project": "runs/train",
        "name": "youge_yolo11n",
        "version": None,
        "patience": 50,
        "mosaic": 0.0,
        "cache": False,
        "exist_ok": False,
        "amp": True,
        "report_group": None,
        "report_label": None,
        "report_description": None,
        "extra_train_args": {},
    }

    if config_path is None:
        return defaults

    with config_path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)

    defaults.update(user_config)
    return defaults


def merge_args_into_config(args: argparse.Namespace, config: dict) -> dict:
    for key in (
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
        "amp",
        "report_group",
        "report_label",
        "report_description",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if args.cache:
        config["cache"] = True
    if args.exist_ok:
        config["exist_ok"] = True

    return config


def make_runtime_dataset_yaml(source_yaml: Path, dataset_root: Path, output_yaml: Path) -> Path:
    with source_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    runtime_data = {
        "path": dataset_root.as_posix(),
        "train": data.get("train", "images/train"),
        "val": data.get("val", "images/val"),
        "names": data["names"],
    }

    if "test" in data:
        runtime_data["test"] = data["test"]

    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_data, f, allow_unicode=True, sort_keys=False)

    return output_yaml


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def build_run_summary(*, train_run_dir: Path, train_config: dict, metadata: dict, dataset_yaml: Path, source_config_path: Path) -> dict:
    results_csv = train_run_dir / "results.csv"
    rows: list[dict] = []
    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    best_row = None
    if rows:
        best_row = max(rows, key=lambda row: _to_float(row.get("metrics/mAP50-95(B)")) or float("-inf"))
    last_row = rows[-1] if rows else None

    summary = {
        "species": metadata["species"],
        "version": metadata["version"],
        "train_run_name": metadata["train_run_name"],
        "train_run_dir": metadata["train_run_dir"],
        "weights": metadata["weights"],
        "last_weights": metadata["last_weights"],
        "source_model": metadata["source_model"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_yaml": str(dataset_yaml),
        "source_config_path": str(source_config_path),
        "report_group": train_config.get("report_group"),
        "report_label": train_config.get("report_label"),
        "report_description": train_config.get("report_description"),
        "train_config": train_config,
        "best_epoch": int(float(best_row["epoch"])) if best_row and best_row.get("epoch") else None,
        "best_metrics": {
            "precision": _to_float(best_row.get("metrics/precision(B)")) if best_row else None,
            "recall": _to_float(best_row.get("metrics/recall(B)")) if best_row else None,
            "map50": _to_float(best_row.get("metrics/mAP50(B)")) if best_row else None,
            "map50_95": _to_float(best_row.get("metrics/mAP50-95(B)")) if best_row else None,
        },
        "final_metrics": {
            "precision": _to_float(last_row.get("metrics/precision(B)")) if last_row else None,
            "recall": _to_float(last_row.get("metrics/recall(B)")) if last_row else None,
            "map50": _to_float(last_row.get("metrics/mAP50(B)")) if last_row else None,
            "map50_95": _to_float(last_row.get("metrics/mAP50-95(B)")) if last_row else None,
        },
        "elapsed_seconds": _to_float(last_row.get("time")) if last_row else None,
        "epochs_completed": int(float(last_row["epoch"])) if last_row and last_row.get("epoch") else None,
        "results_csv": str(results_csv) if results_csv.exists() else None,
    }
    return summary


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if not args.resume and not args.resume_from:
        return None
    if not args.resume_from:
        raise ValueError("Resume mode requires --resume-from pointing to a checkpoint such as weights/last.pt")
    resume_path = Path(args.resume_from).resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    return resume_path


def load_resume_train_config(resume_checkpoint: Path) -> dict:
    run_dir = resume_checkpoint.resolve().parent.parent
    args_yaml = run_dir / "args.yaml"
    config = load_train_config(None)
    config.update(
        {
            "project": str(run_dir.parent),
            "name": run_dir.name,
            "version": None,
            "exist_ok": True,
        }
    )

    saved_args = {}
    if args_yaml.exists():
        with args_yaml.open("r", encoding="utf-8") as f:
            saved_args = yaml.safe_load(f) or {}

    config["epochs"] = saved_args.get("epochs", config["epochs"])
    config["imgsz"] = saved_args.get("imgsz", config["imgsz"])
    config["batch"] = saved_args.get("batch", config["batch"])
    config["device"] = str(saved_args.get("device", config["device"]))
    config["workers"] = saved_args.get("workers", config["workers"])
    config["patience"] = saved_args.get("patience", config["patience"])
    config["mosaic"] = saved_args.get("mosaic", config["mosaic"])
    config["cache"] = bool(saved_args.get("cache", config["cache"]))
    config["amp"] = bool(saved_args.get("amp", config["amp"]))

    extra_train_args = {}
    for key in ("translate", "scale", "erasing", "mixup", "copy_paste", "degrees", "shear", "perspective"):
        if key in saved_args:
            extra_train_args[key] = saved_args[key]
    config["extra_train_args"] = extra_train_args
    return config


def refresh_train_config_from_args_yaml(args_yaml: Path, train_config: dict) -> dict:
    if not args_yaml.exists():
        return train_config

    with args_yaml.open("r", encoding="utf-8") as f:
        saved_args = yaml.safe_load(f) or {}

    refreshed = dict(train_config)
    for key in ("epochs", "imgsz", "batch", "device", "workers", "patience", "mosaic", "cache", "amp"):
        if key in saved_args:
            refreshed[key] = saved_args[key]

    extra = dict(refreshed.get("extra_train_args") or {})
    for key in ("translate", "scale", "erasing", "mixup", "copy_paste", "degrees", "shear", "perspective"):
        if key in saved_args:
            extra[key] = saved_args[key]
    refreshed["extra_train_args"] = extra
    return refreshed


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from ultralytics import YOLO
    from src.youge_versioning import build_versioned_name, get_next_version, normalize_version, register_training_artifacts

    dataset_root = repo_root / "datasets" / "data" / "youge"
    source_yaml = dataset_root / "dataset.yaml"
    model_path = repo_root / "yolo11n.pt"
    runtime_yaml = script_path.with_name("youge_dataset.runtime.yaml")
    config_path = Path(args.config).resolve() if args.config else script_path.with_name("train_youge.json")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    if not source_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if args.config and not config_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {config_path}")

    runtime_yaml = make_runtime_dataset_yaml(source_yaml, dataset_root, runtime_yaml)
    resume_checkpoint = resolve_resume_checkpoint(args)
    if resume_checkpoint:
        train_config = load_resume_train_config(resume_checkpoint)
    else:
        train_config = load_train_config(config_path if config_path.exists() else None)
    train_config = merge_args_into_config(args, train_config)
    train_config["project"] = resolve_output_dir(script_path.parent, train_config["project"])
    model_output_dir = repo_root / "src" / "opt" / "youge" / "model-output"
    from src.youge_versioning import extract_version_from_text

    version = normalize_version(train_config.get("version"))
    if resume_checkpoint:
        version = version or extract_version_from_text(Path(train_config["name"]).name)
    version = version or get_next_version(model_output_dir)
    train_config["version"] = version
    if not resume_checkpoint:
        train_config["name"] = build_versioned_name(str(train_config["name"]), version)

    raw_extra_train_args = train_config.get("extra_train_args") or {}
    if not isinstance(raw_extra_train_args, dict):
        raise TypeError("extra_train_args must be a JSON object")
    extra_train_args = dict(raw_extra_train_args)

    if resume_checkpoint:
        model = YOLO(str(resume_checkpoint))
        train_kwargs = {
            "resume": True,
            "device": train_config["device"],
            "imgsz": train_config["imgsz"],
            "batch": train_config["batch"],
            "workers": train_config["workers"],
        }
    else:
        model = YOLO(str(model_path))
        train_kwargs = {
            "data": str(runtime_yaml),
            "epochs": train_config["epochs"],
            "imgsz": train_config["imgsz"],
            "batch": train_config["batch"],
            "device": train_config["device"],
            "workers": train_config["workers"],
            "project": train_config["project"],
            "name": train_config["name"],
            "patience": train_config["patience"],
            "mosaic": train_config["mosaic"],
            "cache": train_config["cache"],
            "exist_ok": train_config["exist_ok"],
            "amp": train_config["amp"],
        }
        train_kwargs.update(extra_train_args)
    results = model.train(**train_kwargs)
    final_train_config = refresh_train_config_from_args_yaml(Path(results.save_dir) / "args.yaml", train_config)

    metadata = register_training_artifacts(
        repo_root=repo_root,
        species="youge",
        version=version,
        train_run_name=train_config["name"],
        train_run_dir=Path(results.save_dir),
        source_model_name=model_path.name,
        overwrite_existing=bool(resume_checkpoint),
    )
    run_summary = build_run_summary(
        train_run_dir=Path(results.save_dir),
        train_config=final_train_config,
        metadata=metadata,
        dataset_yaml=runtime_yaml,
        source_config_path=config_path,
    )
    (Path(results.save_dir) / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Training finished. Version={version}. Results saved to: {results.save_dir}")
    print(f"Versioned model saved to: {metadata['weights']}")


if __name__ == "__main__":
    main()
