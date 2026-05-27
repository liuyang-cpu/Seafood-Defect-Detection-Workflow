from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


VERSION_PATTERN = re.compile(r"version(\d+)", re.IGNORECASE)


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "ultralytics").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate repo root from: {start}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate aggregate comparison reports for youge training runs.")
    parser.add_argument("--group", type=str, default=None, help="Optional report_group filter.")
    parser.add_argument("--runs-dir", type=str, default=None, help="Optional explicit training runs directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit report output directory.")
    parser.add_argument("--top-k", type=int, default=20, help="Maximum number of runs to include in the chart.")
    return parser.parse_args()


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


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _load_results_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _build_fallback_summary(run_dir: Path) -> dict[str, Any] | None:
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    rows = _load_results_rows(results_csv)
    if not rows:
        return None

    args_payload: dict[str, Any] = {}
    if args_yaml.exists():
        args_payload = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}

    best_row = max(rows, key=lambda row: _to_float(row.get("metrics/mAP50-95(B)")) or float("-inf"))
    last_row = rows[-1]

    version_match = VERSION_PATTERN.search(run_dir.name)
    version = f"version{int(version_match.group(1)):03d}" if version_match else None

    return {
        "species": "youge",
        "version": version,
        "train_run_name": run_dir.name,
        "train_run_dir": str(run_dir),
        "weights": str(run_dir / "weights" / "best.pt") if (run_dir / "weights" / "best.pt").exists() else None,
        "last_weights": str(run_dir / "weights" / "last.pt") if (run_dir / "weights" / "last.pt").exists() else None,
        "source_model": args_payload.get("model"),
        "created_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
        "dataset_yaml": args_payload.get("data"),
        "source_config_path": None,
        "report_group": None,
        "report_label": None,
        "report_description": None,
        "train_config": args_payload,
        "best_epoch": _to_int(best_row.get("epoch")),
        "best_metrics": {
            "precision": _to_float(best_row.get("metrics/precision(B)")),
            "recall": _to_float(best_row.get("metrics/recall(B)")),
            "map50": _to_float(best_row.get("metrics/mAP50(B)")),
            "map50_95": _to_float(best_row.get("metrics/mAP50-95(B)")),
        },
        "final_metrics": {
            "precision": _to_float(last_row.get("metrics/precision(B)")),
            "recall": _to_float(last_row.get("metrics/recall(B)")),
            "map50": _to_float(last_row.get("metrics/mAP50(B)")),
            "map50_95": _to_float(last_row.get("metrics/mAP50-95(B)")),
        },
        "elapsed_seconds": _to_float(last_row.get("time")),
        "epochs_completed": _to_int(last_row.get("epoch")),
        "results_csv": str(results_csv),
    }


def load_run_summaries(runs_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime):
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue

        fallback = _build_fallback_summary(run_dir)
        if fallback is not None:
            summaries.append(fallback)
    return summaries


def normalize_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    best_metrics = dict(summary.get("best_metrics") or {})
    final_metrics = dict(summary.get("final_metrics") or {})
    config = dict(summary.get("train_config") or {})
    label = summary.get("report_label") or summary.get("version") or summary.get("train_run_name")
    return {
        "label": label,
        "version": summary.get("version"),
        "run_name": summary.get("train_run_name"),
        "run_dir": summary.get("train_run_dir"),
        "report_group": summary.get("report_group"),
        "description": summary.get("report_description"),
        "epochs": summary.get("epochs_completed"),
        "best_epoch": summary.get("best_epoch"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "precision": best_metrics.get("precision"),
        "recall": best_metrics.get("recall"),
        "map50": best_metrics.get("map50"),
        "map50_95": best_metrics.get("map50_95"),
        "final_map50_95": final_metrics.get("map50_95"),
        "batch": config.get("batch"),
        "imgsz": config.get("imgsz"),
        "mosaic": config.get("mosaic"),
        "amp": config.get("amp"),
        "weights": summary.get("weights"),
        "results_csv": summary.get("results_csv"),
    }


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "label",
        "version",
        "run_name",
        "report_group",
        "description",
        "epochs",
        "best_epoch",
        "elapsed_seconds",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "final_map50_95",
        "batch",
        "imgsz",
        "mosaic",
        "amp",
        "weights",
        "results_csv",
        "run_dir",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]], output_path: Path, chart_name: str) -> None:
    lines = [
        "# Youge Training Comparison Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Total runs: {len(rows)}",
        "",
        f"![comparison]({chart_name})",
        "",
        "| Label | Version | mAP50-95 | Precision | Recall | Best Epoch | Epochs | Batch | ImgSz | Mosaic | AMP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {version} | {map50_95:.4f} | {precision:.4f} | {recall:.4f} | {best_epoch} | {epochs} | {batch} | {imgsz} | {mosaic} | {amp} |".format(
                label=row["label"],
                version=row["version"] or "-",
                map50_95=row["map50_95"] or 0.0,
                precision=row["precision"] or 0.0,
                recall=row["recall"] or 0.0,
                best_epoch=row["best_epoch"] or "-",
                epochs=row["epochs"] or "-",
                batch=row["batch"] or "-",
                imgsz=row["imgsz"] or "-",
                mosaic=row["mosaic"] or "-",
                amp=row["amp"] if row["amp"] is not None else "-",
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_chart(rows: list[dict[str, Any]], output_path: Path, top_k: int) -> None:
    import matplotlib.pyplot as plt

    selected = rows[: max(1, top_k)]
    labels = [row["label"] for row in selected]
    map_values = [row["map50_95"] or 0.0 for row in selected]
    precision_values = [row["precision"] or 0.0 for row in selected]
    recall_values = [row["recall"] or 0.0 for row in selected]

    fig, axes = plt.subplots(2, 1, figsize=(14, max(8, 0.6 * len(selected) + 4)), constrained_layout=True)

    axes[0].barh(labels, map_values, color="#2E86AB")
    axes[0].invert_yaxis()
    axes[0].set_title("Best mAP50-95 by Run")
    axes[0].set_xlabel("mAP50-95")
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    y_positions = range(len(selected))
    axes[1].barh([y - 0.2 for y in y_positions], precision_values, height=0.35, label="Precision", color="#F18F01")
    axes[1].barh([y + 0.2 for y in y_positions], recall_values, height=0.35, label="Recall", color="#C73E1D")
    axes[1].set_yticks(list(y_positions))
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_title("Best Precision / Recall by Run")
    axes[1].set_xlabel("Score")
    axes[1].legend()
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else script_path.parent / "runs" / "train"
    output_root = Path(args.output_dir).resolve() if args.output_dir else script_path.parent / "runs" / "reports"
    output_root.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Training runs directory not found: {runs_dir}")

    summaries = load_run_summaries(runs_dir)
    if args.group:
        summaries = [summary for summary in summaries if summary.get("report_group") == args.group]
    if not summaries:
        raise RuntimeError("No matching training runs found for report generation.")

    rows = [normalize_summary_row(summary) for summary in summaries]
    rows.sort(key=lambda row: (row["map50_95"] is not None, row["map50_95"] or -1.0), reverse=True)

    report_slug = args.group or "all_runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_root / f"{report_slug}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "summary.csv"
    md_path = report_dir / "report.md"
    chart_path = report_dir / "comparison.png"

    write_summary_csv(rows, csv_path)
    render_chart(rows, chart_path, top_k=args.top_k)
    render_markdown(rows, md_path, chart_path.name)

    latest_dir = output_root / f"{report_slug}_latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for name, source in {
        "summary.csv": csv_path,
        "report.md": md_path,
        "comparison.png": chart_path,
    }.items():
        (latest_dir / name).write_bytes(source.read_bytes())

    print(f"Report generated for {len(rows)} runs.")
    print(f"Report directory: {report_dir}")
    print(f"Latest snapshot: {latest_dir}")


if __name__ == "__main__":
    main()
