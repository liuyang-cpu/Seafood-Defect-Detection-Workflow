from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from common import default_studies_root, find_repo_root, load_json, sanitize_name, write_json

try:
    import optuna
except ImportError:
    optuna = None


def require_optuna() -> Any:
    if optuna is None:
        raise RuntimeError(
            f"Optuna is not installed for Python interpreter: {sys.executable}\n"
            "Activate the environment used for YOLO training, then install and rerun with:\n"
            "  python -m pip install optuna"
        )
    return optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a report for one youge Optuna study.")
    parser.add_argument("--study-dir", type=str, default=None, help="Explicit study directory path.")
    parser.add_argument("--study-name", type=str, default=None, help="Study name under src/train-optuna/runs/studies.")
    parser.add_argument("--storage", type=str, default=None, help="Explicit Optuna storage URL.")
    parser.add_argument("--top-k", type=int, default=20, help="Maximum number of completed trials to chart.")
    return parser.parse_args()


def resolve_study_dir(args: argparse.Namespace, module_dir: Path) -> Path:
    if args.study_dir:
        path = Path(args.study_dir).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Study directory not found: {path}")
        return path
    if not args.study_name:
        raise ValueError("Provide --study-dir or --study-name.")
    studies_root = default_studies_root(module_dir)
    path = (studies_root / sanitize_name(args.study_name)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Study directory not found: {path}")
    return path


def resolve_storage(study_dir: Path, explicit_storage: str | None) -> str:
    if explicit_storage:
        return explicit_storage
    return f"sqlite:///{(study_dir / 'study.db').resolve().as_posix()}"


def load_trial_result(trial_dir: Path) -> dict[str, Any]:
    result_path = trial_dir / "trial_result.json"
    if not result_path.exists():
        return {}
    return load_json(result_path)


def build_trial_rows(study: Any, study_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        trial_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
        trial_result = load_trial_result(trial_dir)
        run_summary = dict(trial_result.get("run_summary") or {})
        best_metrics = dict(run_summary.get("best_metrics") or {})
        final_metrics = dict(run_summary.get("final_metrics") or {})
        row = {
            "trial_number": trial.number,
            "state": getattr(trial.state, "name", str(trial.state)),
            "value": trial.value,
            "report_label": trial.user_attrs.get("report_label"),
            "version": trial.user_attrs.get("version"),
            "run_dir": trial.user_attrs.get("train_run_dir"),
            "weights": trial.user_attrs.get("weights"),
            "metric_name": trial.user_attrs.get("metric_name"),
            "precision": best_metrics.get("precision"),
            "recall": best_metrics.get("recall"),
            "map50": best_metrics.get("map50"),
            "map50_95": best_metrics.get("map50_95"),
            "final_map50_95": final_metrics.get("map50_95"),
            "best_epoch": run_summary.get("best_epoch"),
            "epochs_completed": run_summary.get("epochs_completed"),
            "elapsed_seconds": run_summary.get("elapsed_seconds"),
            "params": dict(trial.params),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["state"] == "COMPLETE",
            row["value"] is not None,
            row["value"] if row["value"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    return rows


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    param_keys = sorted({key for row in rows for key in dict(row.get("params") or {}).keys()})
    fieldnames = [
        "trial_number",
        "state",
        "value",
        "metric_name",
        "report_label",
        "version",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "final_map50_95",
        "best_epoch",
        "epochs_completed",
        "elapsed_seconds",
        "weights",
        "run_dir",
    ] + [f"param:{key}" for key in param_keys]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames if not key.startswith("param:")}
            for key in param_keys:
                csv_row[f"param:{key}"] = dict(row.get("params") or {}).get(key)
            writer.writerow(csv_row)


def render_markdown(study_name: str, rows: list[dict[str, Any]], output_path: Path, chart_name: str | None) -> None:
    lines = [
        f"# Optuna Study Report: {study_name}",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Total trials: {len(rows)}",
        f"- Completed trials: {sum(1 for row in rows if row['state'] == 'COMPLETE')}",
        "",
    ]
    if chart_name:
        lines.extend([f"![leaderboard]({chart_name})", ""])
    lines.extend(
        [
            "| Trial | State | Value | Version | mAP50-95 | Precision | Recall | Label |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {trial_number} | {state} | {value_text} | {version} | {map50_95:.4f} | {precision:.4f} | {recall:.4f} | {report_label} |".format(
                trial_number=row["trial_number"],
                state=row["state"],
                value_text=f"{row['value']:.6f}" if isinstance(row["value"], (int, float)) else "-",
                version=row["version"] or "-",
                map50_95=float(row["map50_95"] or 0.0),
                precision=float(row["precision"] or 0.0),
                recall=float(row["recall"] or 0.0),
                report_label=row["report_label"] or "-",
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_chart(rows: list[dict[str, Any]], output_path: Path, top_k: int) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    completed = [row for row in rows if row["state"] == "COMPLETE" and row["value"] is not None][: max(1, top_k)]
    if not completed:
        return False

    labels = [f"trial_{row['trial_number']:04d}" for row in completed]
    values = [float(row["value"]) for row in completed]
    fig, ax = plt.subplots(figsize=(14, max(6, 0.55 * len(completed) + 2)), constrained_layout=True)
    ax.barh(labels, values, color="#2E86AB")
    ax.invert_yaxis()
    ax.set_title("Optuna Trial Objective Values")
    ax.set_xlabel("Objective")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def build_report_payload(study: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_trial = None
    if rows and any(row["state"] == "COMPLETE" and row["value"] is not None for row in rows):
        best_trial = study.best_trial
    return {
        "study_name": study.study_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trials_total": len(rows),
        "trials_complete": sum(1 for row in rows if row["state"] == "COMPLETE"),
        "best_trial": None
        if best_trial is None
        else {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": dict(best_trial.params),
            "user_attrs": dict(best_trial.user_attrs),
        },
    }


def generate_study_report(study_dir: Path, *, storage: str, top_k: int = 20) -> dict[str, Any]:
    optuna_mod = require_optuna()
    study_name = study_dir.name
    study = optuna_mod.load_study(study_name=study_name, storage=storage)
    rows = build_trial_rows(study, study_dir)

    reports_root = study_dir / "reports"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = reports_root / f"report_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = report_dir / "summary.csv"
    summary_json = report_dir / "summary.json"
    report_md = report_dir / "report.md"
    chart_png = report_dir / "leaderboard.png"

    write_summary_csv(rows, summary_csv)
    write_json(summary_json, {"study_name": study.study_name, "rows": rows})
    chart_created = render_chart(rows, chart_png, top_k=top_k)
    render_markdown(study.study_name, rows, report_md, chart_png.name if chart_created else None)
    write_json(report_dir / "report_meta.json", build_report_payload(study, rows))

    latest_dir = reports_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source in (summary_csv, summary_json, report_md, report_dir / "report_meta.json"):
        (latest_dir / source.name).write_bytes(source.read_bytes())
    if chart_created:
        (latest_dir / chart_png.name).write_bytes(chart_png.read_bytes())

    return {
        "study_name": study.study_name,
        "report_dir": str(report_dir),
        "latest_dir": str(latest_dir),
        "rows": len(rows),
        "chart_created": chart_created,
    }


def main() -> None:
    args = parse_args()
    module_dir = Path(__file__).resolve().parent
    _repo_root = find_repo_root(module_dir)
    study_dir = resolve_study_dir(args, module_dir)
    storage = resolve_storage(study_dir, args.storage)
    result = generate_study_report(study_dir, storage=storage, top_k=args.top_k)
    print(f"Study report generated: {result['report_dir']}")
    print(f"Latest snapshot: {result['latest_dir']}")


if __name__ == "__main__":
    main()
