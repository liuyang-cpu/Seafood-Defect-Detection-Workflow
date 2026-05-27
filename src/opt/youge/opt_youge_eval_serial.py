from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


def find_latest_versioned_dir_name(runs_dir: Path, base_name: str) -> str | None:
    if not runs_dir.exists():
        return None

    clean_base = base_name
    lower_name = clean_base.lower()
    version_pos = lower_name.rfind("version")
    if version_pos != -1:
        prefix = clean_base[:version_pos].rstrip("_-")
        if prefix:
            clean_base = prefix

    latest_name = None
    latest_number = -1
    prefix = f"{clean_base}_version"
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        lower_stem = path.name.lower()
        if not lower_stem.startswith(prefix.lower()):
            continue
        suffix = path.name[len(prefix) :]
        if not suffix.isdigit():
            continue
        number = int(suffix)
        if number > latest_number:
            latest_number = number
            latest_name = path.name

    return latest_name


def expand_threshold_token(token: str) -> list[float]:
    text = token.strip()
    if not text:
        return []
    if "-" not in text:
        try:
            return [float(text)]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid float value: '{text}'") from exc

    range_text, _, step_text = text.partition(":")
    start_text, end_text = [part.strip() for part in range_text.split("-", maxsplit=1)]
    if not start_text or not end_text:
        raise argparse.ArgumentTypeError(f"invalid range value: '{text}'")
    try:
        start = Decimal(start_text)
        end = Decimal(end_text)
        if step_text.strip():
            step = Decimal(step_text.strip())
        else:
            decimal_places = max(
                len(start_text.split(".")[1]) if "." in start_text else 0,
                len(end_text.split(".")[1]) if "." in end_text else 0,
                2,
            )
            step = Decimal("1").scaleb(-decimal_places)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid range value: '{text}'") from exc

    if step <= 0:
        raise argparse.ArgumentTypeError(f"range step must be positive: '{text}'")
    if end < start:
        raise argparse.ArgumentTypeError(f"range end must be >= start: '{text}'")

    values: list[float] = []
    current = start
    epsilon = step / Decimal("1000")
    while current <= end + epsilon:
        values.append(float(current))
        current += step
    return values


def parse_conf_thresholds(values: list[str]) -> list[float]:
    thresholds: list[float] = []
    for value in values:
        for part in value.split(","):
            thresholds.extend(expand_threshold_token(part))
    if not thresholds:
        raise argparse.ArgumentTypeError("at least one conf threshold is required")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run opt_youge_eval.py serially with different conf thresholds.")
    parser.add_argument(
        "--conf-thresholds",
        nargs="+",
        required=True,
        help="One or more confidence thresholds, supports space/comma values and ranges like 0.70-0.85 or 0.70-0.85:0.02.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to the JSON config file.")
    parser.add_argument("--species", type=str, default=None, help="Dataset species name, e.g. youge.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to evaluate, e.g. train or val.")
    parser.add_argument("--predict-name", type=str, default=None, help="Prediction run directory name.")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of original images.")
    parser.add_argument("--predict-image-dir", type=str, default=None, help="Directory of rendered prediction images.")
    parser.add_argument("--gt-labels-dir", type=str, default=None, help="Directory of GT label txt files.")
    parser.add_argument("--pred-labels-dir", type=str, default=None, help="Directory of prediction label txt files.")
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name-prefix", type=str, default=None, help="Prefix for each run name.")
    parser.add_argument("--batch-name", type=str, default=None, help="Directory name for the aggregated conf report.")
    parser.add_argument("--version", type=str, default=None, help="Model version token, e.g. version001.")
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU threshold for matching.")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum images to show in HTML.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory.")
    args = parser.parse_args()
    args.conf_thresholds = parse_conf_thresholds(args.conf_thresholds)
    return args


def build_base_command(args: argparse.Namespace, eval_script: Path) -> list[str]:
    command = [sys.executable, str(eval_script)]
    option_map = {
        "--config": args.config,
        "--species": args.species,
        "--split": args.split,
        "--predict-name": args.predict_name,
        "--image-dir": args.image_dir,
        "--predict-image-dir": args.predict_image_dir,
        "--gt-labels-dir": args.gt_labels_dir,
        "--pred-labels-dir": args.pred_labels_dir,
        "--project": args.project,
        "--version": args.version,
        "--iou-threshold": args.iou_threshold,
        "--max-items": args.max_items,
    }
    for key, value in option_map.items():
        if value is not None:
            command.extend([key, str(value)])

    if args.exist_ok:
        command.append("--exist-ok")
    return command


def resolve_project_dir(args: argparse.Namespace, script_dir: Path) -> Path:
    project = Path(args.project) if args.project else Path("runs/opt")
    if not project.is_absolute():
        project = script_dir / project
    return project.resolve()


def format_threshold_text(threshold: float) -> str:
    return f"{threshold:.6f}".rstrip("0").rstrip(".")


def format_threshold_suffix(threshold: float) -> str:
    return format_threshold_text(threshold).replace("-", "neg_").replace(".", "_")


def load_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_token(value: object) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".").replace("-", "neg_").replace(".", "_")
    return str(value).replace("-", "_").replace(".", "_").replace(" ", "_")


def load_predict_run_config(predict_run_dir: Path) -> dict:
    run_config_path = predict_run_dir / "run_config.json"
    if not run_config_path.exists():
        return {}
    with run_config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_export_identifier(version: str | None, predict_params: dict) -> str:
    parts = [version or "unversioned"]
    for key in ("conf", "iou", "edge_penalty", "edge_touch_px", "flat_ratio_threshold", "edge_penalty_factor"):
        if key in predict_params:
            parts.append(f"{key}_{format_token(predict_params[key])}")
    return "opt_summary_" + "_".join(parts)


def export_batch_to_model_output(batch_dir: Path, model_output_dir: Path, export_identifier: str) -> Path:
    target_dir = model_output_dir / export_identifier
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "summary.json", "summary.csv"):
        source = batch_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)
    return target_dir


def build_svg_line_chart(
    chart_id: str,
    title: str,
    subtitle: str,
    rows: list[dict],
    series: list[dict],
    y_min: float,
    y_max: float,
    value_formatter: str,
) -> str:
    plot_width = 760
    plot_height = 260
    left = 56
    right = 16
    top = 24
    bottom = 44
    width = plot_width + left + right
    height = plot_height + top + bottom
    confs = [float(row["conf_threshold"]) for row in rows]
    min_conf = min(confs)
    max_conf = max(confs)
    if max_conf == min_conf:
        max_conf = min_conf + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0
    y_padding = (y_max - y_min) * 0.08
    y_min -= y_padding
    y_max += y_padding

    def x_pos(conf: float) -> float:
        return left + (conf - min_conf) / (max_conf - min_conf) * plot_width

    def y_pos(value: float) -> float:
        return top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_height

    grid_lines: list[str] = []
    for step in range(6):
        ratio = step / 5
        y = top + ratio * plot_height
        tick_value = y_max - ratio * (y_max - y_min)
        grid_lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="grid-line"></line>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis-label">{tick_value:{value_formatter}}</text>',
            ]
        )

    x_ticks: list[str] = []
    for row in rows:
        conf = float(row["conf_threshold"])
        x = x_pos(conf)
        x_ticks.extend(
            [
                f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" class="tick-line"></line>',
                f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" class="axis-label">{html.escape(str(row["conf_threshold"]))}</text>',
            ]
        )

    series_html: list[str] = []
    legend_items: list[str] = []
    for metric in series:
        points = []
        point_marks = []
        values = [float(row[metric["key"]]) for row in rows]
        min_value = min(values)
        max_value = max(values)
        for row in rows:
            value = float(row[metric["key"]])
            conf = float(row["conf_threshold"])
            x = x_pos(conf)
            y = y_pos(value)
            points.append(f"{x:.2f},{y:.2f}")
            mark_lines = [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{metric["color"]}"></circle>']
            if value == max_value:
                mark_lines.extend(
                    [
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" class="peak-ring peak-max"></circle>',
                        (
                            f'<text x="{x:.2f}" y="{y - 14:.2f}" text-anchor="middle" class="point-label point-max" '
                            f'fill="{metric["color"]}">max {value:{value_formatter}} @ {conf:g}</text>'
                        ),
                    ]
                )
            if value == min_value:
                mark_lines.extend(
                    [
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" class="peak-ring peak-min"></circle>',
                        (
                            f'<text x="{x:.2f}" y="{y + 18:.2f}" text-anchor="middle" class="point-label point-min" '
                            f'fill="{metric["color"]}">min {value:{value_formatter}} @ {conf:g}</text>'
                        ),
                    ]
                )
            point_marks.append("\n".join(mark_lines))
        series_html.append(
            "\n".join(
                [
                    f'<polyline points="{" ".join(points)}" fill="none" stroke="{metric["color"]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>',
                    *point_marks,
                ]
            )
        )
        legend_items.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{metric["color"]};"></span>{html.escape(metric["label"])}</div>'
        )

    return "\n".join(
        [
            '<section class="chart-card">',
            f'  <div class="chart-header"><div><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div><div class="legend">{"".join(legend_items)}</div></div>',
            f'  <svg id="{html.escape(chart_id)}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            *grid_lines,
            f'    <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis-line"></line>',
            f'    <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis-line"></line>',
            *x_ticks,
            *series_html,
            "  </svg>",
            "</section>",
        ]
    )


def write_batch_csv(batch_dir: Path, rows: list[dict]) -> None:
    fieldnames = [
        "conf_threshold",
        "images",
        "gt_boxes",
        "pred_boxes",
        "avg_pred_boxes_per_image",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "delta_pred_boxes",
        "delta_precision",
        "delta_recall",
        "delta_f1",
    ]
    with (batch_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_batch_json(batch_dir: Path, payload: dict) -> None:
    (batch_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_batch_html(batch_dir: Path, rows: list[dict], meta: dict) -> None:
    metric_chart = build_svg_line_chart(
        chart_id="metric-chart",
        title="指标折线图",
        subtitle="直接比较不同 conf 下 Precision / Recall / F1 的变化",
        rows=rows,
        series=[
            {"key": "precision", "label": "Precision", "color": "#0f766e"},
            {"key": "recall", "label": "Recall", "color": "#2563eb"},
            {"key": "f1", "label": "F1", "color": "#dc2626"},
        ],
        y_min=min(min(row["precision"], row["recall"], row["f1"]) for row in rows),
        y_max=max(max(row["precision"], row["recall"], row["f1"]) for row in rows),
        value_formatter=".4f",
    )
    count_chart = build_svg_line_chart(
        chart_id="count-chart",
        title="数量折线图",
        subtitle="观察预测框数量以及 TP / FP / FN 随 conf 的变化",
        rows=rows,
        series=[
            {"key": "pred_boxes", "label": "Pred", "color": "#7c3aed"},
            {"key": "tp", "label": "TP", "color": "#15803d"},
            {"key": "fp", "label": "FP", "color": "#ea580c"},
            {"key": "fn", "label": "FN", "color": "#b91c1c"},
        ],
        y_min=min(min(row["pred_boxes"], row["tp"], row["fp"], row["fn"]) for row in rows),
        y_max=max(max(row["pred_boxes"], row["tp"], row["fp"], row["fn"]) for row in rows),
        value_formatter=".0f",
    )
    table_rows: list[str] = []
    for row in rows:
        table_rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"  <td>{html.escape(str(row['conf_threshold']))}</td>",
                    f"  <td>{row['pred_boxes']}</td>",
                    f"  <td>{row['avg_pred_boxes_per_image']:.4f}</td>",
                    f"  <td>{row['tp']}</td>",
                    f"  <td>{row['fp']}</td>",
                    f"  <td>{row['fn']}</td>",
                    f"  <td>{row['precision']:.4f}</td>",
                    f"  <td>{row['recall']:.4f}</td>",
                    f"  <td>{row['f1']:.4f}</td>",
                    f"  <td>{row['delta_pred_boxes']:+.0f}</td>",
                    f"  <td>{row['delta_precision']:+.4f}</td>",
                    f"  <td>{row['delta_recall']:+.4f}</td>",
                    f"  <td>{row['delta_f1']:+.4f}</td>",
                    "</tr>",
                ]
            )
        )

    best_f1 = max(rows, key=lambda item: item["f1"])
    best_precision = max(rows, key=lambda item: item["precision"])
    best_recall = max(rows, key=lambda item: item["recall"])
    predict_params = meta.get("predict_params", {})
    predict_param_text = " | ".join(
        f"{key}={predict_params[key]}"
        for key in ("conf", "iou", "edge_penalty", "edge_touch_px", "flat_ratio_threshold", "edge_penalty_factor")
        if key in predict_params
    )
    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            "  <title>Youge Conf Summary</title>",
            "  <style>",
            "    body { margin: 0; padding: 24px; font-family: 'Segoe UI', sans-serif; background: #f3f6fa; color: #0f172a; }",
            "    h1 { margin: 0 0 10px; font-size: 28px; }",
            "    .topline { margin: 0 0 18px; color: #334155; font-size: 14px; }",
            "    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 20px; }",
            "    .card { background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; padding: 16px; }",
            "    .label { color: #475569; font-size: 13px; }",
            "    .value { margin-top: 8px; font-size: 26px; font-weight: 700; }",
            "    .sub { margin-top: 6px; color: #475569; font-size: 13px; }",
            "    .chart-grid { display: grid; grid-template-columns: 1fr; gap: 18px; margin-bottom: 20px; }",
            "    .chart-card { background: #fff; border: 1px solid #dbe3ec; border-radius: 16px; padding: 16px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); }",
            "    .chart-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 14px; }",
            "    .chart-header h2 { margin: 0 0 6px; font-size: 20px; }",
            "    .chart-header p { margin: 0; color: #475569; font-size: 13px; }",
            "    .legend { display: flex; flex-wrap: wrap; gap: 10px 14px; justify-content: flex-end; }",
            "    .legend-item { display: inline-flex; align-items: center; gap: 8px; color: #334155; font-size: 13px; }",
            "    .legend-swatch { width: 12px; height: 12px; border-radius: 999px; display: inline-block; }",
            "    svg { width: 100%; height: auto; display: block; }",
            "    .grid-line { stroke: #e2e8f0; stroke-width: 1; }",
            "    .axis-line, .tick-line { stroke: #94a3b8; stroke-width: 1.2; }",
            "    .axis-label { fill: #64748b; font-size: 12px; }",
            "    .point-label { font-size: 11px; font-weight: 600; }",
            "    .peak-ring { fill: none; stroke-width: 2; }",
            "    .peak-max { stroke: #0f172a; }",
            "    .peak-min { stroke: #94a3b8; stroke-dasharray: 3 2; }",
            "    .point-max, .point-min { paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; }",
            "    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; overflow: hidden; }",
            "    th, td { padding: 12px 10px; border-bottom: 1px solid #e2e8f0; text-align: center; font-size: 14px; }",
            "    th { background: #eaf2fb; font-weight: 700; }",
            "    tr:nth-child(even) td { background: #f8fbff; }",
            "    a { color: #0369a1; text-decoration: none; }",
            "    @media (max-width: 1200px) { body { padding: 14px; } .chart-header { flex-direction: column; } .legend { justify-content: flex-start; } table { display: block; overflow-x: auto; white-space: nowrap; } }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Youge Confidence Sweep Report</h1>",
            f'  <p class="topline">species={html.escape(str(meta["species"]))} | split={html.escape(str(meta["split"]))} | predict_name={html.escape(str(meta["predict_name"]))} | iou={meta["iou_threshold"]} | conf_count={len(rows)}</p>',
            (
                f'  <p class="topline">model_version={html.escape(str(meta.get("model_version", "")))}'
                + (f" | {html.escape(predict_param_text)}" if predict_param_text else "")
                + "</p>"
            ),
            '  <section class="cards">',
            f'    <div class="card"><div class="label">Best F1</div><div class="value">{best_f1["f1"]:.4f}</div><div class="sub">conf={best_f1["conf_threshold"]}</div></div>',
            f'    <div class="card"><div class="label">Best Precision</div><div class="value">{best_precision["precision"]:.4f}</div><div class="sub">conf={best_precision["conf_threshold"]}</div></div>',
            f'    <div class="card"><div class="label">Best Recall</div><div class="value">{best_recall["recall"]:.4f}</div><div class="sub">conf={best_recall["conf_threshold"]}</div></div>',
            f'    <div class="card"><div class="label">Images</div><div class="value">{rows[0]["images"]}</div><div class="sub">GT boxes={rows[0]["gt_boxes"]}</div></div>',
            "  </section>",
            '  <section class="chart-grid">',
            metric_chart,
            count_chart,
            "  </section>",
            "  <table>",
            "    <thead>",
            "      <tr>",
            "        <th>conf</th>",
            "        <th>Pred</th>",
            "        <th>Pred/Image</th>",
            "        <th>TP</th>",
            "        <th>FP</th>",
            "        <th>FN</th>",
            "        <th>Precision</th>",
            "        <th>Recall</th>",
            "        <th>F1</th>",
            "        <th>dPred</th>",
            "        <th>dPrec</th>",
            "        <th>dRecall</th>",
            "        <th>dF1</th>",
            "      </tr>",
            "    </thead>",
            "    <tbody>",
            *table_rows,
            "    </tbody>",
            "  </table>",
            "</body>",
            "</html>",
        ]
    )
    (batch_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.youge_versioning import (
        build_versioned_name,
        extract_version_from_text,
        get_model_output_dir,
        normalize_version,
        strip_trailing_version,
    )

    eval_script = script_dir / "opt_youge_eval.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")

    project_dir = resolve_project_dir(args, script_dir)
    version = normalize_version(args.version)
    if version is None:
        version = (
            extract_version_from_text(args.name_prefix)
            or extract_version_from_text(args.batch_name)
            or extract_version_from_text(args.predict_name)
        )
    if version is None:
        predict_runs_dir = repo_root / "src" / "predict" / (args.species or "youge") / "runs" / "predict"
        latest_predict_name = find_latest_versioned_dir_name(predict_runs_dir, args.predict_name or "youge_predict")
        version = extract_version_from_text(latest_predict_name)
    predict_name = (
        build_versioned_name("youge_predict", version)
        if not args.predict_name
        else build_versioned_name(args.predict_name, version)
    )
    predict_run_dir = repo_root / "src" / "predict" / (args.species or "youge") / "runs" / "predict" / predict_name
    predict_run_config = load_predict_run_config(predict_run_dir)
    predict_params = {
        "conf": predict_run_config.get("conf"),
        "iou": predict_run_config.get("iou"),
        "edge_penalty": predict_run_config.get("edge_penalty"),
        "edge_touch_px": predict_run_config.get("edge_touch_px"),
        "flat_ratio_threshold": predict_run_config.get("flat_ratio_threshold"),
        "edge_penalty_factor": predict_run_config.get("edge_penalty_factor"),
    }
    predict_params = {key: value for key, value in predict_params.items() if value is not None}
    version_root_name = build_versioned_name(strip_trailing_version(args.name_prefix or "youge_opt_report"), version)
    version_root_dir = project_dir / version_root_name
    if version_root_dir.exists():
        shutil.rmtree(version_root_dir)
    version_root_dir.mkdir(parents=True, exist_ok=True)
    base_command = build_base_command(args, eval_script)
    batch_name = strip_trailing_version(args.batch_name or "conf_summary")
    batch_dir = version_root_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary_cache_dir = batch_dir / "_summaries"
    summary_cache_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for index, conf_threshold in enumerate(args.conf_thresholds, start=1):
        run_name = f"conf_{format_threshold_suffix(conf_threshold)}"
        summary_path = summary_cache_dir / f"{run_name}.json"
        command = [
            *base_command,
            "--project",
            str(version_root_dir),
            "--conf-threshold",
            str(conf_threshold),
            "--name",
            run_name,
            "--summary-output",
            str(summary_path),
        ]
        print(f"[{index}/{len(args.conf_thresholds)}] Running: {' '.join(command)}")
        subprocess.run(command, check=True, cwd=script_dir)

        summary_payload = load_summary(summary_path)
        summary = summary_payload["summary"]
        runs.append(
            {
                "conf_threshold": summary_payload["config"]["conf_threshold"],
                "images": summary["images"],
                "gt_boxes": summary["gt_boxes"],
                "pred_boxes": summary["pred_boxes"],
                "avg_pred_boxes_per_image": summary["avg_pred_boxes_per_image"],
                "tp": summary["tp"],
                "fp": summary["fp"],
                "fn": summary["fn"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
            }
        )

    runs.sort(key=lambda item: float(item["conf_threshold"]))
    previous: dict | None = None
    for row in runs:
        row["delta_pred_boxes"] = row["pred_boxes"] - previous["pred_boxes"] if previous else 0
        row["delta_precision"] = row["precision"] - previous["precision"] if previous else 0.0
        row["delta_recall"] = row["recall"] - previous["recall"] if previous else 0.0
        row["delta_f1"] = row["f1"] - previous["f1"] if previous else 0.0
        previous = row

    meta = {
        "species": args.species or "youge",
        "split": args.split or "val",
        "predict_name": predict_name,
        "version": version or "",
        "model_version": version or "",
        "predict_params": predict_params,
        "iou_threshold": args.iou_threshold if args.iou_threshold is not None else 0.5,
    }
    write_batch_csv(batch_dir, runs)
    write_batch_json(batch_dir, {"meta": meta, "runs": runs})
    write_batch_html(batch_dir, runs, meta)
    export_identifier = build_export_identifier(version, predict_params)
    exported_dir = export_batch_to_model_output(
        batch_dir, get_model_output_dir(repo_root, args.species or "youge"), export_identifier
    )
    print(f"Batch summary report generated at: {batch_dir}")
    print(f"Exported model-output summary at: {exported_dir}")


if __name__ == "__main__":
    main()
