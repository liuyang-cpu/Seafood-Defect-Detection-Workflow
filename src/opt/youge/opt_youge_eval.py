from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


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
    parser = argparse.ArgumentParser(description="Evaluate prediction labels against GT labels and generate an HTML report.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--species", type=str, default=None, help="Dataset species name, e.g. youge.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to evaluate, e.g. train or val.")
    parser.add_argument("--predict-name", type=str, default=None, help="Prediction run directory name.")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of original images.")
    parser.add_argument("--predict-image-dir", type=str, default=None, help="Directory of rendered prediction images.")
    parser.add_argument("--gt-labels-dir", type=str, default=None, help="Directory of GT label txt files.")
    parser.add_argument("--pred-labels-dir", type=str, default=None, help="Directory of prediction label txt files.")
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name.")
    parser.add_argument("--version", type=str, default=None, help="Model version token, e.g. version001.")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Minimum prediction confidence.")
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU threshold for matching.")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum images to show in HTML.")
    parser.add_argument("--summary-only", action="store_true", help="Only compute and write summary.json without HTML details.")
    parser.add_argument("--summary-output", type=str, default=None, help="Optional path for summary JSON output.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory.")
    return parser.parse_args()


def load_config(config_path: Path | None) -> dict:
    defaults = {
        "species": "youge",
        "split": "val",
        "predict_name": "youge_predict",
        "image_dir": None,
        "predict_image_dir": None,
        "gt_labels_dir": None,
        "pred_labels_dir": None,
        "project": "runs/opt",
        "name": "youge_opt_report",
        "version": None,
        "conf_threshold": None,
        "iou_threshold": 0.5,
        "max_items": 200,
        "exist_ok": True,
    }
    if config_path is None:
        return defaults

    with config_path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    defaults.update(user_config)
    return defaults


def merge_args_into_config(args: argparse.Namespace, config: dict) -> dict:
    for key in (
        "species",
        "split",
        "predict_name",
        "image_dir",
        "predict_image_dir",
        "gt_labels_dir",
        "pred_labels_dir",
        "project",
        "name",
        "version",
        "conf_threshold",
        "iou_threshold",
        "max_items",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if args.exist_ok:
        config["exist_ok"] = True
    return config


def ensure_output_dir(output_dir: Path, exist_ok: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not exist_ok:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def collect_images(image_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem not in images:
            images[path.stem] = path
    return images


def merge_stem_path_maps(named_maps: list[tuple[str, dict[str, Path]]], kind: str) -> dict[str, Path]:
    merged: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for scope_name, path_map in named_maps:
        for stem, path in path_map.items():
            if stem in merged:
                duplicates.setdefault(stem, [str(merged[stem])]).append(str(path))
                continue
            merged[stem] = path
    if duplicates:
        duplicate_lines = ", ".join(f"{stem}: {paths}" for stem, paths in sorted(duplicates.items()))
        raise ValueError(f"Duplicate {kind} stems found across splits, evaluation is ambiguous: {duplicate_lines}")
    return merged


def collect_txts(label_dir: Path) -> dict[str, Path]:
    labels: dict[str, Path] = {}
    if not label_dir.exists():
        return labels
    for path in sorted(label_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".txt" and path.stem.lower() != "classes" and path.stem not in labels:
            labels[path.stem] = path
    return labels


def resolve_split_dirs(dataset_root: Path, split: str) -> tuple[list[Path], list[Path]]:
    if split == "both":
        return (
            [dataset_root / "images" / "train", dataset_root / "images" / "val"],
            [dataset_root / "labels" / "train", dataset_root / "labels" / "val"],
        )
    return ([dataset_root / "images" / split], [dataset_root / "labels" / split])


def resolve_best_split(
    dataset_root: Path, predicted_images: dict[str, Path], preferred_split: str
) -> tuple[str, Path, Path, dict[str, Path]]:
    candidates = [preferred_split] + [split for split in ("train", "val") if split != preferred_split]
    best_split = preferred_split
    best_image_dir = dataset_root / "images" / preferred_split
    best_gt_labels_dir = dataset_root / "labels" / preferred_split
    best_images = collect_images(best_image_dir) if best_image_dir.exists() else {}
    best_match_count = len(best_images.keys() & predicted_images.keys())

    for split in candidates[1:]:
        image_dir = dataset_root / "images" / split
        gt_labels_dir = dataset_root / "labels" / split
        if not image_dir.exists() or not gt_labels_dir.exists():
            continue
        images = collect_images(image_dir)
        match_count = len(images.keys() & predicted_images.keys())
        if match_count > best_match_count:
            best_split = split
            best_image_dir = image_dir
            best_gt_labels_dir = gt_labels_dir
            best_images = images
            best_match_count = match_count

    return best_split, best_image_dir, best_gt_labels_dir, best_images


def load_class_names(dataset_root: Path) -> dict[int, str]:
    classes_path = dataset_root / "classes.txt"
    if not classes_path.exists():
        return {}
    class_names: dict[int, str] = {}
    for idx, line in enumerate(classes_path.read_text(encoding="utf-8").splitlines()):
        name = line.strip()
        if name:
            class_names[idx] = name
    return class_names


def load_yolo_labels(label_path: Path | None, class_names: dict[int, str], conf_threshold: float | None) -> list[dict]:
    if label_path is None or not label_path.exists():
        return []

    detections: list[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        confidence = float(parts[5]) if len(parts) >= 6 else None
        if conf_threshold is not None and confidence is not None and confidence < conf_threshold:
            continue

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "confidence": confidence,
                "xyxy": xywhn_to_xyxy(x_center, y_center, width, height),
            }
        )
    return detections


def xywhn_to_xyxy(x_center: float, y_center: float, width: float, height: float) -> tuple[float, float, float, float]:
    half_w = width / 2.0
    half_h = height / 2.0
    return (
        x_center - half_w,
        y_center - half_h,
        x_center + half_w,
        y_center + half_h,
    )


def bbox_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def match_detections(gt_detections: list[dict], pred_detections: list[dict], iou_threshold: float) -> dict:
    gt_used = [False] * len(gt_detections)
    pred_sorted = sorted(
        enumerate(pred_detections),
        key=lambda item: item[1]["confidence"] if item[1]["confidence"] is not None else -1.0,
        reverse=True,
    )

    tp: list[dict] = []
    fp: list[dict] = []
    fn: list[dict] = []

    for pred_index, pred in pred_sorted:
        best_gt_index = -1
        best_iou = 0.0
        for gt_index, gt in enumerate(gt_detections):
            if gt_used[gt_index] or gt["class_id"] != pred["class_id"]:
                continue
            iou = bbox_iou(gt["xyxy"], pred["xyxy"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        pred_record = dict(pred)
        pred_record["pred_index"] = pred_index
        pred_record["best_iou"] = best_iou
        if best_gt_index >= 0:
            gt_used[best_gt_index] = True
            pred_record["gt_index"] = best_gt_index
            tp.append(pred_record)
        else:
            fp.append(pred_record)

    for gt_index, gt in enumerate(gt_detections):
        if not gt_used[gt_index]:
            gt_record = dict(gt)
            gt_record["gt_index"] = gt_index
            fn.append(gt_record)

    return {"tp": tp, "fp": fp, "fn": fn}


def path_to_href(from_dir: Path, target_path: Path) -> str:
    relative_path = Path(os.path.relpath(target_path, start=from_dir))
    return quote(relative_path.as_posix(), safe="/")


def format_detection_line(prefix: str, det: dict, show_iou: bool = False) -> str:
    label = html.escape(str(det["class_name"]))
    conf = det["confidence"]
    parts = [prefix, label]
    if conf is not None:
        parts.append(f"conf={conf:.4f}")
    if show_iou and "best_iou" in det:
        parts.append(f"IoU={det['best_iou']:.4f}")
    return " | ".join(parts)


def write_item_html(report_dir: Path, item: dict, config: dict) -> str:
    details_dir = report_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    item_path = details_dir / f'{item["stem"]}.html'
    tp_html = "\n".join(f"      <li>{html.escape(line)}</li>" for line in item["tp_lines"]) or "      <li>无</li>"
    fp_html = "\n".join(f"      <li>{html.escape(line)}</li>" for line in item["fp_lines"]) or "      <li>无</li>"
    fn_html = "\n".join(f"      <li>{html.escape(line)}</li>" for line in item["fn_lines"]) or "      <li>无</li>"
    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'  <title>{html.escape(item["stem"])} - Youge Eval Detail</title>',
            "  <style>",
            "    body { margin: 0; padding: 24px; font-family: 'Segoe UI', sans-serif; background: #f3f6fa; color: #0f172a; }",
            "    h1 { margin: 0 0 10px; font-size: 28px; }",
            "    .topline { margin: 0 0 20px; color: #334155; font-size: 14px; }",
            "    .back-link { display: inline-block; margin-bottom: 18px; color: #0369a1; text-decoration: none; font-weight: 600; }",
            "    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }",
            "    .summary-card { background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; padding: 16px; }",
            "    .summary-card .label { color: #475569; font-size: 13px; }",
            "    .summary-card .value { margin-top: 8px; font-size: 28px; font-weight: 700; }",
            "    .images { display: flex; flex-direction: column; gap: 14px; margin-bottom: 24px; }",
            "    .panel { overflow: hidden; border-radius: 12px; border: 1px solid #e2e8f0; background: #fff; }",
            "    .panel-title { padding: 10px 14px; font-size: 15px; font-weight: 700; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }",
            "    .panel img { display: block; width: 100%; height: auto; }",
            "    .sections { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }",
            "    .section { background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; padding: 16px; }",
            "    .section-title { font-size: 15px; font-weight: 700; margin: 0 0 10px; }",
            "    .section-title.tp { color: #15803d; }",
            "    .section-title.fp { color: #b45309; }",
            "    .section-title.fn { color: #b91c1c; }",
            "    .detail-list { margin: 0; padding-left: 20px; color: #1e293b; }",
            "    .detail-list li { margin: 4px 0; }",
            "  </style>",
            "</head>",
            "<body>",
            f'  <a class="back-link" href="{path_to_href(details_dir, report_dir / "index.html")}">返回汇总页</a>',
            f'  <h1>{html.escape(item["stem"])}</h1>',
            f'  <p class="topline">split={html.escape(str(config["split"]))} | predict_name={html.escape(str(config["predict_name"]))} | conf={config["conf_threshold"]} | iou={config["iou_threshold"]}</p>',
            '  <section class="summary">',
            f'    <div class="summary-card"><div class="label">GT</div><div class="value">{item["gt_count"]}</div></div>',
            f'    <div class="summary-card"><div class="label">Pred</div><div class="value">{item["pred_count"]}</div></div>',
            f'    <div class="summary-card"><div class="label">TP</div><div class="value">{item["tp_count"]}</div></div>',
            f'    <div class="summary-card"><div class="label">FP</div><div class="value">{item["fp_count"]}</div></div>',
            f'    <div class="summary-card"><div class="label">FN</div><div class="value">{item["fn_count"]}</div></div>',
            "  </section>",
            '  <section class="images">',
            f'    <div class="panel"><div class="panel-title">Original</div><img src="{path_to_href(details_dir, report_dir / item["image_href"])}" alt="{item["stem"]} original"></div>',
            f'    <div class="panel"><div class="panel-title">Predict</div><img src="{path_to_href(details_dir, report_dir / item["predict_image_href"])}" alt="{item["stem"]} predict"></div>',
            "  </section>",
            '  <section class="sections">',
            f'    <div class="section"><div class="section-title tp">TP</div><ul class="detail-list">{tp_html}</ul></div>',
            f'    <div class="section"><div class="section-title fp">FP</div><ul class="detail-list">{fp_html}</ul></div>',
            f'    <div class="section"><div class="section-title fn">FN</div><ul class="detail-list">{fn_html}</ul></div>',
            "  </section>",
            "</body>",
            "</html>",
        ]
    )
    item_path.write_text(html_text, encoding="utf-8")
    return path_to_href(report_dir, item_path)


def write_html(report_dir: Path, items: list[dict], summary: dict, config: dict) -> None:
    cards: list[str] = []
    for item in items:
        item["detail_href"] = write_item_html(report_dir, item, config)
        tp_html = "\n".join(f"      <li>{html.escape(line)}</li>" for line in item["tp_lines"]) or "      <li>无</li>"
        fp_html = "\n".join(f"      <li>{html.escape(line)}</li>" for line in item["fp_lines"]) or "      <li>无</li>"
        fn_html = "\n".join(f"      <li>{html.escape(line)}</li>" for line in item["fn_lines"]) or "      <li>无</li>"
        cards.append(
            "\n".join(
                [
                    '<section class="card">',
                    '  <div class="images">',
                    f'    <div class="panel"><div class="panel-title">Original</div><img src="{item["image_href"]}" alt="{item["stem"]} original"></div>',
                    f'    <div class="panel"><div class="panel-title">Predict</div><img src="{item["predict_image_href"]}" alt="{item["stem"]} predict"></div>',
                    "  </div>",
                    '  <div class="meta">',
                    f'    <h2>{html.escape(item["stem"])}</h2>',
                    f'    <div class="metric">GT: {item["gt_count"]} | Pred: {item["pred_count"]}</div>',
                    f'    <div class="metric">TP: {item["tp_count"]} | FP: {item["fp_count"]} | FN: {item["fn_count"]}</div>',
                    f'    <div class="metric"><a href="{item["detail_href"]}" target="_blank" rel="noopener noreferrer">打开单图页面</a></div>',
                    '    <div class="section-title tp">TP</div>',
                    f'    <ul class="detail-list">{tp_html}</ul>',
                    '    <div class="section-title fp">FP</div>',
                    f'    <ul class="detail-list">{fp_html}</ul>',
                    '    <div class="section-title fn">FN</div>',
                    f'    <ul class="detail-list">{fn_html}</ul>',
                    "  </div>",
                    "</section>",
                ]
            )
        )

    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            "  <title>Youge Eval Report</title>",
            "  <style>",
            "    body { margin: 0; padding: 24px; font-family: 'Segoe UI', sans-serif; background: #f3f6fa; color: #0f172a; }",
            "    h1 { margin: 0 0 10px; font-size: 28px; }",
            "    .topline { margin: 0 0 20px; color: #334155; font-size: 14px; }",
            "    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }",
            "    .summary-card { background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; padding: 16px; }",
            "    .summary-card .label { color: #475569; font-size: 13px; }",
            "    .summary-card .value { margin-top: 8px; font-size: 28px; font-weight: 700; }",
            "    .container { display: flex; flex-direction: column; gap: 18px; }",
            "    .card { display: flex; gap: 18px; padding: 16px; background: #fff; border: 1px solid #dbe3ec; border-radius: 16px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); }",
            "    .images { flex: 1 1 auto; display: flex; flex-direction: column; gap: 14px; }",
            "    .panel { overflow: hidden; border-radius: 12px; border: 1px solid #e2e8f0; background: #fff; }",
            "    .panel-title { padding: 10px 14px; font-size: 15px; font-weight: 700; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }",
            "    .panel img { display: block; width: 100%; height: auto; }",
            "    .meta { width: 420px; min-width: 320px; display: flex; flex-direction: column; gap: 10px; }",
            "    .meta h2 { margin: 0; font-size: 22px; word-break: break-all; }",
            "    .metric { font-size: 15px; color: #334155; }",
            "    .section-title { font-size: 15px; font-weight: 700; margin-top: 8px; }",
            "    .section-title.tp { color: #15803d; }",
            "    .section-title.fp { color: #b45309; }",
            "    .section-title.fn { color: #b91c1c; }",
            "    .detail-list { margin: 0; padding-left: 20px; color: #1e293b; }",
            "    .detail-list li { margin: 4px 0; }",
            "    @media (max-width: 1200px) { .card { flex-direction: column; } .meta { width: auto; min-width: 0; } }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Youge Prediction vs GT Report</h1>",
            f'  <p class="topline">split={html.escape(str(config["split"]))} | predict_name={html.escape(str(config["predict_name"]))} | conf={config["conf_threshold"]} | iou={config["iou_threshold"]}</p>',
            '  <section class="summary">',
            f'    <div class="summary-card"><div class="label">Images</div><div class="value">{summary["images"]}</div></div>',
            f'    <div class="summary-card"><div class="label">GT Boxes</div><div class="value">{summary["gt_boxes"]}</div></div>',
            f'    <div class="summary-card"><div class="label">Pred Boxes</div><div class="value">{summary["pred_boxes"]}</div></div>',
            f'    <div class="summary-card"><div class="label">TP</div><div class="value">{summary["tp"]}</div></div>',
            f'    <div class="summary-card"><div class="label">FP</div><div class="value">{summary["fp"]}</div></div>',
            f'    <div class="summary-card"><div class="label">FN</div><div class="value">{summary["fn"]}</div></div>',
            f'    <div class="summary-card"><div class="label">Precision</div><div class="value">{summary["precision"]:.4f}</div></div>',
            f'    <div class="summary-card"><div class="label">Recall</div><div class="value">{summary["recall"]:.4f}</div></div>',
            "  </section>",
            '  <main class="container">',
            *cards,
            "  </main>",
            "</body>",
            "</html>",
        ]
    )
    (report_dir / "index.html").write_text(html_text, encoding="utf-8")


def write_summary_json(report_dir: Path, summary: dict, config: dict) -> None:
    payload = {
        "config": {
            "species": config["species"],
            "split": config["split"],
            "predict_name": config["predict_name"],
            "project": config["project"],
            "name": config["name"],
            "conf_threshold": config["conf_threshold"],
            "iou_threshold": config["iou_threshold"],
            "max_items": config["max_items"],
        },
        "summary": summary,
    }
    (report_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_payload(summary_path: Path, summary: dict, config: dict) -> None:
    payload = {
        "config": {
            "species": config["species"],
            "split": config["split"],
            "predict_name": config["predict_name"],
            "project": config["project"],
            "name": config["name"],
            "conf_threshold": config["conf_threshold"],
            "iou_threshold": config["iou_threshold"],
            "max_items": config["max_items"],
        },
        "summary": summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.youge_versioning import (
        build_versioned_name,
        extract_version_from_text,
        get_model_output_dir,
        load_latest_metadata,
        normalize_version,
    )

    config_path = Path(args.config).resolve() if args.config else script_path.with_name("opt_youge_eval.json")

    config = load_config(config_path if config_path.exists() else None)
    config = merge_args_into_config(args, config)
    version = normalize_version(config.get("version"))
    if version is None:
        version = extract_version_from_text(config.get("predict_name")) or extract_version_from_text(config.get("name"))
    if version is None:
        latest_meta = load_latest_metadata(get_model_output_dir(repo_root, str(config["species"])))
        version = latest_meta["version"] if latest_meta else None
    config["version"] = version
    config["predict_name"] = build_versioned_name(str(config["predict_name"]), version)
    config["name"] = build_versioned_name(str(config["name"]), version)

    species = config["species"]
    split = str(config["split"]).lower()
    if split not in {"train", "val", "both"}:
        raise ValueError(f"Unsupported split: {split}. Expected one of: train, val, both")

    dataset_root = repo_root / "datasets" / "data" / species
    default_image_dirs, default_gt_label_dirs = resolve_split_dirs(dataset_root, split)
    default_predict_image_dir = repo_root / "src" / "predict" / species / "runs" / "predict" / config["predict_name"]
    default_pred_labels_dir = default_predict_image_dir / "labels"

    image_dir = Path(config["image_dir"]).resolve() if config["image_dir"] else default_image_dirs[0]
    predict_image_dir = Path(config["predict_image_dir"]).resolve() if config["predict_image_dir"] else default_predict_image_dir
    gt_labels_dir = Path(config["gt_labels_dir"]).resolve() if config["gt_labels_dir"] else default_gt_label_dirs[0]
    pred_labels_dir = Path(config["pred_labels_dir"]).resolve() if config["pred_labels_dir"] else default_pred_labels_dir

    if config["conf_threshold"] is None:
        run_config_path = predict_image_dir / "run_config.json"
        if run_config_path.exists():
            with run_config_path.open("r", encoding="utf-8") as f:
                run_config = json.load(f)
            config["conf_threshold"] = run_config.get("conf")
        else:
            predict_config_path = repo_root / "src" / "predict" / species / f"predict_{species}.json"
            if predict_config_path.exists():
                with predict_config_path.open("r", encoding="utf-8") as f:
                    predict_config = json.load(f)
                config["conf_threshold"] = predict_config.get("conf")

    if not predict_image_dir.exists():
        raise FileNotFoundError(f"Prediction image directory not found: {predict_image_dir}")
    if not pred_labels_dir.exists():
        raise FileNotFoundError(f"Prediction label directory not found: {pred_labels_dir}")

    config["project"] = resolve_output_dir(script_path.parent, config["project"])
    output_dir = Path(config["project"]) / config["name"]
    if not args.summary_only:
        ensure_output_dir(output_dir, config["exist_ok"])

    class_names = load_class_names(dataset_root)
    predicted_images = collect_images(predict_image_dir)
    if config["image_dir"] is not None:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        original_images = collect_images(image_dir)
    else:
        missing_dirs = [str(path) for path in default_image_dirs if not path.exists()]
        if missing_dirs:
            raise FileNotFoundError(f"Image directory not found: {', '.join(missing_dirs)}")
        original_images = merge_stem_path_maps(
            [(path.name, collect_images(path)) for path in default_image_dirs],
            kind="image",
        )

    if config["gt_labels_dir"] is not None:
        if not gt_labels_dir.exists():
            raise FileNotFoundError(f"GT label directory not found: {gt_labels_dir}")
        gt_label_map = collect_txts(gt_labels_dir)
    else:
        missing_dirs = [str(path) for path in default_gt_label_dirs if not path.exists()]
        if missing_dirs:
            raise FileNotFoundError(f"GT label directory not found: {', '.join(missing_dirs)}")
        gt_label_map = merge_stem_path_maps(
            [(path.name, collect_txts(path)) for path in default_gt_label_dirs],
            kind="label",
        )

    if split != "both" and not (original_images.keys() & predicted_images.keys()) and config["image_dir"] is None and config["gt_labels_dir"] is None:
        split, image_dir, gt_labels_dir, original_images = resolve_best_split(dataset_root, predicted_images, split)
        config["split"] = split
        gt_label_map = collect_txts(gt_labels_dir)
    pred_label_map = collect_txts(pred_labels_dir)

    common_names = sorted(original_images.keys() & predicted_images.keys())
    if not common_names:
        raise FileNotFoundError("No matching original and predicted images were found.")

    max_items = int(config["max_items"]) if config["max_items"] else len(common_names)
    selected_names = common_names[:max_items]

    summary = {"images": 0, "gt_boxes": 0, "pred_boxes": 0, "tp": 0, "fp": 0, "fn": 0}
    items: list[dict] = []
    for stem in selected_names:
        gt_detections = load_yolo_labels(gt_label_map.get(stem), class_names, conf_threshold=None)
        pred_detections = load_yolo_labels(pred_label_map.get(stem), class_names, config["conf_threshold"])
        matched = match_detections(gt_detections, pred_detections, float(config["iou_threshold"]))

        summary["images"] += 1
        summary["gt_boxes"] += len(gt_detections)
        summary["pred_boxes"] += len(pred_detections)
        summary["tp"] += len(matched["tp"])
        summary["fp"] += len(matched["fp"])
        summary["fn"] += len(matched["fn"])

        if not args.summary_only:
            tp_lines = [format_detection_line(f"{idx}.", det, show_iou=True) for idx, det in enumerate(matched["tp"], start=1)]
            fp_lines = [format_detection_line(f"{idx}.", det, show_iou=False) for idx, det in enumerate(matched["fp"], start=1)]
            fn_lines = [format_detection_line(f"{idx}.", det, show_iou=False) for idx, det in enumerate(matched["fn"], start=1)]
            items.append(
                {
                    "stem": stem,
                    "image_href": path_to_href(output_dir, original_images[stem]),
                    "predict_image_href": path_to_href(output_dir, predicted_images[stem]),
                    "gt_count": len(gt_detections),
                    "pred_count": len(pred_detections),
                    "tp_count": len(matched["tp"]),
                    "fp_count": len(matched["fp"]),
                    "fn_count": len(matched["fn"]),
                    "tp_lines": tp_lines,
                    "fp_lines": fp_lines,
                    "fn_lines": fn_lines,
                }
            )

    summary["precision"] = summary["tp"] / (summary["tp"] + summary["fp"]) if (summary["tp"] + summary["fp"]) else 0.0
    summary["recall"] = summary["tp"] / (summary["tp"] + summary["fn"]) if (summary["tp"] + summary["fn"]) else 0.0
    summary["f1"] = (
        2 * summary["precision"] * summary["recall"] / (summary["precision"] + summary["recall"])
        if (summary["precision"] + summary["recall"])
        else 0.0
    )
    summary["avg_pred_boxes_per_image"] = summary["pred_boxes"] / summary["images"] if summary["images"] else 0.0

    if args.summary_only:
        summary_output = Path(args.summary_output).resolve() if args.summary_output else (output_dir / "summary.json")
        write_summary_payload(summary_output, summary, config)
        print(f"Evaluation summary generated at: {summary_output}")
    else:
        write_html(output_dir, items, summary, config)
        write_summary_json(output_dir, summary, config)
        if args.summary_output:
            write_summary_payload(Path(args.summary_output).resolve(), summary, config)
        print(f"Evaluation report generated at: {output_dir}")


if __name__ == "__main__":
    main()
