from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFECT_CLASS_NAMES = {"broken", "muddy", "empty"}
NORMAL_CLASS_NAME = "normal"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction labels against GT labels and report per-class metrics."
    )
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
    parser.add_argument(
        "--postprocess-version", type=str, default=None, help="Postprocess recipe version token, e.g. pp001."
    )
    parser.add_argument("--conf-threshold", type=float, default=None, help="Minimum prediction confidence.")
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU threshold for matching.")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum images to show in HTML.")
    parser.add_argument(
        "--summary-only", action="store_true", help="Only compute and write summary JSON without HTML details."
    )
    parser.add_argument("--summary-output", type=str, default=None, help="Optional path for summary JSON output.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory.")
    return parser.parse_args()


def load_config(config_path: Path | None) -> dict:
    defaults = {
        "species": "youge",
        "split": "both",
        "predict_name": "youge_predict",
        "image_dir": None,
        "predict_image_dir": None,
        "gt_labels_dir": None,
        "pred_labels_dir": None,
        "project": "runs/opt-class",
        "name": "youge_opt_class_report",
        "version": None,
        "postprocess_version": None,
        "conf_threshold": None,
        "iou_threshold": 0.5,
        "max_items": 200,
        "exist_ok": True,
    }
    if config_path is None:
        return defaults

    with config_path.open("r", encoding="utf-8-sig") as f:
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
        "postprocess_version",
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
        if (
            path.is_file()
            and path.suffix.lower() == ".txt"
            and path.stem.lower() != "classes"
            and path.stem not in labels
        ):
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


def build_class_metric_map(class_names: dict[int, str]) -> dict[str, dict[str, float | int]]:
    names = list(class_names.values())
    return {
        name: {"gt_boxes": 0, "pred_boxes": 0, "tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for name in names
    }


def finalize_class_metrics(class_metrics: dict[str, dict[str, float | int]]) -> None:
    for metrics in class_metrics.values():
        tp = int(metrics["tp"])
        fp = int(metrics["fp"])
        fn = int(metrics["fn"])
        metrics["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
        metrics["f1"] = (
            2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
            if (metrics["precision"] + metrics["recall"])
            else 0.0
        )


def match_detections(
    gt_detections: list[dict], pred_detections: list[dict], iou_threshold: float, require_same_class: bool = True
) -> dict:
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
            if gt_used[gt_index]:
                continue
            if require_same_class and gt["class_id"] != pred["class_id"]:
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


def filter_detections_by_class_names(detections: list[dict], class_names: set[str]) -> list[dict]:
    return [det for det in detections if str(det["class_name"]) in class_names]


def build_business_metrics() -> dict[str, float | int]:
    return {
        "count_diff": 0,
        "abs_count_error": 0,
        "defect_gt_boxes": 0,
        "defect_pred_boxes": 0,
        "defect_detected_boxes": 0,
        "defect_missed_boxes": 0,
        "defect_detect_rate": 0.0,
        "defect_miss_rate": 0.0,
        "normal_gt_boxes": 0,
        "normal_hit_by_defect_boxes": 0,
        "normal_to_defect_rate": 0.0,
    }


def finalize_business_metrics(metrics: dict[str, float | int]) -> None:
    count_diff = int(metrics["count_diff"])
    defect_gt = int(metrics["defect_gt_boxes"])
    defect_detected = int(metrics["defect_detected_boxes"])
    defect_missed = int(metrics["defect_missed_boxes"])
    normal_gt = int(metrics["normal_gt_boxes"])
    normal_hit = int(metrics["normal_hit_by_defect_boxes"])
    metrics["abs_count_error"] = abs(count_diff)
    metrics["defect_detect_rate"] = defect_detected / defect_gt if defect_gt else 0.0
    metrics["defect_miss_rate"] = defect_missed / defect_gt if defect_gt else 0.0
    metrics["normal_to_defect_rate"] = normal_hit / normal_gt if normal_gt else 0.0


def build_count_diagnostics(items: list[dict], limit: int = 12) -> dict[str, list[dict]]:
    ranked = sorted(
        items,
        key=lambda item: (
            -abs(int(item["count_diff"])),
            -abs(int(item["fp_count"]) - int(item["fn_count"])),
            item["stem"],
        ),
    )
    overcounted = [item for item in ranked if int(item["count_diff"]) > 0][:limit]
    undercounted = [item for item in ranked if int(item["count_diff"]) < 0][:limit]
    largest_abs_errors = ranked[:limit]
    return {
        "largest_abs_errors": largest_abs_errors,
        "overcounted": overcounted,
        "undercounted": undercounted,
    }


def serialize_detections(detections: list[dict]) -> list[dict]:
    return [
        {
            "class_id": int(det["class_id"]),
            "class_name": str(det["class_name"]),
            "confidence": det.get("confidence"),
            "xyxy": [float(value) for value in det["xyxy"]],
        }
        for det in detections
    ]


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
    item_path = details_dir / f"{item['stem']}.html"
    tp_html = "\n".join(f"      <li>{html.escape(line)}</li>" for line in item["tp_lines"]) or "      <li>无</li>"
    fp_html = "\n".join(f"      <li>{html.escape(line)}</li>" for line in item["fp_lines"]) or "      <li>无</li>"
    fn_html = "\n".join(f"      <li>{html.escape(line)}</li>" for line in item["fn_lines"]) or "      <li>无</li>"
    class_rows = []
    for class_name, metrics in item["per_class"].items():
        class_rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"  <td>{html.escape(class_name)}</td>",
                    f"  <td>{metrics['gt_boxes']}</td>",
                    f"  <td>{metrics['pred_boxes']}</td>",
                    f"  <td>{metrics['tp']}</td>",
                    f"  <td>{metrics['fp']}</td>",
                    f"  <td>{metrics['fn']}</td>",
                    f"  <td>{metrics['precision']:.4f}</td>",
                    f"  <td>{metrics['recall']:.4f}</td>",
                    f"  <td>{metrics['f1']:.4f}</td>",
                    "</tr>",
                ]
            )
        )
    business = item["business"]
    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f"  <title>{html.escape(item['stem'])} - Youge Class Eval Detail</title>",
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
            "    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; overflow: hidden; margin-bottom: 24px; }",
            "    th, td { padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: center; font-size: 14px; }",
            "    th { background: #eaf2fb; font-weight: 700; }",
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
            f"  <h1>{html.escape(item['stem'])}</h1>",
            f'  <p class="topline">split={html.escape(str(config["split"]))} | predict_name={html.escape(str(config["predict_name"]))} | conf={config["conf_threshold"]} | iou={config["iou_threshold"]}</p>',
            '  <p class="topline">说明: 右侧 Predict Render 复用后处理可视化图，图上的 intact / defective 文本仅用于展示，不参与 GT / Pred 匹配与指标计算。实际评估只读取 labels 目录中的预测框。</p>',
            '  <section class="summary">',
            f'    <div class="summary-card"><div class="label">GT</div><div class="value">{item["gt_count"]}</div></div>',
            f'    <div class="summary-card"><div class="label">Pred</div><div class="value">{item["pred_count"]}</div></div>',
            f'    <div class="summary-card"><div class="label">TP</div><div class="value">{item["tp_count"]}</div></div>',
            f'    <div class="summary-card"><div class="label">FP</div><div class="value">{item["fp_count"]}</div></div>',
            f'    <div class="summary-card"><div class="label">FN</div><div class="value">{item["fn_count"]}</div></div>',
            f'    <div class="summary-card"><div class="label">计数误差</div><div class="value">{business["abs_count_error"]}</div></div>',
            f'    <div class="summary-card"><div class="label">缺陷检出率</div><div class="value">{business["defect_detect_rate"]:.4f}</div></div>',
            f'    <div class="summary-card"><div class="label">缺陷漏检率</div><div class="value">{business["defect_miss_rate"]:.4f}</div></div>',
            f'    <div class="summary-card"><div class="label">normal误检缺陷率</div><div class="value">{business["normal_to_defect_rate"]:.4f}</div></div>',
            "  </section>",
            '  <section class="images">',
            f'    <div class="panel"><div class="panel-title">Original</div><img src="{path_to_href(details_dir, report_dir / item["image_href"])}" alt="{item["stem"]} original"></div>',
            f'    <div class="panel"><div class="panel-title">Predict Render (visual only)</div><img src="{path_to_href(details_dir, report_dir / item["predict_image_href"])}" alt="{item["stem"]} predict"></div>',
            "  </section>",
            "  <table>",
            "    <thead><tr><th>业务指标</th><th>值</th><th>说明</th></tr></thead>",
            "    <tbody>",
            f"      <tr><td>计数差值</td><td>{business['count_diff']}</td><td>pred_total_count - gt_total_count</td></tr>",
            f"      <tr><td>绝对计数误差</td><td>{business['abs_count_error']}</td><td>abs(pred_count - gt_count)</td></tr>",
            f"      <tr><td>缺陷GT框</td><td>{business['defect_gt_boxes']}</td><td>broken / muddy / empty 原始标记总框数</td></tr>",
            f"      <tr><td>缺陷已检出框</td><td>{business['defect_detected_boxes']}</td><td>三种缺陷被任意缺陷框检出的数量</td></tr>",
            f"      <tr><td>缺陷漏检框</td><td>{business['defect_missed_boxes']}</td><td>三种缺陷原框中没有被缺陷框检出的数量</td></tr>",
            f"      <tr><td>缺陷漏检率</td><td>{business['defect_miss_rate']:.4f}</td><td>三种缺陷原框总共没有被检测出来的概率</td></tr>",
            f"      <tr><td>normal原始框</td><td>{business['normal_gt_boxes']}</td><td>normal 原始标记总框数</td></tr>",
            f"      <tr><td>normal被检为缺陷框</td><td>{business['normal_hit_by_defect_boxes']}</td><td>normal 原框被 broken / muddy / empty 检中的数量</td></tr>",
            f"      <tr><td>normal误检缺陷率</td><td>{business['normal_to_defect_rate']:.4f}</td><td>normal 原框被检测为缺陷框的概率</td></tr>",
            "    </tbody>",
            "  </table>",
            "  <table>",
            "    <thead><tr><th>Class</th><th>GT</th><th>Pred</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>",
            "    <tbody>",
            *class_rows,
            "    </tbody>",
            "  </table>",
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
    class_summary_rows: list[str] = []
    for class_name, metrics in summary["per_class"].items():
        class_summary_rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"  <td>{html.escape(class_name)}</td>",
                    f"  <td>{metrics['gt_boxes']}</td>",
                    f"  <td>{metrics['pred_boxes']}</td>",
                    f"  <td>{metrics['tp']}</td>",
                    f"  <td>{metrics['fp']}</td>",
                    f"  <td>{metrics['fn']}</td>",
                    f"  <td>{metrics['precision']:.4f}</td>",
                    f"  <td>{metrics['recall']:.4f}</td>",
                    f"  <td>{metrics['f1']:.4f}</td>",
                    "</tr>",
                ]
            )
        )

    for item in items:
        item["detail_href"] = write_item_html(report_dir, item, config)
        business = item["business"]
        cards.append(
            "\n".join(
                [
                    '<section class="card">',
                    '  <div class="images">',
                    f'    <div class="panel"><div class="panel-title">Original</div><img src="{item["image_href"]}" alt="{item["stem"]} original"></div>',
                    f'    <div class="panel"><div class="panel-title">Predict Render (visual only)</div><img src="{item["predict_image_href"]}" alt="{item["stem"]} predict"></div>',
                    "  </div>",
                    '  <div class="meta">',
                    f"    <h2>{html.escape(item['stem'])}</h2>",
                    f'    <div class="metric">GT: {item["gt_count"]} | Pred: {item["pred_count"]}</div>',
                    f'    <div class="metric">TP: {item["tp_count"]} | FP: {item["fp_count"]} | FN: {item["fn_count"]}</div>',
                    f'    <div class="metric">计数差值: {business["count_diff"]} | 绝对计数误差: {business["abs_count_error"]}</div>',
                    f'    <div class="metric">缺陷漏检率: {business["defect_miss_rate"]:.4f}</div>',
                    f'    <div class="metric">normal误检缺陷率: {business["normal_to_defect_rate"]:.4f}</div>',
                    f'    <div class="metric"><a href="{item["detail_href"]}" target="_blank" rel="noopener noreferrer">打开单图页面</a></div>',
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
            "  <title>Youge Per-Class Eval Report</title>",
            "  <style>",
            "    body { margin: 0; padding: 24px; font-family: 'Segoe UI', sans-serif; background: #f3f6fa; color: #0f172a; }",
            "    h1 { margin: 0 0 10px; font-size: 28px; }",
            "    .topline { margin: 0 0 20px; color: #334155; font-size: 14px; }",
            "    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }",
            "    .summary-card { background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; padding: 16px; }",
            "    .summary-card .label { color: #475569; font-size: 13px; }",
            "    .summary-card .value { margin-top: 8px; font-size: 28px; font-weight: 700; }",
            "    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; overflow: hidden; margin-bottom: 24px; }",
            "    th, td { padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: center; font-size: 14px; }",
            "    th { background: #eaf2fb; font-weight: 700; }",
            "    .container { display: flex; flex-direction: column; gap: 18px; }",
            "    .card { display: flex; gap: 18px; padding: 16px; background: #fff; border: 1px solid #dbe3ec; border-radius: 16px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); }",
            "    .images { flex: 1 1 auto; display: flex; flex-direction: column; gap: 14px; }",
            "    .panel { overflow: hidden; border-radius: 12px; border: 1px solid #e2e8f0; background: #fff; }",
            "    .panel-title { padding: 10px 14px; font-size: 15px; font-weight: 700; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }",
            "    .panel img { display: block; width: 100%; height: auto; }",
            "    .meta { width: 360px; min-width: 280px; display: flex; flex-direction: column; gap: 10px; }",
            "    .meta h2 { margin: 0; font-size: 20px; word-break: break-all; }",
            "    .metric { font-size: 15px; color: #334155; }",
            "    @media (max-width: 1200px) { .card { flex-direction: column; } .meta { width: auto; min-width: 0; } }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Youge Per-Class Prediction vs GT Report</h1>",
            f'  <p class="topline">split={html.escape(str(config["split"]))} | predict_name={html.escape(str(config["predict_name"]))} | conf={config["conf_threshold"]} | iou={config["iou_threshold"]}</p>',
            '  <p class="topline">说明: 页面中的 Predict Render 仅用于查看后处理可视化效果，intact / defective 等附加文本不会参与 GT / Pred 匹配。指标计算只依赖 labels 目录中的预测框。</p>',
            '  <section class="summary">',
            f'    <div class="summary-card"><div class="label">Images</div><div class="value">{summary["images"]}</div></div>',
            f'    <div class="summary-card"><div class="label">缺陷GT框</div><div class="value">{summary["business"]["defect_gt_boxes"]}</div></div>',
            f'    <div class="summary-card"><div class="label">绝对计数误差</div><div class="value">{summary["business"]["abs_count_error"]}</div></div>',
            f'    <div class="summary-card"><div class="label">缺陷检出率</div><div class="value">{summary["business"]["defect_detect_rate"]:.4f}</div></div>',
            f'    <div class="summary-card"><div class="label">缺陷漏检率</div><div class="value">{summary["business"]["defect_miss_rate"]:.4f}</div></div>',
            f'    <div class="summary-card"><div class="label">normal误检缺陷率</div><div class="value">{summary["business"]["normal_to_defect_rate"]:.4f}</div></div>',
            f'    <div class="summary-card"><div class="label">Overall Precision</div><div class="value">{summary["precision"]:.4f}</div></div>',
            f'    <div class="summary-card"><div class="label">Overall Recall</div><div class="value">{summary["recall"]:.4f}</div></div>',
            "  </section>",
            "  <table>",
            "    <thead><tr><th>业务指标</th><th>值</th><th>说明</th></tr></thead>",
            "    <tbody>",
            f"      <tr><td>计数差值</td><td>{summary['business']['count_diff']}</td><td>pred_total_count - gt_total_count</td></tr>",
            f"      <tr><td>绝对计数误差</td><td>{summary['business']['abs_count_error']}</td><td>abs(pred_count - gt_count)</td></tr>",
            f"      <tr><td>缺陷GT框</td><td>{summary['business']['defect_gt_boxes']}</td><td>broken / muddy / empty 原始标记总框数</td></tr>",
            f"      <tr><td>缺陷已检出框</td><td>{summary['business']['defect_detected_boxes']}</td><td>三种缺陷被任意缺陷框检出的数量</td></tr>",
            f"      <tr><td>缺陷漏检框</td><td>{summary['business']['defect_missed_boxes']}</td><td>三种缺陷原框中没有被缺陷框检出的数量</td></tr>",
            f"      <tr><td>缺陷漏检率</td><td>{summary['business']['defect_miss_rate']:.4f}</td><td>三种缺陷原框总共没有被检测出来的概率</td></tr>",
            f"      <tr><td>normal原始框</td><td>{summary['business']['normal_gt_boxes']}</td><td>normal 原始标记总框数</td></tr>",
            f"      <tr><td>normal被检为缺陷框</td><td>{summary['business']['normal_hit_by_defect_boxes']}</td><td>normal 原框被 broken / muddy / empty 检中的数量</td></tr>",
            f"      <tr><td>normal误检缺陷率</td><td>{summary['business']['normal_to_defect_rate']:.4f}</td><td>normal 原框被检测为缺陷框的概率</td></tr>",
            "    </tbody>",
            "  </table>",
            "  <table>",
            "    <thead><tr><th>Class</th><th>GT</th><th>Pred</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>",
            "    <tbody>",
            *class_summary_rows,
            "    </tbody>",
            "  </table>",
            '  <main class="container">',
            *cards,
            "  </main>",
            "</body>",
            "</html>",
        ]
    )
    (report_dir / "index.html").write_text(html_text, encoding="utf-8")


def build_summary_payload(summary: dict, config: dict) -> dict:
    return {
        "config": {
            "species": config["species"],
            "split": config["split"],
            "predict_name": config["predict_name"],
            "version": config.get("version"),
            "model_version": config.get("version"),
            "postprocess_version": config.get("postprocess_version"),
            "project": config["project"],
            "name": config["name"],
            "conf_threshold": config["conf_threshold"],
            "iou_threshold": config["iou_threshold"],
            "max_items": config["max_items"],
        },
        "class_names": summary.get("class_names", {}),
        "summary": summary,
    }


def write_summary_json(report_dir: Path, summary: dict, config: dict) -> None:
    (report_dir / "summary.json").write_text(
        json.dumps(build_summary_payload(summary, config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary_payload(summary_path: Path, summary: dict, config: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(build_summary_payload(summary, config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.youge_versioning import (
        build_predict_run_name,
        extract_postprocess_version_from_text,
        extract_version_from_text,
        normalize_postprocess_version,
        normalize_version,
    )

    config_path = Path(args.config).resolve() if args.config else script_path.with_name("opt_class_eval.json")
    config = load_config(config_path if config_path.exists() else None)
    config = merge_args_into_config(args, config)
    version = normalize_version(config.get("version"))
    postprocess_version = normalize_postprocess_version(config.get("postprocess_version"))
    if version is None:
        version = extract_version_from_text(config.get("predict_name")) or extract_version_from_text(config.get("name"))
    if postprocess_version is None:
        postprocess_version = extract_postprocess_version_from_text(
            config.get("predict_name")
        ) or extract_postprocess_version_from_text(config.get("name"))
    if version is None:
        predict_runs_dir = repo_root / "src" / "predict" / str(config["species"]) / "runs" / "predict"
        latest_predict_name = find_latest_versioned_dir_name(predict_runs_dir, str(config["predict_name"]))
        version = extract_version_from_text(latest_predict_name)
        if postprocess_version is None:
            postprocess_version = extract_postprocess_version_from_text(latest_predict_name)
    config["version"] = version
    config["postprocess_version"] = postprocess_version
    config["predict_name"] = build_predict_run_name(str(config["predict_name"]), version, postprocess_version)
    config["name"] = build_predict_run_name(str(config["name"]), version, postprocess_version)

    species = config["species"]
    split = str(config["split"]).lower()
    if split not in {"train", "val", "both"}:
        raise ValueError(f"Unsupported split: {split}. Expected one of: train, val, both")

    dataset_root = repo_root / "datasets" / "data" / species
    default_image_dirs, default_gt_label_dirs = resolve_split_dirs(dataset_root, split)
    default_predict_image_dir = repo_root / "src" / "predict" / species / "runs" / "predict" / config["predict_name"]
    default_pred_labels_dir = default_predict_image_dir / "labels"

    image_dir = Path(config["image_dir"]).resolve() if config["image_dir"] else default_image_dirs[0]
    predict_image_dir = (
        Path(config["predict_image_dir"]).resolve() if config["predict_image_dir"] else default_predict_image_dir
    )
    gt_labels_dir = Path(config["gt_labels_dir"]).resolve() if config["gt_labels_dir"] else default_gt_label_dirs[0]
    pred_labels_dir = (
        Path(config["pred_labels_dir"]).resolve() if config["pred_labels_dir"] else default_pred_labels_dir
    )

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
    if not class_names:
        raise FileNotFoundError(f"Class definitions not found in: {dataset_root / 'classes.txt'}")

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

    if (
        split != "both"
        and not (original_images.keys() & predicted_images.keys())
        and config["image_dir"] is None
        and config["gt_labels_dir"] is None
    ):
        split, image_dir, gt_labels_dir, original_images = resolve_best_split(dataset_root, predicted_images, split)
        config["split"] = split
        gt_label_map = collect_txts(gt_labels_dir)
    pred_label_map = collect_txts(pred_labels_dir)

    common_names = sorted(original_images.keys() & predicted_images.keys())
    if not common_names:
        raise FileNotFoundError("No matching original and predicted images were found.")

    max_items = int(config["max_items"]) if config["max_items"] else len(common_names)
    selected_names = common_names[:max_items]
    summary = {
        "images": 0,
        "gt_boxes": 0,
        "pred_boxes": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "avg_pred_boxes_per_image": 0.0,
        "per_class": build_class_metric_map(class_names),
        "business": build_business_metrics(),
    }
    items: list[dict] = []
    count_diagnostic_items: list[dict] = []

    for stem in selected_names:
        gt_detections = load_yolo_labels(gt_label_map.get(stem), class_names, conf_threshold=None)
        pred_detections = load_yolo_labels(pred_label_map.get(stem), class_names, config["conf_threshold"])
        matched = match_detections(gt_detections, pred_detections, float(config["iou_threshold"]))
        gt_defect_detections = filter_detections_by_class_names(gt_detections, DEFECT_CLASS_NAMES)
        pred_defect_detections = filter_detections_by_class_names(pred_detections, DEFECT_CLASS_NAMES)
        gt_normal_detections = filter_detections_by_class_names(gt_detections, {NORMAL_CLASS_NAME})
        defect_group_match = match_detections(
            gt_defect_detections,
            pred_defect_detections,
            float(config["iou_threshold"]),
            require_same_class=False,
        )
        normal_to_defect_match = match_detections(
            gt_normal_detections,
            pred_defect_detections,
            float(config["iou_threshold"]),
            require_same_class=False,
        )

        summary["images"] += 1
        summary["gt_boxes"] += len(gt_detections)
        summary["pred_boxes"] += len(pred_detections)
        summary["tp"] += len(matched["tp"])
        summary["fp"] += len(matched["fp"])
        summary["fn"] += len(matched["fn"])
        summary["business"]["count_diff"] += len(pred_detections) - len(gt_detections)
        summary["business"]["defect_gt_boxes"] += len(gt_defect_detections)
        summary["business"]["defect_pred_boxes"] += len(pred_defect_detections)
        summary["business"]["defect_detected_boxes"] += len(defect_group_match["tp"])
        summary["business"]["defect_missed_boxes"] += len(defect_group_match["fn"])
        summary["business"]["normal_gt_boxes"] += len(gt_normal_detections)
        summary["business"]["normal_hit_by_defect_boxes"] += len(normal_to_defect_match["tp"])

        item_class_metrics = build_class_metric_map(class_names)
        item_business_metrics = build_business_metrics()
        for det in gt_detections:
            item_class_metrics[str(det["class_name"])]["gt_boxes"] += 1
            summary["per_class"][str(det["class_name"])]["gt_boxes"] += 1
        for det in pred_detections:
            item_class_metrics[str(det["class_name"])]["pred_boxes"] += 1
            summary["per_class"][str(det["class_name"])]["pred_boxes"] += 1
        for det in matched["tp"]:
            item_class_metrics[str(det["class_name"])]["tp"] += 1
            summary["per_class"][str(det["class_name"])]["tp"] += 1
        for det in matched["fp"]:
            item_class_metrics[str(det["class_name"])]["fp"] += 1
            summary["per_class"][str(det["class_name"])]["fp"] += 1
        for det in matched["fn"]:
            item_class_metrics[str(det["class_name"])]["fn"] += 1
            summary["per_class"][str(det["class_name"])]["fn"] += 1
        finalize_class_metrics(item_class_metrics)
        item_business_metrics["count_diff"] = len(pred_detections) - len(gt_detections)
        item_business_metrics["defect_gt_boxes"] = len(gt_defect_detections)
        item_business_metrics["defect_pred_boxes"] = len(pred_defect_detections)
        item_business_metrics["defect_detected_boxes"] = len(defect_group_match["tp"])
        item_business_metrics["defect_missed_boxes"] = len(defect_group_match["fn"])
        item_business_metrics["normal_gt_boxes"] = len(gt_normal_detections)
        item_business_metrics["normal_hit_by_defect_boxes"] = len(normal_to_defect_match["tp"])
        finalize_business_metrics(item_business_metrics)

        count_diagnostic_items.append(
            {
                "stem": stem,
                "image_path": str(original_images[stem].resolve()),
                "predict_image_path": str(predicted_images[stem].resolve()),
                "gt_label_path": str(gt_label_map[stem].resolve()) if stem in gt_label_map else None,
                "gt_count": len(gt_detections),
                "pred_count": len(pred_detections),
                "count_diff": item_business_metrics["count_diff"],
                "abs_count_error": item_business_metrics["abs_count_error"],
                "tp_count": len(matched["tp"]),
                "fp_count": len(matched["fp"]),
                "fn_count": len(matched["fn"]),
                "defect_miss_rate": item_business_metrics["defect_miss_rate"],
                "normal_to_defect_rate": item_business_metrics["normal_to_defect_rate"],
                "gt_detections": serialize_detections(gt_detections),
                "pred_detections": serialize_detections(pred_detections),
                "labelimg_server_port": 8765,
            }
        )

        if not args.summary_only:
            tp_lines = [
                format_detection_line(f"{idx}.", det, show_iou=True) for idx, det in enumerate(matched["tp"], start=1)
            ]
            fp_lines = [
                format_detection_line(f"{idx}.", det, show_iou=False) for idx, det in enumerate(matched["fp"], start=1)
            ]
            fn_lines = [
                format_detection_line(f"{idx}.", det, show_iou=False) for idx, det in enumerate(matched["fn"], start=1)
            ]
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
                    "per_class": item_class_metrics,
                    "business": item_business_metrics,
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
    finalize_class_metrics(summary["per_class"])
    finalize_business_metrics(summary["business"])
    summary["count_diagnostics"] = build_count_diagnostics(count_diagnostic_items)
    summary["class_names"] = {str(class_id): class_name for class_id, class_name in class_names.items()}

    if args.summary_only:
        summary_output = Path(args.summary_output).resolve() if args.summary_output else (output_dir / "summary.json")
        write_summary_payload(summary_output, summary, config)
        print(f"Per-class evaluation summary generated at: {summary_output}")
    else:
        write_html(output_dir, items, summary, config)
        write_summary_json(output_dir, summary, config)
        if args.summary_output:
            write_summary_payload(Path(args.summary_output).resolve(), summary, config)
        print(f"Per-class evaluation report generated at: {output_dir}")


if __name__ == "__main__":
    main()
