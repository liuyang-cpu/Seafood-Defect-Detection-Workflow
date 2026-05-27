from __future__ import annotations

import argparse
from datetime import datetime
import json
import html
import os
import sys
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np


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
    parser = argparse.ArgumentParser(description="Create before/after comparison images for the youge dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--species", type=str, default=None, help="Dataset species name, e.g. youge.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to compare, e.g. train or val.")
    parser.add_argument("--predict-name", type=str, default=None, help="Prediction run directory name.")
    parser.add_argument("--before-dir", type=str, default=None, help="Directory of original images.")
    parser.add_argument("--after-dir", type=str, default=None, help="Directory of predicted images.")
    parser.add_argument("--labels-dir", type=str, default=None, help="Directory of label txt files.")
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name.")
    parser.add_argument("--version", type=str, default=None, help="Model version token, e.g. version001.")
    parser.add_argument(
        "--postprocess-version",
        type=str,
        default=None,
        help="Postprocess recipe version token, e.g. pp001.",
    )
    parser.add_argument("--conf-threshold", type=float, default=None, help="Minimum confidence for HTML statistics.")
    parser.add_argument(
        "--layout",
        type=str,
        choices=("horizontal", "vertical"),
        default=None,
        help="Comparison layout direction.",
    )
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory.")
    return parser.parse_args()


def load_compare_config(config_path: Path | None) -> dict:
    defaults = {
        "species": "youge",
        "split": "val",
        "predict_name": "youge_predict",
        "before_dir": None,
        "after_dir": None,
        "labels_dir": None,
        "project": "runs/compare",
        "name": "youge_compare",
        "version": None,
        "postprocess_version": None,
        "conf_threshold": None,
        "layout": "horizontal",
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
        "before_dir",
        "after_dir",
        "labels_dir",
        "project",
        "name",
        "version",
        "postprocess_version",
        "conf_threshold",
        "layout",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if args.exist_ok:
        config["exist_ok"] = True

    return config


def collect_images(image_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem not in images:
            images[path.stem] = path
    return images


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


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image

    scale = target_height / image.shape[0]
    target_width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    if image.shape[1] == target_width:
        return image

    scale = target_width / image.shape[1]
    target_height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def add_title_bar(image: np.ndarray, title: str) -> np.ndarray:
    bar_height = 50
    bar = np.full((bar_height, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack((bar, image))


def build_comparison(before_path: Path, after_path: Path, layout: str) -> np.ndarray:
    before_image = cv2.imread(str(before_path))
    after_image = cv2.imread(str(after_path))

    if before_image is None:
        raise ValueError(f"Failed to read image: {before_path}")
    if after_image is None:
        raise ValueError(f"Failed to read image: {after_path}")

    before_panel = add_title_bar(before_image, "Before")
    after_panel = add_title_bar(after_image, "After")

    if layout == "vertical":
        target_width = max(before_panel.shape[1], after_panel.shape[1])
        before_panel = resize_to_width(before_panel, target_width)
        after_panel = resize_to_width(after_panel, target_width)
        separator = np.full((24, before_panel.shape[1], 3), 255, dtype=np.uint8)
        return np.vstack((before_panel, separator, after_panel))

    target_height = max(before_panel.shape[0], after_panel.shape[0])
    before_panel = resize_to_height(before_panel, target_height)
    after_panel = resize_to_height(after_panel, target_height)
    separator = np.full((before_panel.shape[0], 24, 3), 255, dtype=np.uint8)
    return np.hstack((before_panel, separator, after_panel))


def xywhn_to_xyxy_pixels(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    half_w = width * image_width / 2.0
    half_h = height * image_height / 2.0
    x1 = int(round(x_center * image_width - half_w))
    y1 = int(round(y_center * image_height - half_h))
    x2 = int(round(x_center * image_width + half_w))
    y2 = int(round(y_center * image_height + half_h))
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))
    return x1, y1, x2, y2


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (37, 99, 235),
        (22, 163, 74),
        (234, 88, 12),
        (220, 38, 38),
        (124, 58, 237),
        (8, 145, 178),
    ]
    return palette[class_id % len(palette)]


def build_label_map(labels_dir: Path | None) -> dict[str, Path]:
    if labels_dir is None or not labels_dir.exists():
        return {}

    labels: dict[str, Path] = {}
    for path in sorted(labels_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".txt" and path.stem not in labels:
            labels[path.stem] = path
    return labels


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


def parse_prediction_label(
    label_path: Path, class_names: dict[int, str], conf_threshold: float | None = None
) -> dict[str, object]:
    detections: list[dict[str, str | float | int | None]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue

        try:
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue

        confidence = None
        if len(parts) >= 6:
            try:
                confidence = float(parts[5])
            except ValueError:
                confidence = None

        if conf_threshold is not None and confidence is not None and confidence < conf_threshold:
            continue

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "confidence": confidence,
                "xywhn": (x_center, y_center, width, height),
            }
        )

    return {"count": len(detections), "detections": detections}


def render_prediction_overlay(
    image_path: Path,
    prediction_info: dict[str, object],
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    rendered = image.copy()
    image_height, image_width = rendered.shape[:2]
    for detection in prediction_info["detections"]:
        class_id = int(detection["class_id"])
        x_center, y_center, width, height = detection["xywhn"]
        x1, y1, x2, y2 = xywhn_to_xyxy_pixels(
            x_center,
            y_center,
            width,
            height,
            image_width,
            image_height,
        )
        color = color_for_class(class_id)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = str(detection["class_name"])
        confidence = detection["confidence"]
        if confidence is not None:
            label = f"{label} {confidence:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_top = max(0, y1 - text_height - baseline - 8)
        text_bottom = min(image_height - 1, text_top + text_height + baseline + 8)
        text_right = min(image_width - 1, x1 + text_width + 12)
        cv2.rectangle(rendered, (x1, text_top), (text_right, text_bottom), color, -1, cv2.LINE_AA)
        cv2.putText(
            rendered,
            label,
            (x1 + 6, text_bottom - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return rendered


def build_comparison_from_images(before_image: np.ndarray, after_image: np.ndarray, layout: str) -> np.ndarray:
    before_panel = add_title_bar(before_image, "Before")
    after_panel = add_title_bar(after_image, "After")

    if layout == "vertical":
        target_width = max(before_panel.shape[1], after_panel.shape[1])
        before_panel = resize_to_width(before_panel, target_width)
        after_panel = resize_to_width(after_panel, target_width)
        separator = np.full((24, before_panel.shape[1], 3), 255, dtype=np.uint8)
        return np.vstack((before_panel, separator, after_panel))

    target_height = max(before_panel.shape[0], after_panel.shape[0])
    before_panel = resize_to_height(before_panel, target_height)
    after_panel = resize_to_height(after_panel, target_height)
    separator = np.full((before_panel.shape[0], 24, 3), 255, dtype=np.uint8)
    return np.hstack((before_panel, separator, after_panel))


def path_to_href(from_dir: Path, target_path: Path) -> str:
    relative_path = Path(os.path.relpath(target_path, start=from_dir))
    return quote(relative_path.as_posix(), safe="/")


def write_index_html(report_dir: Path, output_dir: Path, items: list[dict[str, str | None]]) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cards: list[str] = []
    for item in items:
        meta_lines = [f'    <h2>{html.escape(str(item["stem"]))}</h2>']
        if item["box_count"] is not None:
            meta_lines.append(f'    <div class="summary">框数量：{item["box_count"]}</div>')
        if item["detection_lines"]:
            meta_lines.append('    <div class="summary">检测结果：</div>')
            meta_lines.append('    <ul class="detection-list">')
            meta_lines.extend(f"      {line}" for line in item["detection_lines"])
            meta_lines.append("    </ul>")
        if item["label_href"]:
            meta_lines.append(
                f'    <a class="label-link" href="{item["label_href"]}" target="_blank" rel="noopener noreferrer">打开标签 txt</a>'
            )
        cards.append(
            "\n".join(
                [
                    '<section class="card">',
                    f'  <img src="{item["image_href"]}" alt="{item["stem"]} comparison">',
                    '  <div class="meta">',
                    *meta_lines,
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
            "  <title>Comparison Report</title>",
            "  <style>",
            "    :root { color-scheme: light; }",
            "    body { margin: 0; padding: 24px; font-family: 'Segoe UI', sans-serif; background: #f5f7fa; color: #1f2937; }",
            "    h1 { margin: 0 0 24px; font-size: 28px; }",
            "    .header { margin-bottom: 24px; }",
            "    .generated-at { font-size: 14px; color: #475569; }",
            "    .container { display: flex; flex-direction: column; gap: 18px; }",
            "    .card { display: flex; align-items: flex-start; gap: 18px; padding: 16px; background: #ffffff; border: 1px solid #dbe3ec; border-radius: 14px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); }",
            "    .card img { max-width: min(72vw, 1200px); height: auto; border-radius: 10px; border: 1px solid #e5e7eb; background: #fff; }",
            "    .meta { min-width: 180px; padding-top: 6px; display: flex; flex-direction: column; gap: 10px; }",
            "    .meta h2 { margin: 0; font-size: 18px; word-break: break-all; }",
            "    .summary { font-size: 14px; color: #334155; }",
            "    .detection-list { margin: 0; padding-left: 20px; color: #0f172a; }",
            "    .detection-list li { margin: 4px 0; }",
            "    .label-link { display: inline-block; padding: 10px 14px; border-radius: 10px; text-decoration: none; color: #ffffff; background: #2563eb; }",
            "    .label-link:hover { background: #1d4ed8; }",
            "    @media (max-width: 960px) { .card { flex-direction: column; } .card img { max-width: 100%; } }",
            "  </style>",
            "</head>",
            "<body>",
            '  <header class="header">',
            "    <h1>Comparison Report</h1>",
            f"    <div class=\"generated-at\">生成时间：{generated_at}</div>",
            "  </header>",
            '  <main class="container">',
            *cards,
            "  </main>",
            "</body>",
            "</html>",
        ]
    )

    (report_dir / "index.html").write_text(html_text, encoding="utf-8")


def ensure_output_dir(output_dir: Path, exist_ok: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not exist_ok:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.youge_versioning import (
        build_predict_run_name,
        extract_version_from_text,
        extract_postprocess_version_from_text,
        normalize_postprocess_version,
        normalize_version,
    )

    config_path = Path(args.config).resolve() if args.config else script_path.with_name("compare_youge.json")

    compare_config = load_compare_config(config_path if config_path.exists() else None)
    compare_config = merge_args_into_config(args, compare_config)
    version = normalize_version(compare_config.get("version"))
    postprocess_version = normalize_postprocess_version(compare_config.get("postprocess_version"))
    if version is None:
        version = extract_version_from_text(compare_config.get("predict_name")) or extract_version_from_text(compare_config.get("name"))
    if version is None:
        version = (
            extract_version_from_text(compare_config.get("after_dir"))
            or extract_version_from_text(compare_config.get("labels_dir"))
        )
    if postprocess_version is None:
        postprocess_version = (
            extract_postprocess_version_from_text(compare_config.get("predict_name"))
            or extract_postprocess_version_from_text(compare_config.get("name"))
            or extract_postprocess_version_from_text(compare_config.get("after_dir"))
            or extract_postprocess_version_from_text(compare_config.get("labels_dir"))
        )
    if version is None:
        predict_runs_dir = repo_root / "src" / "predict" / str(compare_config["species"]) / "runs" / "predict"
        latest_predict_name = find_latest_versioned_dir_name(predict_runs_dir, str(compare_config["predict_name"]))
        version = extract_version_from_text(latest_predict_name)
        if postprocess_version is None:
            postprocess_version = extract_postprocess_version_from_text(latest_predict_name)
    compare_config["version"] = version
    compare_config["postprocess_version"] = postprocess_version
    compare_config["predict_name"] = build_predict_run_name(
        str(compare_config["predict_name"]),
        version,
        postprocess_version,
    )
    compare_config["name"] = build_predict_run_name(
        str(compare_config["name"]),
        version,
        postprocess_version,
    )

    species = compare_config["species"]
    split = str(compare_config["split"]).lower()
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}. Expected one of: train, val")

    dataset_root = repo_root / "datasets" / "data" / species
    default_before_dir = dataset_root / "images" / split
    default_after_dir = repo_root / "src" / "predict" / species / "runs" / "predict" / compare_config["predict_name"]
    before_dir = Path(compare_config["before_dir"]).resolve() if compare_config["before_dir"] else default_before_dir
    after_dir = Path(compare_config["after_dir"]).resolve() if compare_config["after_dir"] else default_after_dir
    labels_dir = Path(compare_config["labels_dir"]).resolve() if compare_config["labels_dir"] else None
    layout = str(compare_config["layout"]).lower()
    conf_threshold = compare_config["conf_threshold"]
    if conf_threshold is None:
        predict_config_path = repo_root / "src" / "predict" / species / f"predict_{species}.json"
        if predict_config_path.exists():
            with predict_config_path.open("r", encoding="utf-8") as f:
                predict_config = json.load(f)
            conf_threshold = predict_config.get("conf")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset species directory not found: {dataset_root}")
    if not before_dir.exists():
        raise FileNotFoundError(f"Original image directory not found: {before_dir}")
    if not after_dir.exists():
        raise FileNotFoundError(f"Prediction run directory not found: {after_dir}")
    if layout not in {"horizontal", "vertical"}:
        raise ValueError(f"Unsupported layout: {layout}. Expected one of: horizontal, vertical")

    compare_config["project"] = resolve_output_dir(script_path.parent, compare_config["project"])
    output_dir = Path(compare_config["project"]) / compare_config["name"]
    ensure_output_dir(output_dir, compare_config["exist_ok"])
    report_dir = output_dir.parent

    before_images = collect_images(before_dir)
    label_files = build_label_map(labels_dir)
    prediction_label_files = build_label_map(after_dir / "labels")
    class_names = load_class_names(dataset_root)
    common_names = sorted(before_images.keys())

    if not common_names:
        raise FileNotFoundError(f"No source images found in: {before_dir}")

    generated = 0
    report_items: list[dict[str, str | None]] = []
    for image_stem in common_names:
        label_path = label_files.get(image_stem)
        prediction_label_path = prediction_label_files.get(image_stem)
        prediction_info = (
            parse_prediction_label(prediction_label_path, class_names, conf_threshold)
            if prediction_label_path
            else {"count": 0, "detections": []}
        )
        before_image = cv2.imread(str(before_images[image_stem]))
        if before_image is None:
            raise ValueError(f"Failed to read image: {before_images[image_stem]}")
        after_image = render_prediction_overlay(before_images[image_stem], prediction_info)
        comparison = build_comparison_from_images(before_image, after_image, layout)
        output_path = output_dir / f"{image_stem}.jpg"
        if not cv2.imwrite(str(output_path), comparison):
            raise OSError(f"Failed to write comparison image: {output_path}")
        detection_lines = []
        for idx, detection in enumerate(prediction_info["detections"], start=1):
            class_name = html.escape(str(detection["class_name"]))
            confidence = detection["confidence"]
            if confidence is None:
                detection_lines.append(f"<li>{idx}. {class_name}</li>")
            else:
                detection_lines.append(f"<li>{idx}. {class_name} ({confidence:.4f})</li>")
        report_items.append(
            {
                "stem": image_stem,
                "image_name": output_path.name,
                "image_href": path_to_href(report_dir, output_path),
                "box_count": prediction_info["count"],
                "detection_lines": detection_lines,
                "label_href": path_to_href(report_dir, label_path.resolve()) if label_path else None,
            }
        )
        generated += 1

    write_index_html(report_dir, output_dir, report_items)
    print(f"Comparison finished. Generated {generated} image(s) in: {output_dir} and report in: {report_dir}")


if __name__ == "__main__":
    main()
