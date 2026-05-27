from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import quote, urlencode

TARGET_CLASSES = ["broken", "normal", "muddy", "empty"]
COMPARE_COLORS = ["#2563eb", "#dc2626", "#0f766e", "#d97706", "#7c3aed", "#db2777"]


def remove_tree_with_retry(path: Path, *, retries: int = 8, delay_seconds: float = 0.5) -> None:
    if not path.exists():
        return

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(delay_seconds)

    raise RuntimeError(
        "无法删除旧输出目录，目录可能正被其它程序占用。\n"
        f"目录: {path}\n"
        "请先关闭可能打开该目录或其文件的程序，例如资源管理器、图片查看器、labelImg 或其它脚本后重试。"
    ) from last_error


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
    parser = argparse.ArgumentParser(description="Run per-class evaluation serially with different conf thresholds.")
    parser.add_argument(
        "--conf-thresholds",
        nargs="+",
        default=None,
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
    parser.add_argument(
        "--versions", nargs="+", default=None, help="Multiple model versions, e.g. version002 version003."
    )
    parser.add_argument(
        "--postprocess-version", type=str, default=None, help="Postprocess recipe version token, e.g. pp001."
    )
    parser.add_argument(
        "--postprocess-versions",
        nargs="+",
        default=None,
        help="Multiple postprocess recipe versions, e.g. pp001 pp002.",
    )
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU threshold for matching.")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum images to show in HTML.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory.")
    args = parser.parse_args()
    args.conf_thresholds = parse_conf_thresholds(args.conf_thresholds) if args.conf_thresholds else None
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
        "--postprocess-version": args.postprocess_version,
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
    project = Path(args.project) if args.project else Path("runs/opt-class")
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


def path_to_href(from_dir: Path, target_path: Path) -> str:
    relative_path = Path(os.path.relpath(target_path, start=from_dir))
    return quote(relative_path.as_posix(), safe="/")


def write_labelimg_launcher(
    launcher_dir: Path,
    batch_dir: Path,
    item: dict,
    labelimg_exe: Path | None,
    class_file_path: Path | None,
) -> str | None:
    if labelimg_exe is None or class_file_path is None:
        return None
    image_path = Path(str(item["image_path"]))
    gt_label_path = item.get("gt_label_path")
    if gt_label_path is None:
        return None
    label_path = Path(str(gt_label_path))
    params = urlencode(
        {
            "labelimg_exe": str(labelimg_exe),
            "image_dir": str(image_path),
            "class_file": str(class_file_path),
            "save_dir": str(label_path.parent),
        },
        quote_via=quote,
    )
    port = int(item.get("labelimg_server_port", 8765))
    return f"http://127.0.0.1:{port}/open?{params}"


def build_overlay_panel_html(
    image_href: str,
    title: str,
    detections: list[dict],
    stroke_color: str = "#2563eb",
    label_prefix: str = "GT",
    show_confidence: bool = False,
) -> str:
    if not detections:
        return (
            '<div class="example-panel">'
            f'<div class="example-panel-title">{html.escape(title)}</div>'
            f'<img src="{image_href}" alt="{html.escape(title)}">'
            "</div>"
        )

    svg_parts: list[str] = []
    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = [float(value) for value in det["xyxy"]]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        label_text = f"{label_prefix}{index} {det['class_name']}"
        confidence = det.get("confidence")
        if show_confidence and confidence is not None:
            label_text = f"{label_text} {float(confidence):.2f}"
        label_x = x1
        label_y = y1 + 0.028 if y1 < 0.05 else y1 - 0.010
        label_width = min(0.34, max(0.11, 0.022 + len(label_text) * 0.013))
        svg_parts.extend(
            [
                f'<rect x="{x1:.6f}" y="{y1:.6f}" width="{width:.6f}" height="{height:.6f}" class="overlay-box" stroke="{stroke_color}"></rect>',
                f'<rect x="{label_x:.6f}" y="{max(0.002, label_y - 0.023):.6f}" width="{label_width:.6f}" height="0.030000" class="overlay-label-bg" fill="{stroke_color}"></rect>',
                f'<text x="{(label_x + 0.008):.6f}" y="{label_y:.6f}" class="overlay-label">{html.escape(label_text)}</text>',
            ]
        )

    return "\n".join(
        [
            '<div class="example-panel">',
            f'  <div class="example-panel-title">{html.escape(title)}</div>',
            '  <div class="overlay-stage">',
            f'    <img src="{image_href}" alt="{html.escape(title)}">',
            '    <svg class="overlay-svg" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">',
            *svg_parts,
            "    </svg>",
            "  </div>",
            "</div>",
        ]
    )


def write_gt_editor_html(
    editor_dir: Path, batch_dir: Path, conf_threshold: object, item: dict, class_names: dict[str, str]
) -> str:
    editor_dir.mkdir(parents=True, exist_ok=True)
    editor_path = editor_dir / f"{item['stem']}.html"
    image_href = path_to_href(editor_dir, Path(str(item["image_path"])))
    label_path = item.get("gt_label_path")
    label_href = path_to_href(editor_dir, Path(str(label_path))) if label_path else ""
    payload = {
        "stem": item["stem"],
        "conf_threshold": conf_threshold,
        "image_href": image_href,
        "label_href": label_href,
        "label_path": label_path,
        "class_names": class_names,
        "boxes": item.get("gt_detections", []),
    }
    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f"  <title>{html.escape(str(item['stem']))} - GT Editor</title>",
            "  <style>",
            "    body { margin: 0; padding: 18px; font-family: 'Segoe UI', sans-serif; background: #f3f6fa; color: #0f172a; }",
            "    .layout { display: grid; grid-template-columns: minmax(0, 1.2fr) 360px; gap: 18px; }",
            "    .stage-card, .side-card { background: #fff; border: 1px solid #dbe3ec; border-radius: 16px; padding: 14px; }",
            "    .stage-wrap { position: relative; overflow: hidden; border-radius: 12px; border: 1px solid #e2e8f0; background: #fff; }",
            "    .stage-wrap img { display: block; width: 100%; height: auto; }",
            "    .stage-wrap svg { position: absolute; inset: 0; width: 100%; height: 100%; }",
            "    .box-rect { fill: rgba(37, 99, 235, 0.08); stroke: #2563eb; stroke-width: 0.003; cursor: move; }",
            "    .box-rect.selected { fill: rgba(220, 38, 38, 0.10); stroke: #dc2626; }",
            "    .box-label { font-size: 0.028px; font-weight: 700; paint-order: stroke; stroke: #ffffff; stroke-width: 0.01px; }",
            "    .toolbar { display: grid; gap: 12px; }",
            "    .toolbar h1 { margin: 0; font-size: 22px; }",
            "    .hint { margin: 0; color: #475569; font-size: 13px; }",
            "    .box-list { display: grid; gap: 8px; max-height: 220px; overflow-y: auto; }",
            "    .box-item { border: 1px solid #dbe3ec; border-radius: 10px; padding: 8px 10px; cursor: pointer; }",
            "    .box-item.active { border-color: #2563eb; background: #eff6ff; }",
            "    .controls { display: grid; gap: 10px; }",
            "    .row { display: grid; gap: 8px; }",
            "    .row.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }",
            "    label { font-size: 13px; color: #334155; }",
            "    input, select, textarea, button { font: inherit; }",
            "    input, select, textarea { width: 100%; box-sizing: border-box; padding: 8px 10px; border: 1px solid #cbd5e1; border-radius: 10px; }",
            "    textarea { min-height: 170px; resize: vertical; }",
            "    button { padding: 9px 12px; border: 0; border-radius: 10px; background: #2563eb; color: #fff; cursor: pointer; }",
            "    button.secondary { background: #475569; }",
            "    button.danger { background: #dc2626; }",
            "    .move-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }",
            "    .footer-actions { display: grid; gap: 8px; }",
            "    .meta-line { font-size: 13px; color: #475569; word-break: break-all; }",
            "    @media (max-width: 1200px) { .layout { grid-template-columns: 1fr; } }",
            "  </style>",
            "</head>",
            "<body>",
            '  <div class="layout">',
            '    <section class="stage-card">',
            f'      <div class="toolbar"><h1>{html.escape(str(item["stem"]))}</h1><p class="hint">拖拽框可以移动位置。右侧可以微调、改类别、增删框，并导出新的 YOLO txt。浏览器页不能直接覆盖本地标签文件，所以导出后需要你手动替换。</p></div>',
            '      <div class="stage-wrap" id="stageWrap">',
            f'        <img id="mainImage" src="{image_href}" alt="{html.escape(str(item["stem"]))}">',
            '        <svg id="overlaySvg" viewBox="0 0 1 1" preserveAspectRatio="none"></svg>',
            "      </div>",
            "    </section>",
            '    <aside class="side-card">',
            '      <div class="controls">',
            f'        <div class="meta-line">conf={html.escape(str(conf_threshold))}</div>',
            f'        <div class="meta-line">GT label path: {html.escape(str(label_path or ""))}</div>',
            '        <div class="row"><button type="button" id="openLabelBtn" class="secondary">打开当前 GT txt</button></div>',
            '        <div class="row"><button type="button" id="addBoxBtn">新增框</button></div>',
            '        <div class="row"><div id="boxList" class="box-list"></div></div>',
            '        <div class="row two"><label>class<select id="classSelect"></select></label><label>step<input id="stepInput" type="number" min="0.0001" max="0.1" step="0.0005" value="0.002"></label></div>',
            '        <div class="move-grid">',
            '          <div></div><button type="button" data-move="up">上移</button><div></div>',
            '          <button type="button" data-move="left">左移</button><button type="button" class="secondary" id="centerBtn">居中显示</button><button type="button" data-move="right">右移</button>',
            '          <div></div><button type="button" data-move="down">下移</button><div></div>',
            "        </div>",
            '        <div class="row two"><button type="button" data-resize="w-">宽度-</button><button type="button" data-resize="w+">宽度+</button></div>',
            '        <div class="row two"><button type="button" data-resize="h-">高度-</button><button type="button" data-resize="h+">高度+</button></div>',
            '        <div class="row"><button type="button" id="deleteBoxBtn" class="danger">删除当前框</button></div>',
            '        <div class="footer-actions">',
            '          <button type="button" id="refreshTxtBtn" class="secondary">刷新导出内容</button>',
            '          <button type="button" id="downloadBtn">下载新的 txt</button>',
            '          <textarea id="exportText"></textarea>',
            "        </div>",
            "      </div>",
            "    </aside>",
            "  </div>",
            f"  <script>const initialData = {json.dumps(payload, ensure_ascii=False)};</script>",
            "  <script>",
            "const classMap = initialData.class_names || {};",
            "const classIds = Object.keys(classMap).sort((a, b) => Number(a) - Number(b));",
            "let boxes = (initialData.boxes || []).map((box) => ({ class_id: Number(box.class_id), xyxy: box.xyxy.map(Number) }));",
            "let selectedIndex = boxes.length ? 0 : -1;",
            "let dragState = null;",
            "const svg = document.getElementById('overlaySvg');",
            "const boxList = document.getElementById('boxList');",
            "const classSelect = document.getElementById('classSelect');",
            "const exportText = document.getElementById('exportText');",
            "const stepInput = document.getElementById('stepInput');",
            "function clamp01(v) { return Math.max(0, Math.min(1, v)); }",
            "function normalizeBox(box) {",
            "  let [x1, y1, x2, y2] = box.xyxy;",
            "  x1 = clamp01(x1); y1 = clamp01(y1); x2 = clamp01(x2); y2 = clamp01(y2);",
            "  if (x2 < x1) [x1, x2] = [x2, x1];",
            "  if (y2 < y1) [y1, y2] = [y2, y1];",
            "  box.xyxy = [x1, y1, x2, y2];",
            "}",
            "function getClassName(classId) { return classMap[String(classId)] || String(classId); }",
            "function getStep() { const step = Number(stepInput.value || 0.002); return step > 0 ? step : 0.002; }",
            "function selectBox(index) { selectedIndex = index; render(); }",
            "function renderBoxList() {",
            "  boxList.innerHTML = '';",
            "  boxes.forEach((box, index) => {",
            "    const div = document.createElement('div');",
            "    div.className = 'box-item' + (index === selectedIndex ? ' active' : '');",
            "    div.textContent = `#${index + 1} ${getClassName(box.class_id)} | ${box.xyxy.map((v) => v.toFixed(4)).join(', ')}`;",
            "    div.onclick = () => selectBox(index);",
            "    boxList.appendChild(div);",
            "  });",
            "}",
            "function renderClassSelect() {",
            "  classSelect.innerHTML = '';",
            "  classIds.forEach((classId) => {",
            "    const option = document.createElement('option');",
            "    option.value = classId;",
            "    option.textContent = `${classId} ${getClassName(Number(classId))}`;",
            "    classSelect.appendChild(option);",
            "  });",
            "  if (selectedIndex >= 0 && boxes[selectedIndex]) classSelect.value = String(boxes[selectedIndex].class_id);",
            "}",
            "function renderSvg() {",
            "  svg.innerHTML = '';",
            "  boxes.forEach((box, index) => {",
            "    normalizeBox(box);",
            "    const [x1, y1, x2, y2] = box.xyxy;",
            "    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');",
            "    rect.setAttribute('x', x1); rect.setAttribute('y', y1);",
            "    rect.setAttribute('width', Math.max(0.0001, x2 - x1));",
            "    rect.setAttribute('height', Math.max(0.0001, y2 - y1));",
            "    rect.setAttribute('class', 'box-rect' + (index === selectedIndex ? ' selected' : ''));",
            "    rect.addEventListener('mousedown', (event) => { dragState = { index, startX: event.clientX, startY: event.clientY, box: [...box.xyxy] }; selectedIndex = index; render(); event.preventDefault(); });",
            "    rect.addEventListener('click', () => selectBox(index));",
            "    svg.appendChild(rect);",
            "    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');",
            "    label.setAttribute('x', x1); label.setAttribute('y', Math.max(0.02, y1 - 0.01));",
            "    label.setAttribute('class', 'box-label');",
            "    label.textContent = `GT${index + 1} ${getClassName(box.class_id)}`;",
            "    svg.appendChild(label);",
            "  });",
            "}",
            "function boxToYoloLine(box) {",
            "  const [x1, y1, x2, y2] = box.xyxy;",
            "  const xc = (x1 + x2) / 2; const yc = (y1 + y2) / 2; const w = x2 - x1; const h = y2 - y1;",
            "  return `${box.class_id} ${xc.toFixed(6)} ${yc.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;",
            "}",
            "function refreshExportText() { exportText.value = boxes.map(boxToYoloLine).join('\\n') + (boxes.length ? '\\n' : ''); }",
            "function render() { renderBoxList(); renderClassSelect(); renderSvg(); refreshExportText(); }",
            "function moveSelected(dx, dy) {",
            "  if (selectedIndex < 0 || !boxes[selectedIndex]) return;",
            "  const box = boxes[selectedIndex];",
            "  box.xyxy = [box.xyxy[0] + dx, box.xyxy[1] + dy, box.xyxy[2] + dx, box.xyxy[3] + dy];",
            "  normalizeBox(box); render();",
            "}",
            "function resizeSelected(dw, dh) {",
            "  if (selectedIndex < 0 || !boxes[selectedIndex]) return;",
            "  const box = boxes[selectedIndex];",
            "  const [x1, y1, x2, y2] = box.xyxy;",
            "  const cx = (x1 + x2) / 2; const cy = (y1 + y2) / 2;",
            "  const halfW = Math.max(0.0005, (x2 - x1) / 2 + dw);",
            "  const halfH = Math.max(0.0005, (y2 - y1) / 2 + dh);",
            "  box.xyxy = [cx - halfW, cy - halfH, cx + halfW, cy + halfH];",
            "  normalizeBox(box); render();",
            "}",
            "document.addEventListener('mousemove', (event) => {",
            "  if (!dragState) return;",
            "  const wrap = document.getElementById('stageWrap').getBoundingClientRect();",
            "  const dx = (event.clientX - dragState.startX) / wrap.width;",
            "  const dy = (event.clientY - dragState.startY) / wrap.height;",
            "  boxes[dragState.index].xyxy = [dragState.box[0] + dx, dragState.box[1] + dy, dragState.box[2] + dx, dragState.box[3] + dy];",
            "  normalizeBox(boxes[dragState.index]);",
            "  renderSvg(); renderBoxList(); refreshExportText();",
            "});",
            "document.addEventListener('mouseup', () => { dragState = null; });",
            "classSelect.addEventListener('change', () => { if (selectedIndex >= 0 && boxes[selectedIndex]) { boxes[selectedIndex].class_id = Number(classSelect.value); render(); } });",
            "document.querySelectorAll('[data-move]').forEach((btn) => btn.addEventListener('click', () => { const step = getStep(); const action = btn.getAttribute('data-move'); if (action === 'left') moveSelected(-step, 0); if (action === 'right') moveSelected(step, 0); if (action === 'up') moveSelected(0, -step); if (action === 'down') moveSelected(0, step); }));",
            "document.querySelectorAll('[data-resize]').forEach((btn) => btn.addEventListener('click', () => { const step = getStep(); const action = btn.getAttribute('data-resize'); if (action === 'w-') resizeSelected(-step, 0); if (action === 'w+') resizeSelected(step, 0); if (action === 'h-') resizeSelected(0, -step); if (action === 'h+') resizeSelected(0, step); }));",
            "document.getElementById('addBoxBtn').addEventListener('click', () => { boxes.push({ class_id: classIds.length ? Number(classIds[0]) : 0, xyxy: [0.35, 0.35, 0.55, 0.55] }); selectedIndex = boxes.length - 1; render(); });",
            "document.getElementById('deleteBoxBtn').addEventListener('click', () => { if (selectedIndex < 0) return; boxes.splice(selectedIndex, 1); selectedIndex = Math.min(selectedIndex, boxes.length - 1); render(); });",
            "document.getElementById('refreshTxtBtn').addEventListener('click', refreshExportText);",
            "document.getElementById('downloadBtn').addEventListener('click', () => { refreshExportText(); const blob = new Blob([exportText.value], { type: 'text/plain;charset=utf-8' }); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `${initialData.stem}.txt`; link.click(); URL.revokeObjectURL(url); });",
            "document.getElementById('openLabelBtn').addEventListener('click', () => { if (initialData.label_href) window.location.href = initialData.label_href; });",
            "document.getElementById('centerBtn').addEventListener('click', () => { if (selectedIndex >= 0) document.getElementById('stageWrap').scrollIntoView({ behavior: 'smooth', block: 'center' }); });",
            "render();",
            "  </script>",
            "</body>",
            "</html>",
        ]
    )
    editor_path.write_text(html_text, encoding="utf-8")
    return path_to_href(batch_dir, editor_path)


def load_eval_config(config_path: Path | None) -> dict:
    defaults = {
        "species": "youge",
        "split": "both",
        "predict_name": "youge_predict",
        "project": "runs/opt-class",
        "name": "youge_opt_class_report",
        "version": None,
        "versions": [],
        "postprocess_version": None,
        "postprocess_versions": [],
        "conf_thresholds": ["0.70-0.90:0.01"],
        "iou_threshold": 0.5,
        "max_items": None,
        "exist_ok": True,
    }
    if config_path is None or not config_path.exists():
        return defaults

    with config_path.open("r", encoding="utf-8-sig") as f:
        user_config = json.load(f)
    defaults.update(user_config)
    return defaults


def ensure_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_predict_run_config(predict_run_dir: Path) -> dict:
    run_config_path = predict_run_dir / "run_config.json"
    if not run_config_path.exists():
        return {}
    with run_config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_labelimg_exe() -> Path | None:
    candidates = [
        Path(r"E:\Users\liuyang\anaconda3\envs\labelimg\Scripts\labelImg.exe"),
        Path(r"E:\Users\liuyang\anaconda3\envs\labelimg\Scripts\labelimg.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def unique_conf_thresholds(values: list[float]) -> list[float]:
    unique: list[float] = []
    seen: set[float] = set()
    for value in values:
        normalized = round(float(value), 10)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(float(value))
    unique.sort()
    return unique


def version_sort_key(version: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(version) if ch.isdigit())
    return (int(digits) if digits else -1, str(version).lower())


def build_eval_target_label(version: str, postprocess_version: str | None) -> str:
    return f"{version}_{postprocess_version}" if postprocess_version else version


def format_token(value: object) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".").replace("-", "neg_").replace(".", "_")
    return str(value).replace("-", "_").replace(".", "_").replace(" ", "_")


def build_series_chart(
    title: str,
    subtitle: str,
    rows: list[dict],
    series: list[tuple[str, str, str]],
    peak_key: str | None = None,
    peak_label: str = "best",
    min_key: str | None = None,
    min_label: str = "min",
    extra_min_keys: list[tuple[str, str]] | None = None,
) -> str:
    plot_width = 760
    plot_height = 240
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
    values = []
    for row in rows:
        values.extend(float(row[key]) for key, _, _ in series)
    y_min = min(values)
    y_max = max(values)
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
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis-label">{tick_value:.4f}</text>',
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
    for key, label, color in series:
        points = []
        point_marks = []
        data = [float(row[key]) for row in rows]
        max_value = max(data)
        min_value = min(data)
        for row in rows:
            value = float(row[key])
            conf = float(row["conf_threshold"])
            x = x_pos(conf)
            y = y_pos(value)
            points.append(f"{x:.2f},{y:.2f}")
            mark_lines = [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"></circle>']
            if peak_key == key and value == max_value:
                mark_lines.extend(
                    [
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" class="peak-ring peak-max"></circle>',
                        (
                            f'<text x="{x:.2f}" y="{y - 14:.2f}" text-anchor="middle" class="point-label point-max" '
                            f'fill="{color}">{html.escape(peak_label)} {value:.4f} @ {conf:g}</text>'
                        ),
                    ]
                )
            if min_key == key and value == min_value:
                mark_lines.extend(
                    [
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" class="peak-ring" stroke="{color}"></circle>',
                        (
                            f'<text x="{x:.2f}" y="{y + 18:.2f}" text-anchor="middle" class="point-label point-max" '
                            f'fill="{color}">{html.escape(min_label)} {value:.4f} @ {conf:g}</text>'
                        ),
                    ]
                )
            if extra_min_keys:
                for extra_key, extra_label in extra_min_keys:
                    if extra_key == key and value == min_value:
                        mark_lines.extend(
                            [
                                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" class="peak-ring" stroke="{color}"></circle>',
                                (
                                    f'<text x="{x:.2f}" y="{y + 32:.2f}" text-anchor="middle" class="point-label point-max" '
                                    f'fill="{color}">{html.escape(extra_label)} {value:.4f} @ {conf:g}</text>'
                                ),
                            ]
                        )
            point_marks.append("\n".join(mark_lines))
        series_html.append(
            "\n".join(
                [
                    f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>',
                    *point_marks,
                ]
            )
        )
        legend_items.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{color};"></span>{html.escape(label)}</div>'
        )

    return "\n".join(
        [
            '<section class="chart-card">',
            f'  <div class="chart-header"><div><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div><div class="legend">{"".join(legend_items)}</div></div>',
            f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            *grid_lines,
            f'    <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis-line"></line>',
            f'    <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis-line"></line>',
            *x_ticks,
            *series_html,
            "  </svg>",
            "</section>",
        ]
    )


def build_metric_chart(title: str, subtitle: str, rows: list[dict], class_name: str) -> str:
    return build_series_chart(
        title=title,
        subtitle=subtitle,
        rows=rows,
        series=[
            (f"{class_name}_precision", "Precision", "#0f766e"),
            (f"{class_name}_recall", "Recall", "#2563eb"),
            (f"{class_name}_f1", "F1", "#dc2626"),
        ],
        peak_key=f"{class_name}_f1",
        peak_label="best F1",
    )


def build_business_chart(title: str, subtitle: str, rows: list[dict]) -> str:
    return build_series_chart(
        title=title,
        subtitle=subtitle,
        rows=rows,
        series=[
            ("defect_detect_rate", "缺陷筛出率", "#0f766e"),
            ("defect_miss_rate", "缺陷漏检率", "#dc2626"),
            ("normal_to_defect_rate", "正常品误剔率", "#d97706"),
        ],
        peak_key="defect_detect_rate",
        peak_label="最佳筛出率",
        min_key="defect_miss_rate",
        min_label="最低漏检率",
        extra_min_keys=[("normal_to_defect_rate", "最低误剔率")],
    )


def build_count_chart(title: str, subtitle: str, rows: list[dict]) -> str:
    return build_series_chart(
        title=title,
        subtitle=subtitle,
        rows=rows,
        series=[
            ("abs_count_error", "绝对计数误差", "#7c3aed"),
        ],
        peak_key=None,
        min_key="abs_count_error",
        min_label="最小计数误差",
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
        "count_diff",
        "abs_count_error",
        "defect_gt_boxes",
        "defect_pred_boxes",
        "defect_detected_boxes",
        "defect_missed_boxes",
        "defect_detect_rate",
        "defect_miss_rate",
        "normal_gt_boxes",
        "normal_hit_by_defect_boxes",
        "normal_to_defect_rate",
    ]
    for class_name in TARGET_CLASSES:
        fieldnames.extend(
            [
                f"{class_name}_gt_boxes",
                f"{class_name}_pred_boxes",
                f"{class_name}_tp",
                f"{class_name}_fp",
                f"{class_name}_fn",
                f"{class_name}_precision",
                f"{class_name}_recall",
                f"{class_name}_f1",
            ]
        )
    with (batch_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def write_batch_json(batch_dir: Path, payload: dict) -> None:
    (batch_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_count_example_cards(
    batch_dir: Path,
    items: list[dict],
    empty_text: str,
    class_names: dict[str, str],
    conf_threshold: object,
    labelimg_exe: Path | None,
    class_file_path: Path | None,
) -> str:
    if not items:
        return f'<div class="empty-hint">{html.escape(empty_text)}</div>'

    cards: list[str] = []
    editor_dir = batch_dir / "_gt_editors" / f"conf_{format_threshold_suffix(float(conf_threshold))}"
    for item in items:
        original_path = Path(str(item["image_path"]))
        predict_path = Path(str(item["predict_image_path"]))
        editor_href = write_gt_editor_html(editor_dir, batch_dir, conf_threshold, item, class_names)
        labelimg_href = write_labelimg_launcher(editor_dir, batch_dir, item, labelimg_exe, class_file_path)
        inspector_predict_dir = predict_path.parent
        inspector_href = f"http://127.0.0.1:{int(item.get('labelimg_server_port', 8765))}/open_inspector?" + urlencode(
            {
                "predict_dir": str(inspector_predict_dir),
                "image_name": predict_path.name,
            },
            quote_via=quote,
        )
        original_panel = build_overlay_panel_html(
            image_href=path_to_href(batch_dir, original_path),
            title="Original + GT",
            detections=item.get("gt_detections", []),
            stroke_color="#2563eb",
            label_prefix="GT",
        )
        predict_panel = build_overlay_panel_html(
            image_href=path_to_href(batch_dir, original_path),
            title=f"Predict @ conf>={format_threshold_text(float(conf_threshold))}",
            detections=item.get("pred_detections", []),
            stroke_color="#ea580c",
            label_prefix="Pred",
            show_confidence=True,
        )
        cards.append(
            "\n".join(
                [
                    '<article class="example-card">',
                    '  <div class="example-images">',
                    original_panel,
                    predict_panel,
                    "  </div>",
                    '  <div class="example-meta">',
                    f'    <div class="example-title">{html.escape(str(item["stem"]))}</div>',
                    f'    <div class="example-line">GT={item["gt_count"]} | Pred={item["pred_count"]} | diff={item["count_diff"]:+d} | abs={item["abs_count_error"]}</div>',
                    f'    <div class="example-line">TP={item["tp_count"]} | FP={item["fp_count"]} | FN={item["fn_count"]}</div>',
                    f'    <div class="example-line">缺陷漏检率={float(item["defect_miss_rate"]):.4f} | normal误剔率={float(item["normal_to_defect_rate"]):.4f}</div>',
                    (
                        f'    <div class="example-line"><button type="button" class="action-link" onclick="return triggerLocalAction(\'{html.escape(labelimg_href, quote=True)}\')">打开 LabelImg</button></div>'
                        if labelimg_href
                        else '    <div class="example-line">未找到 LabelImg 启动入口</div>'
                    ),
                    f'    <div class="example-line"><button type="button" class="action-link" onclick="return triggerLocalAction(\'{html.escape(inspector_href, quote=True)}\')">打开参数 GUI</button></div>',
                    f'    <div class="example-line"><a href="{editor_href}" target="_blank" rel="noopener noreferrer">打开轻量 GT 编辑器</a></div>',
                    "  </div>",
                    "</article>",
                ]
            )
        )
    return "\n".join(cards)


def build_count_diagnostic_sections(
    batch_dir: Path,
    rows: list[dict],
    class_names: dict[str, str],
    labelimg_exe: Path | None,
    class_file_path: Path | None,
) -> str:
    sections: list[str] = []
    for row in rows:
        diagnostics = row.get("count_diagnostics") or {}
        overcounted = diagnostics.get("overcounted", [])
        undercounted = diagnostics.get("undercounted", [])
        largest_abs_errors = diagnostics.get("largest_abs_errors", [])
        section_id = f"diag_{format_threshold_suffix(float(row['conf_threshold']))}"
        sections.append(
            "\n".join(
                [
                    f'<details class="diag-section" id="{section_id}">',
                    f"  <summary>conf={html.escape(str(row['conf_threshold']))} | abs={row['abs_count_error']} | count_diff={row['count_diff']} | 点击展开计数样本</summary>",
                    '  <div class="diag-grid">',
                    '    <section class="diag-block">',
                    "      <h3>计多了</h3>",
                    "      <p>Pred 数量大于 GT 的样本，按绝对计数误差从大到小展示。</p>",
                    build_count_example_cards(
                        batch_dir,
                        overcounted,
                        "当前 conf 下没有计多样本。",
                        class_names,
                        row["conf_threshold"],
                        labelimg_exe,
                        class_file_path,
                    ),
                    "    </section>",
                    '    <section class="diag-block">',
                    "      <h3>计少了</h3>",
                    "      <p>Pred 数量小于 GT 的样本，按绝对计数误差从大到小展示。</p>",
                    build_count_example_cards(
                        batch_dir,
                        undercounted,
                        "当前 conf 下没有计少样本。",
                        class_names,
                        row["conf_threshold"],
                        labelimg_exe,
                        class_file_path,
                    ),
                    "    </section>",
                    '    <section class="diag-block diag-block-wide">',
                    "      <h3>绝对计数误差最大样本</h3>",
                    "      <p>不区分正负，方便快速定位最影响 abs 的样本。</p>",
                    build_count_example_cards(
                        batch_dir,
                        largest_abs_errors,
                        "当前 conf 下没有诊断样本。",
                        class_names,
                        row["conf_threshold"],
                        labelimg_exe,
                        class_file_path,
                    ),
                    "    </section>",
                    "  </div>",
                    "</details>",
                ]
            )
        )
    return "\n".join(sections)


def write_batch_html(batch_dir: Path, rows: list[dict], meta: dict) -> None:
    def comparison_lines(row: dict, highlight: str) -> str:
        labels = [
            ("abs_count_error", "绝对计数误差", False),
            ("defect_detect_rate", "缺陷筛出率", True),
            ("defect_miss_rate", "缺陷漏检率", True),
            ("normal_to_defect_rate", "正常品误剔率", True),
        ]
        parts = []
        for key, label, is_rate in labels:
            if key == highlight:
                continue
            value = row[key]
            text = f"{value:.4f}" if is_rate else str(value)
            parts.append(f"{label}={text}")
        return " | ".join(parts)

    predict_params = meta.get("predict_params", {})
    predict_param_text = " | ".join(
        f"{key}={predict_params[key]}"
        for key in ("conf", "iou", "edge_penalty", "edge_touch_px", "flat_ratio_threshold", "edge_penalty_factor")
        if key in predict_params
    )
    best_defect_detect_row = max(rows, key=lambda item: float(item["defect_detect_rate"]))
    best_defect_miss_row = min(rows, key=lambda item: float(item["defect_miss_rate"]))
    best_normal_to_defect_row = min(rows, key=lambda item: float(item["normal_to_defect_rate"]))
    min_abs_count_error_row = min(rows, key=lambda item: float(item["abs_count_error"]))
    max_abs_count_error_row = max(rows, key=lambda item: float(item["abs_count_error"]))
    max_defect_detect_row = max(rows, key=lambda item: float(item["defect_detect_rate"]))
    min_defect_detect_row = min(rows, key=lambda item: float(item["defect_detect_rate"]))
    max_defect_miss_row = max(rows, key=lambda item: float(item["defect_miss_rate"]))
    min_defect_miss_row = min(rows, key=lambda item: float(item["defect_miss_rate"]))
    max_normal_to_defect_row = max(rows, key=lambda item: float(item["normal_to_defect_rate"]))
    min_normal_to_defect_row = min(rows, key=lambda item: float(item["normal_to_defect_rate"]))
    class_cards: list[str] = []
    business_cards = [
        (
            f'<div class="card"><div class="label">绝对计数误差</div><div class="value">{min_abs_count_error_row["abs_count_error"]}</div>'
            f'<div class="sub">min @ conf={min_abs_count_error_row["conf_threshold"]} | max={max_abs_count_error_row["abs_count_error"]} @ conf={max_abs_count_error_row["conf_threshold"]}</div>'
            f'<div class="sub">同conf对比: {comparison_lines(min_abs_count_error_row, "abs_count_error")}</div></div>'
        ),
        (
            f'<div class="card"><div class="label">缺陷筛出率</div><div class="value">{best_defect_detect_row["defect_detect_rate"]:.4f}</div>'
            f'<div class="sub">max @ conf={max_defect_detect_row["conf_threshold"]} | min={min_defect_detect_row["defect_detect_rate"]:.4f} @ conf={min_defect_detect_row["conf_threshold"]}</div>'
            f'<div class="sub">同conf对比: {comparison_lines(best_defect_detect_row, "defect_detect_rate")}</div></div>'
        ),
        (
            f'<div class="card"><div class="label">缺陷漏检率</div><div class="value">{best_defect_miss_row["defect_miss_rate"]:.4f}</div>'
            f'<div class="sub">min @ conf={min_defect_miss_row["conf_threshold"]} | max={max_defect_miss_row["defect_miss_rate"]:.4f} @ conf={max_defect_miss_row["conf_threshold"]}</div>'
            f'<div class="sub">同conf对比: {comparison_lines(best_defect_miss_row, "defect_miss_rate")}</div></div>'
        ),
        (
            f'<div class="card"><div class="label">正常品误剔率</div><div class="value">{best_normal_to_defect_row["normal_to_defect_rate"]:.4f}</div>'
            f'<div class="sub">min @ conf={min_normal_to_defect_row["conf_threshold"]} | max={max_normal_to_defect_row["normal_to_defect_rate"]:.4f} @ conf={max_normal_to_defect_row["conf_threshold"]}</div>'
            f'<div class="sub">同conf对比: {comparison_lines(best_normal_to_defect_row, "normal_to_defect_rate")}</div></div>'
        ),
    ]
    charts: list[str] = [
        build_count_chart(
            title="整体计数误差变化",
            subtitle="比较不同 conf 下 abs(pred_count - gt_count) 的变化",
            rows=rows,
        ),
        build_business_chart(
            title="缺陷筛选业务指标变化",
            subtitle="重点关注缺陷筛出率、缺陷漏检率、以及正常品误剔率",
            rows=rows,
        ),
    ]
    for class_name in TARGET_CLASSES:
        best_row = max(rows, key=lambda item: float(item[f"{class_name}_f1"]))
        class_cards.append(
            f'<div class="card"><div class="label">{html.escape(class_name)} best F1</div><div class="value">{best_row[f"{class_name}_f1"]:.4f}</div><div class="sub">conf={best_row["conf_threshold"]}</div></div>'
        )
        charts.append(
            build_metric_chart(
                title=f"{class_name} 指标变化",
                subtitle="比较不同 conf 下该类别 Precision / Recall / F1 的变化",
                rows=rows,
                class_name=class_name,
            )
        )

    table_rows: list[str] = []
    for row in rows:
        diag_anchor = f"#diag_{format_threshold_suffix(float(row['conf_threshold']))}"
        cells = [
            f"<td>{html.escape(str(row['conf_threshold']))}</td>",
            f"<td>{row['abs_count_error']}</td>",
            f"<td>{row['defect_detect_rate']:.4f}</td>",
            f"<td>{row['defect_miss_rate']:.4f}</td>",
            f"<td>{row['normal_to_defect_rate']:.4f}</td>",
            f"<td>{row['precision']:.4f}</td>",
            f"<td>{row['recall']:.4f}</td>",
            f"<td>{row['f1']:.4f}</td>",
        ]
        for class_name in TARGET_CLASSES:
            cells.extend(
                [
                    f"<td>{row[f'{class_name}_precision']:.4f}</td>",
                    f"<td>{row[f'{class_name}_recall']:.4f}</td>",
                    f"<td>{row[f'{class_name}_f1']:.4f}</td>",
                ]
            )
        cells.append(f'<td><a href="{diag_anchor}">查看样本</a></td>')
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    diagnostic_sections = build_count_diagnostic_sections(
        batch_dir,
        rows,
        meta.get("class_names", {}),
        Path(str(meta["labelimg_exe"])) if meta.get("labelimg_exe") else None,
        Path(str(meta["class_file_path"])) if meta.get("class_file_path") else None,
    )

    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            "  <title>Youge Per-Class Confidence Sweep</title>",
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
            "    .point-max { paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; }",
            "    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; overflow: hidden; }",
            "    th, td { padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: center; font-size: 13px; }",
            "    th { background: #eaf2fb; font-weight: 700; }",
            "    .diag-root { margin-top: 22px; display: flex; flex-direction: column; gap: 16px; }",
            "    .diag-root h2 { margin: 0; font-size: 22px; }",
            "    .diag-intro { margin: 0; color: #475569; font-size: 14px; }",
            "    .diag-section { background: #fff; border: 1px solid #dbe3ec; border-radius: 16px; padding: 16px; }",
            "    .diag-section > summary { cursor: pointer; font-size: 16px; font-weight: 700; color: #0f172a; }",
            "    .diag-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }",
            "    .diag-block { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px; }",
            "    .diag-block-wide { grid-column: 1 / -1; }",
            "    .diag-block h3 { margin: 0 0 6px; font-size: 18px; }",
            "    .diag-block p { margin: 0 0 12px; color: #475569; font-size: 13px; }",
            "    .example-card { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(240px, 0.7fr); gap: 14px; padding: 12px; background: #fff; border: 1px solid #dbe3ec; border-radius: 12px; margin-bottom: 12px; }",
            "    .example-images { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }",
            "    .example-panel { overflow: hidden; border-radius: 10px; border: 1px solid #e2e8f0; background: #fff; }",
            "    .example-panel-title { padding: 8px 10px; font-size: 13px; font-weight: 700; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }",
            "    .example-panel img { display: block; width: 100%; height: auto; }",
            "    .overlay-stage { position: relative; }",
            "    .overlay-svg { position: absolute; inset: 0; width: 100%; height: 100%; }",
            "    .overlay-box { fill: rgba(37, 99, 235, 0.08); stroke-width: 0.0036; }",
            "    .overlay-label-bg { rx: 0.006; ry: 0.006; opacity: 0.96; }",
            "    .overlay-label { fill: #ffffff; font-size: 0.022px; font-weight: 800; letter-spacing: 0.0005px; dominant-baseline: middle; }",
            "    .example-meta { display: flex; flex-direction: column; gap: 8px; justify-content: center; text-align: left; }",
            "    .example-title { font-size: 16px; font-weight: 700; word-break: break-all; }",
            "    .example-line { color: #334155; font-size: 13px; }",
            "    .action-link { background: none; border: 0; padding: 0; color: #2563eb; text-decoration: underline; cursor: pointer; font: inherit; }",
            "    .action-link:hover { color: #1d4ed8; }",
            "    .empty-hint { padding: 12px; background: #fff; border: 1px dashed #cbd5e1; border-radius: 12px; color: #64748b; font-size: 13px; }",
            "    @media (max-width: 1200px) { body { padding: 14px; } .chart-header { flex-direction: column; } .legend { justify-content: flex-start; } table { display: block; overflow-x: auto; white-space: nowrap; } .diag-grid { grid-template-columns: 1fr; } .example-card { grid-template-columns: 1fr; } }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Youge Per-Class Confidence Sweep Report</h1>",
            f'  <p class="topline">species={html.escape(str(meta["species"]))} | split={html.escape(str(meta["split"]))} | predict_name={html.escape(str(meta["predict_name"]))} | iou={meta["iou_threshold"]} | conf_count={len(rows)}</p>',
            (
                f'  <p class="topline">model_version={html.escape(str(meta.get("model_version", "")))}'
                + (f" | {html.escape(predict_param_text)}" if predict_param_text else "")
                + "</p>"
            ),
            '  <p class="topline">说明: 右侧 Predict 面板按当前 conf threshold 直接从 labels 重绘，只显示真正参与本次统计的预测框。</p>',
            '  <p class="topline">如需点击打开 LabelImg 或参数 GUI，请先双击报告目录下的 <code>start_labelimg_launcher.cmd</code> 启动本地服务，然后再点击样本卡片里的对应按钮。</p>',
            '  <section class="cards">',
            *business_cards,
            "  </section>",
            '  <section class="cards">',
            *class_cards,
            "  </section>",
            '  <section class="chart-grid">',
            *charts,
            "  </section>",
            "  <table>",
            "    <thead>",
            "      <tr>",
            "        <th>conf</th>",
            "        <th>绝对计数误差</th>",
            "        <th>缺陷检出率</th>",
            "        <th>缺陷漏检率</th>",
            "        <th>normal误检缺陷率</th>",
            "        <th>Overall P</th>",
            "        <th>Overall R</th>",
            "        <th>Overall F1</th>",
            "        <th>broken P</th>",
            "        <th>broken R</th>",
            "        <th>broken F1</th>",
            "        <th>normal P</th>",
            "        <th>normal R</th>",
            "        <th>normal F1</th>",
            "        <th>muddy P</th>",
            "        <th>muddy R</th>",
            "        <th>muddy F1</th>",
            "        <th>empty P</th>",
            "        <th>empty R</th>",
            "        <th>empty F1</th>",
            "        <th>计数诊断</th>",
            "      </tr>",
            "    </thead>",
            "    <tbody>",
            *table_rows,
            "    </tbody>",
            "  </table>",
            '  <section class="diag-root">',
            "    <h2>计数误差样本定位</h2>",
            '    <p class="diag-intro">展开某个 conf 后，可以直接看到哪些图计多了、哪些图计少了，以及它们的 Original / Predict 对照。右侧只显示当前 conf threshold 下真正参与统计的预测框。</p>',
            diagnostic_sections,
            "  </section>",
            "  <script>",
            "    async function triggerLocalAction(url) {",
            "      try {",
            "        await fetch(url, { method: 'GET', mode: 'no-cors', cache: 'no-store' });",
            "      } catch (error) {",
            "        console.error('Failed to trigger local action:', error);",
            "      }",
            "      return false;",
            "    }",
            "  </script>",
            "</body>",
            "</html>",
        ]
    )
    (batch_dir / "index.html").write_text(html_text, encoding="utf-8")


def write_open_report_cmd(version_root_dir: Path, batch_name: str) -> None:
    cmd_text = "\n".join(
        [
            "@echo off",
            "setlocal",
            f'start "" "%~dp0{batch_name}\\index.html"',
            "endlocal",
            "",
        ]
    )
    (version_root_dir / "open_report.cmd").write_text(cmd_text, encoding="utf-8")


def write_labelimg_server_cmd(batch_dir: Path, python_exe: Path, server_script: Path, port: int) -> None:
    cmd_text = "\n".join(
        [
            "@echo off",
            "setlocal",
            f'start "LabelImg Launcher" "{python_exe}" "{server_script}" --port {port}',
            "endlocal",
            "",
        ]
    )
    (batch_dir / "start_labelimg_launcher.cmd").write_text(cmd_text, encoding="utf-8")


def build_compare_chart(
    title: str,
    subtitle: str,
    versions_payload: list[dict],
    key: str,
    label: str,
    lower_is_better: bool = False,
) -> str:
    plot_width = 760
    plot_height = 240
    left = 72
    right = 48
    top = 54
    bottom = 64
    width = plot_width + left + right
    height = plot_height + top + bottom
    confs = sorted({float(row["conf_threshold"]) for payload in versions_payload for row in payload["rows"]})
    min_conf = min(confs)
    max_conf = max(confs)
    if max_conf == min_conf:
        max_conf = min_conf + 1.0

    values = [float(row[key]) for payload in versions_payload for row in payload["rows"]]
    y_min = min(values)
    y_max = max(values)
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
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis-label">{tick_value:.4f}</text>',
            ]
        )

    x_ticks: list[str] = []
    for conf in confs:
        x = x_pos(conf)
        x_ticks.extend(
            [
                f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" class="tick-line"></line>',
                f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" class="axis-label">{conf:g}</text>',
            ]
        )

    legend_items: list[str] = []
    series_html: list[str] = []
    for idx, payload in enumerate(versions_payload):
        version = str(payload["version"])
        rows = payload["rows"]
        color = COMPARE_COLORS[idx % len(COMPARE_COLORS)]
        points: list[str] = []
        point_marks: list[str] = []
        data = [float(row[key]) for row in rows]
        best_value = min(data) if lower_is_better else max(data)
        best_rows = [row for row in rows if float(row[key]) == best_value]
        for row in rows:
            value = float(row[key])
            conf = float(row["conf_threshold"])
            x = x_pos(conf)
            y = y_pos(value)
            points.append(f"{x:.2f},{y:.2f}")
            point_marks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"></circle>')
        best_row = (
            min(best_rows, key=lambda item: float(item["conf_threshold"]))
            if lower_is_better
            else min(best_rows, key=lambda item: float(item["conf_threshold"]))
        )
        bx = x_pos(float(best_row["conf_threshold"]))
        by = y_pos(float(best_row[key]))
        label_y = max(by - 18, 18)
        point_marks.extend(
            [
                f'<circle cx="{bx:.2f}" cy="{by:.2f}" r="8" class="peak-ring" stroke="{color}"></circle>',
                (
                    f'<text x="{bx:.2f}" y="{label_y:.2f}" text-anchor="middle" class="point-label point-max" '
                    f'fill="{color}">{html.escape(version)} {label}={float(best_row[key]):.4f} @ {float(best_row["conf_threshold"]):g}</text>'
                ),
            ]
        )
        series_html.append(
            "\n".join(
                [
                    f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>',
                    *point_marks,
                ]
            )
        )
        legend_items.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{color};"></span>{html.escape(version)}</div>'
        )

    return "\n".join(
        [
            '<section class="chart-card">',
            f'  <div class="chart-header"><div><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div><div class="legend">{"".join(legend_items)}</div></div>',
            f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            *grid_lines,
            f'    <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis-line"></line>',
            f'    <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis-line"></line>',
            *x_ticks,
            *series_html,
            "  </svg>",
            "</section>",
        ]
    )


def write_version_compare_report(compare_dir: Path, versions_payload: list[dict], conf_thresholds: list[float]) -> None:
    compare_dir.mkdir(parents=True, exist_ok=True)
    versions_payload = sorted(versions_payload, key=lambda item: version_sort_key(str(item["version"])))
    payload = {
        "versions": [
            {
                "version": item["version"],
                "predict_name": item["meta"].get("predict_name"),
                "predict_params": item["meta"].get("predict_params", {}),
                "rows": item["rows"],
            }
            for item in versions_payload
        ],
        "conf_thresholds": conf_thresholds,
    }
    write_batch_json(compare_dir, payload)

    fieldnames = [
        "version",
        "conf_threshold",
        "abs_count_error",
        "defect_detect_rate",
        "defect_miss_rate",
        "normal_to_defect_rate",
        "precision",
        "recall",
        "f1",
    ]
    with (compare_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in versions_payload:
            for row in item["rows"]:
                writer.writerow(
                    {
                        "version": item["version"],
                        "conf_threshold": row["conf_threshold"],
                        "abs_count_error": row["abs_count_error"],
                        "defect_detect_rate": row["defect_detect_rate"],
                        "defect_miss_rate": row["defect_miss_rate"],
                        "normal_to_defect_rate": row["normal_to_defect_rate"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                    }
                )

    cards: list[str] = []
    for item in versions_payload:
        rows = item["rows"]
        best_detect = max(rows, key=lambda row: float(row["defect_detect_rate"]))
        best_count = min(rows, key=lambda row: float(row["abs_count_error"]))
        best_normal = min(rows, key=lambda row: float(row["normal_to_defect_rate"]))
        cards.append(
            "".join(
                [
                    '<div class="card">',
                    f'<div class="label">{html.escape(str(item["version"]))}</div>',
                    f'<div class="sub">best detect: conf={best_detect["conf_threshold"]} | {best_detect["defect_detect_rate"]:.4f}</div>',
                    f'<div class="sub">best count: conf={best_count["conf_threshold"]} | abs={best_count["abs_count_error"]}</div>',
                    f'<div class="sub">best normal: conf={best_normal["conf_threshold"]} | {best_normal["normal_to_defect_rate"]:.4f}</div>',
                    "</div>",
                ]
            )
        )

    charts = [
        build_compare_chart(
            "多版本整体计数误差对比",
            "同一组 conf threshold 下比较各版本绝对计数误差",
            versions_payload,
            "abs_count_error",
            "abs",
            lower_is_better=True,
        ),
        build_compare_chart(
            "多版本缺陷筛出率对比",
            "同一组 conf threshold 下比较各版本缺陷筛出率",
            versions_payload,
            "defect_detect_rate",
            "detect",
        ),
        build_compare_chart(
            "多版本缺陷漏检率对比",
            "同一组 conf threshold 下比较各版本缺陷漏检率",
            versions_payload,
            "defect_miss_rate",
            "miss",
            lower_is_better=True,
        ),
        build_compare_chart(
            "多版本正常品误剔率对比",
            "同一组 conf threshold 下比较各版本 normal 误剔率",
            versions_payload,
            "normal_to_defect_rate",
            "normal",
            lower_is_better=True,
        ),
        build_compare_chart(
            "多版本 Overall F1 对比", "同一组 conf threshold 下比较各版本整体 F1", versions_payload, "f1", "f1"
        ),
    ]

    table_rows: list[str] = []
    for item in versions_payload:
        for row in item["rows"]:
            table_rows.append(
                "<tr>"
                f"<td>{html.escape(str(item['version']))}</td>"
                f"<td>{row['conf_threshold']}</td>"
                f"<td>{row['abs_count_error']}</td>"
                f"<td>{row['defect_detect_rate']:.4f}</td>"
                f"<td>{row['defect_miss_rate']:.4f}</td>"
                f"<td>{row['normal_to_defect_rate']:.4f}</td>"
                f"<td>{row['f1']:.4f}</td>"
                "</tr>"
            )

    version_text = " vs ".join(str(item["version"]) for item in versions_payload)
    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            "  <title>Youge Version Compare</title>",
            "  <style>",
            "    body { margin: 0; padding: 24px; font-family: 'Segoe UI', sans-serif; background: #f3f6fa; color: #0f172a; }",
            "    h1 { margin: 0 0 10px; font-size: 28px; }",
            "    .topline { margin: 0 0 18px; color: #334155; font-size: 14px; }",
            "    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-bottom: 20px; }",
            "    .card { background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; padding: 16px; }",
            "    .label { color: #475569; font-size: 13px; font-weight: 700; }",
            "    .sub { margin-top: 8px; color: #475569; font-size: 13px; }",
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
            "    .point-max { paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; }",
            "    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe3ec; border-radius: 14px; overflow: hidden; }",
            "    th, td { padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: center; font-size: 13px; }",
            "    th { background: #eaf2fb; font-weight: 700; }",
            "    @media (max-width: 1200px) { body { padding: 14px; } .chart-header { flex-direction: column; } .legend { justify-content: flex-start; } table { display: block; overflow-x: auto; white-space: nowrap; } }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Youge Version Compare Report</h1>",
            f'  <p class="topline">versions={html.escape(version_text)} | conf_range={html.escape(", ".join(format_threshold_text(value) for value in conf_thresholds))}</p>',
            '  <section class="cards">',
            *cards,
            "  </section>",
            '  <section class="chart-grid">',
            *charts,
            "  </section>",
            "  <table>",
            "    <thead>",
            "      <tr>",
            "        <th>version</th>",
            "        <th>conf</th>",
            "        <th>绝对计数误差</th>",
            "        <th>缺陷检出率</th>",
            "        <th>缺陷漏检率</th>",
            "        <th>normal误检缺陷率</th>",
            "        <th>Overall F1</th>",
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
    (compare_dir / "index.html").write_text(html_text, encoding="utf-8")


def run_single_version(
    *,
    args: argparse.Namespace,
    serial_config: dict,
    version: str,
    postprocess_version: str | None,
    conf_thresholds: list[float],
    project_dir: Path,
    repo_root: Path,
    script_dir: Path,
    eval_script: Path,
    build_versioned_name,
    build_predict_run_name,
) -> dict:
    species = args.species or serial_config.get("species") or "youge"
    predict_name_base = args.predict_name or serial_config.get("predict_name") or "youge_predict"
    name_prefix_base = args.name_prefix or serial_config.get("name") or "youge_opt_class_report"
    target_label = build_eval_target_label(version, postprocess_version)
    predict_name = build_predict_run_name(predict_name_base, version, postprocess_version)
    predict_run_dir = repo_root / "src" / "predict" / species / "runs" / "predict" / predict_name
    predict_run_config = load_predict_run_config(predict_run_dir)
    labelimg_exe = resolve_labelimg_exe()
    class_file_path = repo_root / "datasets" / "data" / species / "classes.txt"
    labelimg_server_port = 8765
    predict_params = {
        "conf": predict_run_config.get("conf"),
        "iou": predict_run_config.get("iou"),
        "edge_penalty": predict_run_config.get("edge_penalty"),
        "edge_touch_px": predict_run_config.get("edge_touch_px"),
        "flat_ratio_threshold": predict_run_config.get("flat_ratio_threshold"),
        "edge_penalty_factor": predict_run_config.get("edge_penalty_factor"),
    }
    predict_params = {key: value for key, value in predict_params.items() if value is not None}
    version_root_name = build_predict_run_name(str(name_prefix_base).rstrip("_-"), version, postprocess_version)
    version_root_dir = project_dir / version_root_name
    if version_root_dir.exists():
        remove_tree_with_retry(version_root_dir)
    version_root_dir.mkdir(parents=True, exist_ok=True)

    base_args = argparse.Namespace(**vars(args))
    base_args.version = version
    base_args.postprocess_version = postprocess_version
    base_command = build_base_command(base_args, eval_script)
    batch_name = (args.batch_name or "conf_summary").rstrip("_-")
    batch_dir = version_root_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    write_labelimg_server_cmd(
        batch_dir, Path(sys.executable), script_dir / "labelimg_launcher_server.py", labelimg_server_port
    )
    summary_cache_dir = batch_dir / "_summaries"
    summary_cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for index, conf_threshold in enumerate(conf_thresholds, start=1):
        run_name = f"conf_{format_threshold_suffix(conf_threshold)}"
        summary_path = summary_cache_dir / f"{run_name}.json"
        command = [
            *base_command,
            "--summary-only",
            "--project",
            str(version_root_dir),
            "--conf-threshold",
            str(conf_threshold),
            "--name",
            run_name,
            "--summary-output",
            str(summary_path),
        ]
        print(f"[{target_label}][{index}/{len(conf_thresholds)}] Running: {' '.join(command)}")
        subprocess.run(command, check=True, cwd=script_dir)

        summary_payload = load_summary(summary_path)
        summary = summary_payload["summary"]
        business = summary.get("business", {})
        row = {
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
            "count_diff": business.get("count_diff", 0),
            "abs_count_error": business.get("abs_count_error", 0),
            "defect_gt_boxes": business.get("defect_gt_boxes", 0),
            "defect_pred_boxes": business.get("defect_pred_boxes", 0),
            "defect_detected_boxes": business.get("defect_detected_boxes", 0),
            "defect_missed_boxes": business.get("defect_missed_boxes", 0),
            "defect_detect_rate": business.get("defect_detect_rate", 0.0),
            "defect_miss_rate": business.get("defect_miss_rate", 0.0),
            "normal_gt_boxes": business.get("normal_gt_boxes", 0),
            "normal_hit_by_defect_boxes": business.get("normal_hit_by_defect_boxes", 0),
            "normal_to_defect_rate": business.get("normal_to_defect_rate", 0.0),
            "count_diagnostics": summary.get("count_diagnostics", {}),
        }
        for class_name in TARGET_CLASSES:
            metrics = summary["per_class"].get(class_name, {})
            for key in ("gt_boxes", "pred_boxes", "tp", "fp", "fn", "precision", "recall", "f1"):
                row[f"{class_name}_{key}"] = metrics.get(key, 0)
        rows.append(row)

    rows.sort(key=lambda item: float(item["conf_threshold"]))
    meta = {
        "species": species,
        "split": args.split or serial_config.get("split") or "both",
        "predict_name": predict_name,
        "version": target_label,
        "model_version": version,
        "postprocess_version": postprocess_version,
        "predict_params": predict_params,
        "iou_threshold": args.iou_threshold
        if args.iou_threshold is not None
        else serial_config.get("iou_threshold", 0.5),
        "class_names": summary_payload.get("class_names", summary.get("class_names", {})),
        "labelimg_exe": str(labelimg_exe) if labelimg_exe else None,
        "class_file_path": str(class_file_path) if class_file_path.exists() else None,
        "labelimg_server_port": labelimg_server_port,
    }
    write_batch_csv(batch_dir, rows)
    write_batch_json(batch_dir, {"meta": meta, "runs": rows})
    write_batch_html(batch_dir, rows, meta)
    write_open_report_cmd(version_root_dir, batch_name)
    print(f"Per-class batch summary report generated at: {batch_dir}")
    return {
        "version": target_label,
        "model_version": version,
        "postprocess_version": postprocess_version,
        "rows": rows,
        "meta": meta,
        "batch_dir": batch_dir,
    }


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.youge_versioning import (
        build_predict_run_name,
        build_versioned_name,
        extract_postprocess_version_from_text,
        extract_version_from_text,
        normalize_postprocess_version,
        normalize_version,
    )

    config_path = Path(args.config).resolve() if args.config else script_dir / "opt_class_eval.json"
    serial_config = load_eval_config(config_path)
    eval_script = script_dir / "opt_class_eval.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")

    if args.project is None and serial_config.get("project") is not None:
        args.project = str(serial_config["project"])
    project_dir = resolve_project_dir(args, script_dir)
    config_conf_tokens = serial_config.get("conf_thresholds") or ["0.70-0.90:0.01"]
    if isinstance(config_conf_tokens, str):
        config_conf_tokens = [config_conf_tokens]
    conf_thresholds = unique_conf_thresholds(
        args.conf_thresholds or parse_conf_thresholds([str(token) for token in config_conf_tokens])
    )
    requested_versions = list(args.versions or [])
    requested_versions.extend(ensure_list(args.version))
    requested_versions.extend(ensure_list(serial_config.get("versions")))
    if not requested_versions:
        requested_versions.extend(ensure_list(serial_config.get("version")))

    normalized_versions: list[str] = []
    for value in requested_versions:
        normalized = normalize_version(value)
        if normalized and normalized not in normalized_versions:
            normalized_versions.append(normalized)

    if not normalized_versions:
        inferred_version = (
            extract_version_from_text(args.name_prefix)
            or extract_version_from_text(args.batch_name)
            or extract_version_from_text(args.predict_name or serial_config.get("predict_name"))
            or extract_version_from_text(serial_config.get("name"))
        )
        if inferred_version is None:
            species = args.species or serial_config.get("species") or "youge"
            predict_runs_dir = repo_root / "src" / "predict" / species / "runs" / "predict"
            latest_predict_name = find_latest_versioned_dir_name(
                predict_runs_dir, args.predict_name or serial_config.get("predict_name") or "youge_predict"
            )
            inferred_version = extract_version_from_text(latest_predict_name)
        if inferred_version is None:
            raise ValueError("No version could be resolved. Please set version or versions in config.")
        normalized_versions.append(normalize_version(inferred_version))

    requested_postprocess_versions = list(getattr(args, "postprocess_versions", None) or [])
    requested_postprocess_versions.extend(ensure_list(serial_config.get("postprocess_versions")))
    requested_postprocess_versions.extend(ensure_list(args.postprocess_version))
    if not requested_postprocess_versions:
        requested_postprocess_versions.extend(ensure_list(serial_config.get("postprocess_version")))

    normalized_postprocess_versions: list[str | None] = []
    for value in requested_postprocess_versions:
        normalized = normalize_postprocess_version(value)
        if normalized and normalized not in normalized_postprocess_versions:
            normalized_postprocess_versions.append(normalized)

    if not normalized_postprocess_versions:
        inferred_postprocess_version = (
            extract_postprocess_version_from_text(args.name_prefix)
            or extract_postprocess_version_from_text(args.batch_name)
            or extract_postprocess_version_from_text(args.predict_name or serial_config.get("predict_name"))
            or extract_postprocess_version_from_text(serial_config.get("name"))
        )
        if inferred_postprocess_version is not None:
            normalized_postprocess_versions.append(normalize_postprocess_version(inferred_postprocess_version))

    if not normalized_postprocess_versions:
        if len(normalized_versions) == 1:
            species = args.species or serial_config.get("species") or "youge"
            predict_runs_dir = repo_root / "src" / "predict" / species / "runs" / "predict"
            latest_predict_name = find_latest_versioned_dir_name(
                predict_runs_dir, args.predict_name or serial_config.get("predict_name") or "youge_predict"
            )
            inferred_postprocess_version = extract_postprocess_version_from_text(latest_predict_name)
            if inferred_postprocess_version is not None:
                normalized_postprocess_versions.append(normalize_postprocess_version(inferred_postprocess_version))

    if not normalized_postprocess_versions:
        normalized_postprocess_versions.append(None)

    eval_targets = [
        (version, postprocess_version)
        for version in sorted(normalized_versions, key=version_sort_key)
        for postprocess_version in normalized_postprocess_versions
    ]

    version_payloads = [
        run_single_version(
            args=args,
            serial_config=serial_config,
            version=version,
            postprocess_version=postprocess_version,
            conf_thresholds=conf_thresholds,
            project_dir=project_dir,
            repo_root=repo_root,
            script_dir=script_dir,
            eval_script=eval_script,
            build_versioned_name=build_versioned_name,
            build_predict_run_name=build_predict_run_name,
        )
        for version, postprocess_version in eval_targets
    ]

    if len(version_payloads) >= 2:
        compare_root_dir = project_dir / "version-compare"
        compare_name = "_vs_".join(str(item["version"]) for item in version_payloads)
        if args.postprocess_version:
            compare_name = f"{compare_name}_{args.postprocess_version}"
        compare_dir = compare_root_dir / compare_name
        if compare_dir.exists():
            remove_tree_with_retry(compare_dir)
        write_version_compare_report(compare_dir, version_payloads, conf_thresholds)
        print(f"Version compare report generated at: {compare_dir}")


if __name__ == "__main__":
    main()
