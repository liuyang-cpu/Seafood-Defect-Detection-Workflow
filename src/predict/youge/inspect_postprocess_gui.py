from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
except Exception:  # noqa: BLE001
    cv2 = None
    np = None

try:
    from PIL import Image, ImageTk
except Exception:  # noqa: BLE001
    Image = None
    ImageTk = None


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect predict postprocess parameters in a local GUI.")
    parser.add_argument("--predict-dir", type=str, default=None, help="Predict output directory containing phase2_postprocess_summary.json.")
    parser.add_argument("--image-name", type=str, default=None, help="Optional image name to preselect after loading the predict dir.")
    return parser.parse_args()


def format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(format_value(item) for item in value)
    return str(value)


def detect_nearest_vertical_side(top_gap: object, bottom_gap: object) -> str:
    if top_gap is None or bottom_gap is None:
        return "unknown"
    top_gap_value = float(top_gap)
    bottom_gap_value = float(bottom_gap)
    if abs(top_gap_value - bottom_gap_value) < 1e-6:
        return "top_bottom"
    return "top" if top_gap_value < bottom_gap_value else "bottom"


def compute_vertical_gaps(
    *,
    y1: float,
    y2: float,
    image_h: int | None,
    top_gap: object,
    bottom_gap: object,
) -> tuple[float | None, float | None]:
    resolved_top_gap = None if top_gap is None else float(top_gap)
    resolved_bottom_gap = None if bottom_gap is None else float(bottom_gap)
    if image_h is None:
        return resolved_top_gap, resolved_bottom_gap
    if resolved_top_gap is None:
        resolved_top_gap = float(y1)
    if resolved_bottom_gap is None:
        resolved_bottom_gap = float(image_h) - float(y2)
    return resolved_top_gap, resolved_bottom_gap


def detect_vertical_edge_type(y1: float, y2: float, image_h: int, margin_px: float) -> str | None:
    touches_top = y1 <= float(margin_px)
    touches_bottom = y2 >= float(image_h - margin_px)
    if touches_top and touches_bottom:
        return "top_bottom"
    if touches_top:
        return "top"
    if touches_bottom:
        return "bottom"
    return None


def longest_contiguous_foreground_ratio(edge_row: np.ndarray) -> float:
    if edge_row.size == 0:
        return 0.0
    width = float(edge_row.shape[0])
    if width <= 0:
        return 0.0

    best_run = 0
    current_run = 0
    for value in edge_row:
        if int(value) != 0:
            current_run += 1
            best_run = max(best_run, current_run)
        else:
            current_run = 0
    return float(best_run) / width


def compute_edge_span_ratio_from_array(crop: np.ndarray | None, edge_type: str) -> float | None:
    if cv2 is None or np is None or crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, 8)
    if num <= 1:
        return 0.0
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == index).astype("uint8")
    width = float(mask.shape[1])
    if width <= 0:
        return 0.0
    top_ratio = longest_contiguous_foreground_ratio(mask[0])
    bottom_ratio = longest_contiguous_foreground_ratio(mask[-1])
    if edge_type == "top":
        return top_ratio
    if edge_type == "bottom":
        return bottom_ratio
    return max(top_ratio, bottom_ratio)


def build_rule_summary(row: dict) -> list[str]:
    applied_rules = list(row.get("applied_rules") or [])
    phase2_decision = row.get("phase2_decision")
    phase2_reason = row.get("phase2_reason")

    lines = []
    display_rules = []
    if "horizontal_edge_penalty" in applied_rules:
        display_rules.append("horizontal_edge_penalty")
    if "vertical_defective_penalty" in applied_rules:
        display_rules.append("vertical_defective_penalty")
    if "adjacent_frame_dedup_keep" in applied_rules:
        display_rules.append("adjacent_frame_dedup_keep")
    if "adjacent_frame_dedup_drop" in applied_rules:
        display_rules.append("adjacent_frame_dedup_drop")

    lines.append(f"命中规则: {', '.join(display_rules) if display_rules else 'none'}")
    if "horizontal_edge_penalty" in applied_rules:
        lines.append("判定路径: horizontal 命中并降分，vertical 不再参与降分")
    elif "vertical_defective_penalty" in applied_rules:
        lines.append("判定路径: horizontal 未命中，vertical 判为 defective 并降分")
    elif "adjacent_frame_dedup_keep" in applied_rules:
        lines.append("判定路径: 命中相邻帧去重，当前框被保留")
    elif "adjacent_frame_dedup_drop" in applied_rules:
        lines.append("判定路径: 命中相邻帧去重，当前框被删除")
    elif phase2_decision:
        lines.append(f"判定结果: vertical={phase2_decision} ({phase2_reason})")
    else:
        lines.append("判定路径: horizontal/vertical 都未命中")
    return lines


def color_for_row(applied_rules: list[str], phase2_decision: str | None) -> str:
    if "adjacent_frame_dedup_drop" in applied_rules:
        return "#6b7280"
    if "horizontal_edge_penalty" in applied_rules:
        return "#ef4444"
    if "vertical_defective_penalty" in applied_rules:
        return "#f97316"
    if phase2_decision == "intact":
        return "#16a34a"
    return "#2563eb"


class PostprocessInspectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Youge Postprocess Inspector")
        self.root.geometry("1600x920")

        self.predict_dir: Path | None = None
        self.summary_rows: list[dict] = []
        self.rows_by_image: dict[str, list[dict]] = {}
        self.image_names: list[str] = []
        self.current_image_name: str | None = None
        self.current_rows: list[dict] = []
        self.current_selected_det_index: int | None = None
        self.tk_image: tk.PhotoImage | None = None
        self.original_image: tk.PhotoImage | None = None
        self.original_pil_image = None
        self.source_image_paths: dict[str, Path] = {}
        self.current_source_pil_image = None
        self.scale_ratio = 1.0
        self.canvas_items_by_det_index: dict[int, list[int]] = {}
        self.zoom_mode = tk.StringVar(value="fit")
        self.show_suppressed_var = tk.BooleanVar(value=False)
        self.rule_config: dict[str, object] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)

        ttk.Button(toolbar, text="选择 Predict 目录", command=self.choose_predict_dir).grid(row=0, column=0, sticky="w")
        self.dir_var = tk.StringVar(value="未选择目录")
        ttk.Label(toolbar, textvariable=self.dir_var).grid(row=0, column=1, padx=(12, 0), sticky="ew")
        ttk.Button(toolbar, text="适应窗口", command=lambda: self.set_zoom_mode("fit")).grid(row=0, column=2, padx=(12, 0))
        ttk.Button(toolbar, text="1:1", command=lambda: self.set_zoom_mode("actual")).grid(row=0, column=3, padx=(6, 0))
        ttk.Button(toolbar, text="放大", command=self.zoom_in).grid(row=0, column=4, padx=(6, 0))
        ttk.Button(toolbar, text="缩小", command=self.zoom_out).grid(row=0, column=5, padx=(6, 0))
        ttk.Checkbutton(
            toolbar,
            text="显示被去重删除的框",
            variable=self.show_suppressed_var,
            command=self._refresh_current_image,
        ).grid(row=0, column=6, padx=(12, 0), sticky="e")

        main_pane = ttk.Panedwindow(self.root, orient="horizontal")
        main_pane.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(main_pane, padding=(10, 0, 10, 10))
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="图片列表").grid(row=0, column=0, sticky="w")
        self.image_listbox = tk.Listbox(left, width=32, exportselection=False)
        self.image_listbox.grid(row=1, column=0, sticky="nsew")
        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_select)

        center = ttk.Frame(main_pane, padding=(0, 0, 10, 10))
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        vertical_pane = ttk.Panedwindow(center, orient="vertical")
        vertical_pane.grid(row=0, column=0, sticky="nsew")

        image_frame = ttk.Frame(vertical_pane)
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(image_frame, bg="#1f2937", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        x_scroll = ttk.Scrollbar(image_frame, orient="horizontal", command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(image_frame, orient="vertical", command=self.canvas.yview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        bottom_frame = ttk.Frame(vertical_pane)
        bottom_frame.columnconfigure(0, weight=0)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.rowconfigure(1, weight=1)

        det_frame = ttk.Frame(bottom_frame)
        det_frame.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 12))
        det_frame.rowconfigure(1, weight=1)
        ttk.Label(det_frame, text="框列表").grid(row=0, column=0, sticky="w")
        self.det_listbox = tk.Listbox(det_frame, width=48, height=10, exportselection=False)
        self.det_listbox.grid(row=1, column=0, sticky="nsw")
        self.det_listbox.bind("<<ListboxSelect>>", self._on_det_select)

        ttk.Label(bottom_frame, text="参数详情").grid(row=0, column=1, sticky="w")
        self.detail_text = tk.Text(bottom_frame, width=72, height=10, wrap="word")
        self.detail_text.grid(row=1, column=1, sticky="nsew")
        detail_scroll = ttk.Scrollbar(bottom_frame, orient="vertical", command=self.detail_text.yview)
        detail_scroll.grid(row=1, column=2, sticky="ns")
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        self.detail_text.configure(state="disabled")

        vertical_pane.add(image_frame, weight=5)
        vertical_pane.add(bottom_frame, weight=2)
        self.vertical_pane = vertical_pane

        main_pane.add(left, weight=0)
        main_pane.add(center, weight=1)
        self.main_pane = main_pane
        self.root.bind("<Configure>", self._on_window_resize)

    def set_zoom_mode(self, mode: str) -> None:
        self.zoom_mode.set(mode)
        if self.current_image_name is not None:
            self.show_image(self.current_image_name)

    def zoom_in(self) -> None:
        self.zoom_mode.set("manual")
        self.scale_ratio = min(self.scale_ratio * 1.25, 8.0)
        if self.current_image_name is not None:
            self.show_image(self.current_image_name, preserve_ratio=True)

    def zoom_out(self) -> None:
        self.zoom_mode.set("manual")
        self.scale_ratio = max(self.scale_ratio / 1.25, 0.1)
        if self.current_image_name is not None:
            self.show_image(self.current_image_name, preserve_ratio=True)

    def _on_window_resize(self, _event: tk.Event) -> None:
        if self.zoom_mode.get() == "fit" and self.current_image_name is not None:
            self.root.after_idle(lambda: self.current_image_name and self.show_image(self.current_image_name))

    def _refresh_current_image(self) -> None:
        if self.current_image_name is not None:
            self.show_image(self.current_image_name, preserve_ratio=True)

    def choose_predict_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择 predict 输出目录")
        if not selected:
            return
        try:
            self.load_predict_dir(Path(selected))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加载失败", str(exc))

    def select_image_by_name(self, image_name: str) -> None:
        if image_name not in self.image_names:
            raise FileNotFoundError(f"在当前 predict 目录中未找到图片: {image_name}")
        index = self.image_names.index(image_name)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(index)
        self.image_listbox.see(index)
        self.show_image(image_name)

    def load_predict_dir(self, predict_dir: Path) -> None:
        summary_path = predict_dir / "phase2_postprocess_summary.json"
        run_config_path = predict_dir / "run_config.json"
        if not predict_dir.exists():
            raise FileNotFoundError(f"目录不存在: {predict_dir}")
        if not summary_path.exists():
            raise FileNotFoundError(f"未找到 phase2_postprocess_summary.json: {summary_path}")
        if not run_config_path.exists():
            raise FileNotFoundError(f"未找到 run_config.json: {run_config_path}")

        self.summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        self.rule_config = dict(run_config.get("postprocess_rules") or {})
        self.source_image_paths = self._build_source_image_map(run_config)
        self.rows_by_image.clear()
        for row in self.summary_rows:
            image_name = str(row.get("image_name") or "")
            if not image_name:
                continue
            self.rows_by_image.setdefault(image_name, []).append(row)

        self.image_names = sorted(
            name
            for name in self.rows_by_image
            if (predict_dir / name).exists() and (predict_dir / name).suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_names:
            raise FileNotFoundError(f"在目录中没有找到可展示的预测图片: {predict_dir}")

        self.predict_dir = predict_dir
        self.dir_var.set(str(predict_dir))
        self.image_listbox.delete(0, tk.END)
        for name in self.image_names:
            self.image_listbox.insert(tk.END, name)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(0)
        self.image_listbox.event_generate("<<ListboxSelect>>")

    def _build_source_image_map(self, run_config: dict) -> dict[str, Path]:
        source_value = run_config.get("source")
        if not source_value:
            return {}
        source_path = Path(str(source_value))
        image_map: dict[str, Path] = {}
        if source_path.is_file() and source_path.suffix.lower() == ".txt":
            for line in source_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                path = Path(line)
                image_map[path.name] = path
            return image_map
        if source_path.is_dir():
            for path in source_path.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_map[path.name] = path
        return image_map

    def _on_image_select(self, _event: object) -> None:
        selection = self.image_listbox.curselection()
        if not selection or self.predict_dir is None:
            return
        image_name = self.image_listbox.get(selection[0])
        self.show_image(image_name)

    def show_image(self, image_name: str, preserve_ratio: bool = False) -> None:
        if self.predict_dir is None:
            return
        image_path = self.predict_dir / image_name
        self.current_image_name = image_name
        image_rows = list(self.rows_by_image.get(image_name, []))
        if not self.show_suppressed_var.get():
            image_rows = [row for row in image_rows if not bool(row.get("temporal_suppressed"))]
        self.current_rows = sorted(image_rows, key=lambda item: int(item.get("det_index", -1)))
        self.current_selected_det_index = None

        if Image is not None and ImageTk is not None:
            original_pil = Image.open(image_path)
            self.original_pil_image = original_pil
            self.original_image = None
            original_width, original_height = original_pil.size
        else:
            original = tk.PhotoImage(file=str(image_path))
            self.original_image = original
            self.original_pil_image = None
            original_width, original_height = original.width(), original.height()

        source_path = self.source_image_paths.get(image_name)
        self.current_source_pil_image = None
        if source_path is not None and source_path.exists() and Image is not None:
            self.current_source_pil_image = Image.open(source_path)

        if not preserve_ratio:
            if self.zoom_mode.get() == "actual":
                self.scale_ratio = 1.0
            elif self.zoom_mode.get() == "fit":
                frame_w = max(300, self.canvas.winfo_width())
                frame_h = max(300, self.canvas.winfo_height())
                self.scale_ratio = min(frame_w / max(1, original_width), frame_h / max(1, original_height), 1.0)
            elif self.zoom_mode.get() != "manual":
                self.scale_ratio = 1.0

        if Image is not None and ImageTk is not None and self.original_pil_image is not None:
            if self.scale_ratio == 1.0:
                resized = self.original_pil_image
            else:
                target_w = max(1, int(round(original_width * self.scale_ratio)))
                target_h = max(1, int(round(original_height * self.scale_ratio)))
                resized = self.original_pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            display_ratio = self.scale_ratio
        else:
            if self.scale_ratio > 1.0:
                zoom = max(1, round(self.scale_ratio))
                photo = self.original_image.zoom(zoom, zoom)
                display_ratio = float(zoom)
            elif self.scale_ratio == 1.0:
                photo = self.original_image
                display_ratio = 1.0
            else:
                sample = max(1, round(1 / self.scale_ratio))
                photo = self.original_image.subsample(sample, sample)
                display_ratio = 1.0 / sample
        self.tk_image = photo
        self.scale_ratio = display_ratio

        self.canvas.delete("all")
        self.canvas_items_by_det_index.clear()
        self.canvas.config(scrollregion=(0, 0, photo.width(), photo.height()))
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

        for row in self.current_rows:
            det_index = int(row["det_index"])
            applied_rules = list(row.get("applied_rules") or [])
            color = color_for_row(applied_rules, row.get("phase2_decision"))
            x1, y1, x2, y2 = [float(value) for value in row["xyxy"]]
            sx1, sy1, sx2, sy2 = [value * self.scale_ratio for value in (x1, y1, x2, y2)]
            rect_id = self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2)
            text = f"#{det_index}"
            text_id = self.canvas.create_text(
                sx1 + 6,
                max(10, sy1 + 12),
                text=text,
                anchor="w",
                fill=color,
                font=("Segoe UI", 11, "bold"),
            )
            self.canvas_items_by_det_index[det_index] = [rect_id, text_id]

        self.det_listbox.delete(0, tk.END)
        for row in self.current_rows:
            det_index = int(row["det_index"])
            final_conf = format_value(row.get("final_conf"))
            decision = row.get("phase2_decision") or "none"
            rules = ",".join(row.get("applied_rules") or []) or "none"
            status = "drop" if bool(row.get("temporal_suppressed")) else "keep"
            self.det_listbox.insert(
                tk.END,
                f"#{det_index} | {status} | conf={final_conf} | decision={decision} | rules={rules}",
            )

        if self.current_rows:
            self.det_listbox.selection_clear(0, tk.END)
            self.det_listbox.selection_set(0)
            self.det_listbox.event_generate("<<ListboxSelect>>")

    def _on_det_select(self, _event: object) -> None:
        selection = self.det_listbox.curselection()
        if not selection or not self.current_rows:
            return
        row = self.current_rows[selection[0]]
        self.select_detection(int(row["det_index"]))

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not self.current_rows:
            return
        x = event.x / self.scale_ratio
        y = event.y / self.scale_ratio
        for index, row in enumerate(self.current_rows):
            x1, y1, x2, y2 = [float(value) for value in row["xyxy"]]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.det_listbox.selection_clear(0, tk.END)
                self.det_listbox.selection_set(index)
                self.det_listbox.see(index)
                self.select_detection(int(row["det_index"]))
                return

    def select_detection(self, det_index: int) -> None:
        self.current_selected_det_index = det_index
        for current_det_index, item_ids in self.canvas_items_by_det_index.items():
            active = current_det_index == det_index
            self.canvas.itemconfigure(item_ids[0], width=4 if active else 2)
            self.canvas.itemconfigure(item_ids[1], font=("Segoe UI", 12 if active else 11, "bold"))

        row = next((item for item in self.current_rows if int(item["det_index"]) == det_index), None)
        if row is None:
            return

        x1, y1, x2, y2 = [float(value) for value in row["xyxy"]]
        horizontal_edge_type = row.get("horizontal_rule_edge_type")
        source_for_metrics = self.current_source_pil_image
        image_height_for_metrics = None
        if source_for_metrics is not None:
            image_height_for_metrics = int(source_for_metrics.height)
        elif self.original_pil_image is not None:
            image_height_for_metrics = int(self.original_pil_image.height)
        elif self.original_image is not None:
            image_height_for_metrics = int(self.original_image.height())
        top_gap, bottom_gap = compute_vertical_gaps(
            y1=y1,
            y2=y2,
            image_h=image_height_for_metrics,
            top_gap=row.get("horizontal_top_gap"),
            bottom_gap=row.get("horizontal_bottom_gap"),
        )
        nearest_side = detect_nearest_vertical_side(top_gap, bottom_gap)
        vertical_edge_margin_px = float(self.rule_config.get("vertical_edge_margin_px", 0.0) or 0.0)
        vertical_edge_type = row.get("actual_vertical_edge_type")
        if source_for_metrics is not None:
            vertical_edge_type = vertical_edge_type or detect_vertical_edge_type(y1, y2, source_for_metrics.height, vertical_edge_margin_px)
        vertical_edge_span_score = row.get("actual_vertical_edge_span_score")
        if vertical_edge_span_score is None:
            vertical_edge_span_score = row.get("phase2_edge_span_score")
        if vertical_edge_span_score is None and vertical_edge_type not in {"top", "bottom", "top_bottom"}:
            vertical_edge_span_score = 0.0
        if vertical_edge_span_score is None and source_for_metrics is not None and np is not None and vertical_edge_type in {"top", "bottom", "top_bottom"}:
            ix1 = max(0, min(source_for_metrics.width - 1, int(round(x1))))
            iy1 = max(0, min(source_for_metrics.height - 1, int(round(y1))))
            ix2 = max(ix1 + 1, min(source_for_metrics.width, int(round(x2))))
            iy2 = max(iy1 + 1, min(source_for_metrics.height, int(round(y2))))
            crop = np.array(source_for_metrics.crop((ix1, iy1, ix2, iy2)))
            vertical_edge_span_score = compute_edge_span_ratio_from_array(crop, vertical_edge_type)

        header_lines = [
            "[参数比较阈值]",
            f"horizontal_flat_ratio_threshold: {format_value(self.rule_config.get('horizontal_flat_ratio_threshold'))}",
            f"horizontal_edge_span_threshold: {format_value(self.rule_config.get('horizontal_edge_span_threshold'))}",
            f"vertical_intact_aspect_ratio_threshold: {format_value(self.rule_config.get('vertical_intact_aspect_ratio_threshold'))}",
            f"vertical_edge_span_threshold: {format_value(self.rule_config.get('vertical_edge_span_threshold'))}",
            "",
            f"base_conf -> final_conf: {format_value(row.get('base_conf'))} -> {format_value(row.get('final_conf'))}",
            f"temporal_suppressed: {format_value(row.get('temporal_suppressed'))}",
            f"temporal_kept: {format_value(row.get('temporal_kept'))}",
            f"temporal_partner: {format_value(row.get('temporal_partner_image_name'))} / det {format_value(row.get('temporal_partner_det_index'))}",
            f"temporal_expected_offset(dx,dy): {format_value(row.get('temporal_expected_dx'))}, {format_value(row.get('temporal_expected_dy'))}",
            f"temporal_overlap_ratio: {format_value(row.get('temporal_overlap_ratio'))}",
            f"adjacent_compare_self_height: {format_value(row.get('adjacent_compare_self_height'))}",
            f"adjacent_compare_partner_height: {format_value(row.get('adjacent_compare_partner_height'))}",
            f"adjacent_compare_self_width: {format_value(row.get('adjacent_compare_self_width'))}",
            f"adjacent_compare_partner_width: {format_value(row.get('adjacent_compare_partner_width'))}",
            "",
            *build_rule_summary(row),
            "",
            "[边缘信息]",
            f"nearest_side(top/bottom): {nearest_side}",
            f"top_gap: {format_value(top_gap)}",
            f"bottom_gap: {format_value(bottom_gap)}",
            f"nearest_edge_gap(min(top_gap,bottom_gap)): {format_value(min(top_gap, bottom_gap) if top_gap is not None and bottom_gap is not None else None)}",
            f"horizontal_edge_type(actual): {format_value(horizontal_edge_type)}",
            f"vertical_edge_type(actual): {format_value(vertical_edge_type)}",
            "",
            "[horizontal 实际值]",
            f"actual_horizontal_flat_ratio(w/h): {format_value(row.get('horizontal_width_height_ratio'))}",
            f"actual_horizontal_edge_span_score: {format_value(row.get('horizontal_edge_span_score'))}",
            "",
            "[vertical 实际值]",
            f"actual_vertical_intact_aspect_ratio(h/w): {format_value(row.get('phase2_aspect_ratio'))}",
            f"actual_vertical_edge_span_score: {format_value(vertical_edge_span_score)}",
        ]
        text = "\n".join(header_lines)
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = PostprocessInspectorApp(root)
    if args.predict_dir:
        app.load_predict_dir(Path(args.predict_dir))
        if args.image_name:
            app.select_image_by_name(args.image_name)
    if False:
        print(app)  # pragma: no cover
    root.mainloop()


if __name__ == "__main__":
    main()
