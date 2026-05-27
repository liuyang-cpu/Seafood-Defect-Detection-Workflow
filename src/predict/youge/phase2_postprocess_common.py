from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


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


def evaluate_horizontal_edge_penalty(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_height: int,
    edge_touch_px: int,
    flat_ratio_threshold_short: float,
    flat_ratio_threshold: float,
    edge_span_threshold_short: float,
    edge_span_threshold: float,
) -> dict[str, float | bool]:
    width = max(0.0, x2 - x1)
    height = max(1e-6, y2 - y1)
    aspect_ratio = width / height
    top_gap = y1
    bottom_gap = image_height - y2
    min_edge_gap = min(top_gap, bottom_gap)
    edge_type = detect_vertical_edge_type(y1, y2, image_height, edge_touch_px)
    ix1 = max(0, min(image.shape[1] - 1, round(x1)))
    iy1 = max(0, min(image.shape[0] - 1, round(y1)))
    ix2 = max(ix1 + 1, min(image.shape[1], round(x2)))
    iy2 = max(iy1 + 1, min(image.shape[0], round(y2)))
    crop = image[iy1:iy2, ix1:ix2]
    edge_span_score = (
        compute_edge_span_ratio_from_crop(crop, edge_type) if edge_type in {"top", "bottom", "top_bottom"} else 0.0
    )
    short_penalty_apply = (
        min_edge_gap <= float(edge_touch_px)
        and aspect_ratio >= float(flat_ratio_threshold_short)
        and aspect_ratio < float(flat_ratio_threshold)
        and edge_span_score >= float(edge_span_threshold_short)
    )
    return {
        "apply": (
            min_edge_gap <= float(edge_touch_px)
            and aspect_ratio >= float(flat_ratio_threshold)
            and edge_span_score > float(edge_span_threshold)
        ),
        "short_penalty_apply": short_penalty_apply,
        "width_height_ratio": aspect_ratio,
        "top_gap": top_gap,
        "bottom_gap": bottom_gap,
        "min_edge_gap": min_edge_gap,
        "edge_span_score": edge_span_score,
        "edge_type": edge_type,
    }


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


def compute_edge_span_ratio_from_crop(crop: np.ndarray, edge_type: str) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
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


def bbox_area(det: dict) -> float:
    width = max(0.0, float(det["x2"]) - float(det["x1"]))
    height = max(0.0, float(det["y2"]) - float(det["y1"]))
    return width * height


def intersection_area(det_a: dict, det_b: dict) -> float:
    ix1 = max(float(det_a["x1"]), float(det_b["x1"]))
    iy1 = max(float(det_a["y1"]), float(det_b["y1"]))
    ix2 = min(float(det_a["x2"]), float(det_b["x2"]))
    iy2 = min(float(det_a["y2"]), float(det_b["y2"]))
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def center_distance_ratio(smaller: dict, larger: dict) -> float:
    small_w = max(1e-6, float(smaller["x2"]) - float(smaller["x1"]))
    small_h = max(1e-6, float(smaller["y2"]) - float(smaller["y1"]))
    small_diag = max(1e-6, float(np.hypot(small_w, small_h)))
    small_cx = (float(smaller["x1"]) + float(smaller["x2"])) / 2.0
    small_cy = (float(smaller["y1"]) + float(smaller["y2"])) / 2.0
    large_cx = (float(larger["x1"]) + float(larger["x2"])) / 2.0
    large_cy = (float(larger["y1"]) + float(larger["y2"])) / 2.0
    return float(np.hypot(large_cx - small_cx, large_cy - small_cy)) / small_diag


def suppress_same_class_contained_boxes(
    detections: list[dict[str, float | int | str | None]],
    contain_ratio_threshold: float,
) -> list[dict[str, float | int | str | None]]:
    if len(detections) <= 1:
        return detections

    keep_mask = [True] * len(detections)
    ranked_indices = sorted(
        range(len(detections)),
        key=lambda idx: (float(detections[idx]["conf"]), bbox_area(detections[idx])),
        reverse=True,
    )

    for i, high_idx in enumerate(ranked_indices):
        if not keep_mask[high_idx]:
            continue
        det_high = detections[high_idx]
        area_high = bbox_area(det_high)
        if area_high <= 0:
            continue
        for low_idx in ranked_indices[i + 1 :]:
            if not keep_mask[low_idx]:
                continue
            det_low = detections[low_idx]
            if int(det_high["class_id"]) != int(det_low["class_id"]):
                continue
            area_low = bbox_area(det_low)
            if area_low <= 0:
                continue
            inter = intersection_area(det_high, det_low)
            smaller_area = min(area_high, area_low)
            contain_ratio = inter / smaller_area if smaller_area > 0 else 0.0
            if contain_ratio < float(contain_ratio_threshold):
                continue
            # Guard against deleting a real neighbor when one large box spans multiple objects.
            if center_distance_ratio(det_low, det_high) > 0.45:
                continue
            keep_mask[low_idx] = False

    return [det for det, keep in zip(detections, keep_mask) if keep]


def horizontal_overlap_ratio(
    det_a: dict[str, float | int | str | None],
    det_b: dict[str, float | int | str | None],
) -> float:
    width_a = max(1e-6, float(det_a["x2"]) - float(det_a["x1"]))
    width_b = max(1e-6, float(det_b["x2"]) - float(det_b["x1"]))
    overlap = max(0.0, min(float(det_a["x2"]), float(det_b["x2"])) - max(float(det_a["x1"]), float(det_b["x1"])))
    return overlap / min(width_a, width_b)


def choose_adjacent_frame_keep_drop(
    *,
    current_det: dict[str, float | int | str | None],
    current_frame: dict[str, object],
    next_det: dict[str, float | int | str | None],
    next_frame: dict[str, object],
    edge_priority_rule: bool,
    edge_touch_margin_px: float,
    height_confidence_offset_px: float,
) -> tuple[
    dict[str, float | int | str | None], dict[str, object], dict[str, float | int | str | None], dict[str, object]
]:
    if edge_priority_rule:
        current_edge_type = detect_vertical_edge_type(
            float(current_det["y1"]),
            float(current_det["y2"]),
            int(current_frame["image_height"]),
            edge_touch_margin_px,
        )
        next_edge_type = detect_vertical_edge_type(
            float(next_det["y1"]),
            float(next_det["y2"]),
            int(next_frame["image_height"]),
            edge_touch_margin_px,
        )
        current_is_bottom_edge = current_edge_type in {"bottom", "top_bottom"}
        next_is_top_edge = next_edge_type in {"top", "top_bottom"}

        if current_is_bottom_edge != next_is_top_edge:
            if not current_is_bottom_edge:
                return current_det, current_frame, next_det, next_frame
            return next_det, next_frame, current_det, current_frame

        if current_is_bottom_edge and next_is_top_edge:
            current_height = max(0.0, float(current_det["y2"]) - float(current_det["y1"]))
            next_height = max(0.0, float(next_det["y2"]) - float(next_det["y1"]))
            if abs(current_height - next_height) < float(height_confidence_offset_px):
                current_height = next_height = 0.0
            if current_height > next_height:
                return current_det, current_frame, next_det, next_frame
            if next_height > current_height:
                return next_det, next_frame, current_det, current_frame

    current_base_conf = float(current_det["base_conf"])
    next_base_conf = float(next_det["base_conf"])
    if current_base_conf >= next_base_conf:
        return current_det, current_frame, next_det, next_frame
    return next_det, next_frame, current_det, current_frame


def apply_adjacent_frame_dedup(
    frame_entries: list[dict[str, object]],
    *,
    edge_priority_rule: bool,
    x_overlap_threshold: float,
    delta_y: float,
    height_min: float,
    height_max: float,
    height_confidence_offset_px: float,
    non_bottom_y2_correction: float,
    bottom_touch_margin_px: float,
    x_tolerance: float,
) -> None:
    ordered_frames = sorted(frame_entries, key=lambda item: str(item["output_name"]))
    current_min_y1 = float(delta_y) - float(height_max)
    for frame_index in range(len(ordered_frames) - 1):
        current_frame = ordered_frames[frame_index]
        next_frame = ordered_frames[frame_index + 1]
        current_detections = list(current_frame["detections"])
        next_detections = list(next_frame["detections"])
        candidates: list[tuple[float, int, int, float]] = []

        for current_index, current_det in enumerate(current_detections):
            if bool(current_det.get("temporal_matched")):
                continue
            for next_index, next_det in enumerate(next_detections):
                if bool(next_det.get("temporal_matched")):
                    continue
                current_y1 = float(current_det["y1"])
                if current_y1 <= current_min_y1:
                    continue
                next_y2 = float(next_det["y2"])
                expected_next_y2_min = current_y1 - float(delta_y) + float(height_min)
                current_y2 = float(current_det["y2"])
                current_height_px = max(0.0, current_y2 - current_y1)
                image_height = float(current_frame["image_height"])
                is_bottom_touching = (image_height - current_y2) <= float(bottom_touch_margin_px)
                if is_bottom_touching:
                    expected_next_y2_max = current_y1 - float(delta_y) + float(height_max)
                else:
                    expected_next_y2_max = (
                        current_y1 - float(delta_y) + current_height_px + float(non_bottom_y2_correction)
                    )
                if next_y2 < expected_next_y2_min or next_y2 > expected_next_y2_max:
                    continue
                overlap_ratio = horizontal_overlap_ratio(current_det, next_det)
                if overlap_ratio < float(x_overlap_threshold):
                    continue
                left_dx = abs(float(next_det["x1"]) - float(current_det["x1"]))
                right_dx = abs(float(next_det["x2"]) - float(current_det["x2"]))
                if min(left_dx, right_dx) > float(x_tolerance):
                    continue
                y_gap_to_window = min(
                    abs(next_y2 - expected_next_y2_min),
                    abs(next_y2 - expected_next_y2_max),
                )
                score = (
                    (min(left_dx, right_dx) / max(float(x_tolerance), 1.0))
                    + (y_gap_to_window / max(float(height_max - height_min), 1.0))
                    + (1.0 - overlap_ratio)
                )
                candidates.append((score, current_index, next_index, overlap_ratio))

        candidates.sort(key=lambda item: item[0])
        used_current: set[int] = set()
        used_next: set[int] = set()
        for _score, current_index, next_index, overlap_ratio in candidates:
            if current_index in used_current or next_index in used_next:
                continue
            current_det = current_detections[current_index]
            next_det = next_detections[next_index]
            if bool(current_det.get("temporal_matched")) or bool(next_det.get("temporal_matched")):
                continue

            kept_det, kept_frame, dropped_det, dropped_frame = choose_adjacent_frame_keep_drop(
                current_det=current_det,
                current_frame=current_frame,
                next_det=next_det,
                next_frame=next_frame,
                edge_priority_rule=edge_priority_rule,
                edge_touch_margin_px=bottom_touch_margin_px,
                height_confidence_offset_px=height_confidence_offset_px,
            )

            kept_det["temporal_matched"] = True
            kept_det["temporal_kept"] = True
            kept_det["temporal_partner_image_name"] = str(dropped_frame["output_name"])
            kept_det["temporal_partner_det_index"] = int(dropped_det["det_index"])
            kept_det["temporal_partner_base_conf"] = float(dropped_det["base_conf"])
            kept_det["temporal_overlap_ratio"] = overlap_ratio
            kept_det["temporal_expected_dx"] = 0.0
            kept_det["temporal_expected_dy"] = -float(delta_y)
            kept_det["adjacent_compare_self_height"] = max(0.0, float(kept_det["y2"]) - float(kept_det["y1"]))
            kept_det["adjacent_compare_partner_height"] = max(0.0, float(dropped_det["y2"]) - float(dropped_det["y1"]))
            kept_det["adjacent_compare_self_width"] = max(0.0, float(kept_det["x2"]) - float(kept_det["x1"]))
            kept_det["adjacent_compare_partner_width"] = max(0.0, float(dropped_det["x2"]) - float(dropped_det["x1"]))
            if "adjacent_frame_dedup_keep" not in kept_det["applied_rules"]:
                kept_det["applied_rules"].append("adjacent_frame_dedup_keep")

            dropped_det["temporal_matched"] = True
            dropped_det["temporal_kept"] = False
            dropped_det["temporal_suppressed"] = True
            dropped_det["temporal_partner_image_name"] = str(kept_frame["output_name"])
            dropped_det["temporal_partner_det_index"] = int(kept_det["det_index"])
            dropped_det["temporal_partner_base_conf"] = float(kept_det["base_conf"])
            dropped_det["temporal_overlap_ratio"] = overlap_ratio
            dropped_det["temporal_expected_dx"] = 0.0
            dropped_det["temporal_expected_dy"] = -float(delta_y)
            dropped_det["adjacent_compare_self_height"] = max(0.0, float(dropped_det["y2"]) - float(dropped_det["y1"]))
            dropped_det["adjacent_compare_partner_height"] = max(0.0, float(kept_det["y2"]) - float(kept_det["y1"]))
            dropped_det["adjacent_compare_self_width"] = max(0.0, float(dropped_det["x2"]) - float(dropped_det["x1"]))
            dropped_det["adjacent_compare_partner_width"] = max(0.0, float(kept_det["x2"]) - float(kept_det["x1"]))
            if "adjacent_frame_dedup_drop" not in dropped_det["applied_rules"]:
                dropped_det["applied_rules"].append("adjacent_frame_dedup_drop")

            used_current.add(current_index)
            used_next.add(next_index)


def classify_vertical_edge_box(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    edge_type: str | None,
    intact_aspect_ratio_threshold: float,
    edge_span_threshold: float,
    edge_span_threshold_thin: float,
    precomputed_edge_span_score: float | None = None,
) -> dict[str, float | str | None]:
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect_ratio = height / width
    result = {
        "decision": None,
        "aspect_ratio": aspect_ratio,
        "edge_span_score": None,
        "reason": None,
    }
    if edge_type not in {"top", "bottom", "top_bottom"}:
        return result

    if precomputed_edge_span_score is None:
        ix1 = max(0, min(image.shape[1] - 1, round(x1)))
        iy1 = max(0, min(image.shape[0] - 1, round(y1)))
        ix2 = max(ix1 + 1, min(image.shape[1], round(x2)))
        iy2 = max(iy1 + 1, min(image.shape[0], round(y2)))
        crop = image[iy1:iy2, ix1:ix2]
        edge_span_score = compute_edge_span_ratio_from_crop(crop, edge_type)
    else:
        edge_span_score = float(precomputed_edge_span_score)
    result["edge_span_score"] = edge_span_score

    if aspect_ratio >= float(intact_aspect_ratio_threshold) and edge_span_score < float(edge_span_threshold_thin):
        result["decision"] = "intact"
        result["reason"] = "aspect_ratio_override"
        return result

    result["decision"] = "defective" if edge_span_score >= float(edge_span_threshold) else "intact"
    result["reason"] = "edge_span_ratio"
    return result


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def build_penalty_preview_lines(det: dict, rule_name: str) -> list[str]:
    lines = [
        f"image={det['image_name']}",
        f"rule={rule_name}",
        f"class={int(det['class_id'])} base_conf={float(det['base_conf']):.4f} final_conf={float(det['conf']):.4f}",
    ]
    if rule_name == "horizontal_edge_penalty":
        lines.extend(
            [
                (
                    "w_h_ratio="
                    f"{format_metric(det['horizontal_width_height_ratio'])} "
                    f"threshold={format_metric(det['horizontal_flat_ratio_threshold'])}"
                ),
                (
                    "top_gap="
                    f"{format_metric(det['horizontal_top_gap'])} "
                    f"bottom_gap={format_metric(det['horizontal_bottom_gap'])} "
                    f"min_gap={format_metric(det['horizontal_min_edge_gap'])} "
                    f"edge_touch_px={int(det['horizontal_edge_touch_px'])}"
                ),
                (
                    "edge_span_score="
                    f"{format_metric(det['horizontal_edge_span_score'])} "
                    f"threshold>{format_metric(det['horizontal_edge_span_threshold'])}"
                ),
                f"penalty_factor={format_metric(det['horizontal_penalty_factor'])}",
            ]
        )
    elif rule_name == "vertical_defective_penalty":
        lines.extend(
            [
                (
                    f"edge_type={det['edge_type']} "
                    f"h_w_ratio={format_metric(det['phase2_aspect_ratio'])} "
                    f"intact_threshold={format_metric(det['intact_aspect_ratio_threshold'])}"
                ),
                (
                    "edge_span_score="
                    f"{format_metric(det['phase2_edge_span_score'])} "
                    f"defective_threshold={format_metric(det['edge_span_threshold'])}"
                ),
                (
                    f"decision={det['phase2_decision']} "
                    f"reason={det['phase2_reason']} "
                    f"penalty_factor={format_metric(det['defective_penalty_factor'])}"
                ),
            ]
        )
    return lines


def render_penalty_preview(image: np.ndarray, det: dict, rule_name: str, output_path: Path) -> None:
    image_h, image_w = image.shape[:2]
    x1 = max(0, min(image_w - 1, round(float(det["x1"]))))
    y1 = max(0, min(image_h - 1, round(float(det["y1"]))))
    x2 = max(x1 + 1, min(image_w, round(float(det["x2"]))))
    y2 = max(y1 + 1, min(image_h, round(float(det["y2"]))))

    pad = 16
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(image_w, x2 + pad)
    crop_y2 = min(image_h, y2 + pad)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop.size == 0:
        return

    local_x1 = x1 - crop_x1
    local_y1 = y1 - crop_y1
    local_x2 = x2 - crop_x1
    local_y2 = y2 - crop_y1
    color = color_for_class(int(det["class_id"]))
    cv2.rectangle(crop, (local_x1, local_y1), (local_x2, local_y2), color, 2, cv2.LINE_AA)

    lines = build_penalty_preview_lines(det, rule_name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    line_height = 22
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_text_width = max((size[0] for size in text_sizes), default=0)
    side_padding = 12
    top_padding = 12
    bottom_padding = 10
    header_height = top_padding + bottom_padding + line_height * len(lines)
    canvas_width = max(crop.shape[1], max_text_width + side_padding * 2)
    canvas = np.full((crop.shape[0] + header_height, canvas_width, 3), 255, dtype=np.uint8)
    crop_x = (canvas_width - crop.shape[1]) // 2
    canvas[header_height:, crop_x : crop_x + crop.shape[1]] = crop
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, header_height - 1), (245, 247, 250), -1)
    cv2.line(canvas, (0, header_height - 1), (canvas.shape[1] - 1, header_height - 1), (210, 214, 220), 1)
    for idx, line in enumerate(lines):
        y = top_padding + 16 + idx * line_height
        cv2.putText(canvas, line, (side_padding, y), font, font_scale, (30, 41, 59), thickness, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Failed to write penalty preview image: {output_path}")


def render_adjacent_frame_guide_lines(
    image: np.ndarray,
    *,
    delta_y: float,
    height_max: float,
) -> None:
    image_height, image_width = image.shape[:2]
    y_value = delta_y - height_max
    y = round(float(y_value))
    if y < 0 or y >= image_height:
        return
    color = (0, 255, 255)
    cv2.line(image, (0, y), (image_width - 1, y), color, 2, cv2.LINE_AA)
    text = f"y1 scan start: {float(y_value):.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_top = max(0, y - text_height - baseline - 6)
    text_bottom = min(image_height - 1, text_top + text_height + baseline + 6)
    text_right = min(image_width - 1, 8 + text_width + 12)
    cv2.rectangle(image, (8, text_top), (text_right, text_bottom), color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        text,
        (14, text_bottom - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def save_adjusted_predictions(
    *,
    results: list,
    save_dir: Path,
    save_txt: bool,
    save_conf: bool,
    horizontal_penalty_enabled: bool,
    same_class_contain_suppression: bool,
    contain_ratio_threshold: float,
    adjacent_frame_dedup: bool,
    adjacent_frame_edge_priority_rule: bool,
    adjacent_frame_x_overlap_threshold: float,
    adjacent_frame_delta_y: float,
    adjacent_frame_height_min: float,
    adjacent_frame_height_max: float,
    adjacent_frame_height_confidence_offset_px: float,
    adjacent_frame_non_bottom_y2_correction: float,
    adjacent_frame_bottom_touch_margin_px: float,
    adjacent_frame_x_tolerance: float,
    horizontal_edge_touch_px: int,
    horizontal_flat_ratio_threshold_short: float,
    horizontal_flat_ratio_threshold: float,
    horizontal_edge_span_threshold_short: float,
    horizontal_edge_span_threshold: float,
    horizontal_penalty_factor: float,
    vertical_rule_enabled: bool,
    vertical_intact_aspect_ratio_threshold: float,
    vertical_edge_span_threshold: float,
    vertical_edge_span_threshold_thin: float,
    vertical_defective_penalty_factor: float,
    vertical_edge_margin_px: float,
    export_penalty_hits: bool = True,
    penalty_hits_dirname: str = "penalty_hits",
    render_adjacent_frame_guides: bool = False,
) -> None:
    labels_dir = save_dir / "labels"
    if save_txt:
        labels_dir.mkdir(parents=True, exist_ok=True)
    penalty_hits_dir = save_dir / penalty_hits_dirname if export_penalty_hits else None

    frame_entries: list[dict[str, object]] = []
    summary: list[dict] = []
    for result in results:
        image = result.orig_img.copy()
        original_image = result.orig_img.copy()
        image_height, image_width = image.shape[:2]
        output_name = Path(result.path).name
        output_path = save_dir / output_name
        detections: list[dict[str, float | int | str | None]] = []

        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy_rows = boxes.xyxy.tolist()
            xywhn_rows = boxes.xywhn.tolist()
            conf_rows = boxes.conf.tolist()
            cls_rows = boxes.cls.tolist()
            for det_index, (xyxy, xywhn, raw_conf, raw_cls) in enumerate(
                zip(xyxy_rows, xywhn_rows, conf_rows, cls_rows)
            ):
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                cls_id = int(raw_cls)
                conf = float(raw_conf)
                base_conf = conf
                applied_rules: list[str] = []
                horizontal_eval = {
                    "apply": False,
                    "short_penalty_apply": False,
                    "width_height_ratio": None,
                    "top_gap": None,
                    "bottom_gap": None,
                    "min_edge_gap": None,
                    "edge_span_score": 0.0,
                    "edge_type": None,
                }

                if horizontal_penalty_enabled:
                    horizontal_eval = evaluate_horizontal_edge_penalty(
                        image=image,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        image_height=image_height,
                        edge_touch_px=horizontal_edge_touch_px,
                        flat_ratio_threshold_short=horizontal_flat_ratio_threshold_short,
                        flat_ratio_threshold=horizontal_flat_ratio_threshold,
                        edge_span_threshold_short=horizontal_edge_span_threshold_short,
                        edge_span_threshold=horizontal_edge_span_threshold,
                    )
                    if horizontal_eval["apply"]:
                        conf *= float(horizontal_penalty_factor)
                        applied_rules.append("horizontal_edge_penalty")
                    elif horizontal_eval["short_penalty_apply"]:
                        conf *= float(horizontal_penalty_factor)
                        applied_rules.append("horizontal_short_edge_penalty")

                edge_type = detect_vertical_edge_type(y1, y2, image_height, vertical_edge_margin_px)
                actual_vertical_edge_type = edge_type
                actual_vertical_edge_span_score = 0.0
                if actual_vertical_edge_type in {"top", "bottom", "top_bottom"}:
                    ix1 = max(0, min(image.shape[1] - 1, round(x1)))
                    iy1 = max(0, min(image.shape[0] - 1, round(y1)))
                    ix2 = max(ix1 + 1, min(image.shape[1], round(x2)))
                    iy2 = max(iy1 + 1, min(image.shape[0], round(y2)))
                    actual_crop = image[iy1:iy2, ix1:ix2]
                    actual_vertical_edge_span_score = compute_edge_span_ratio_from_crop(
                        actual_crop, actual_vertical_edge_type
                    )
                phase2 = classify_vertical_edge_box(
                    image=image,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    edge_type=edge_type,
                    intact_aspect_ratio_threshold=vertical_intact_aspect_ratio_threshold,
                    edge_span_threshold=vertical_edge_span_threshold,
                    edge_span_threshold_thin=vertical_edge_span_threshold_thin,
                    precomputed_edge_span_score=actual_vertical_edge_span_score
                    if actual_vertical_edge_type in {"top", "bottom", "top_bottom"}
                    else None,
                )
                horizontal_applied = any(
                    rule_name in {"horizontal_edge_penalty", "horizontal_short_edge_penalty"}
                    for rule_name in applied_rules
                )
                if vertical_rule_enabled and not horizontal_applied and phase2["decision"] == "defective":
                    conf *= float(vertical_defective_penalty_factor)
                    applied_rules.append("vertical_defective_penalty")
                elif vertical_rule_enabled and not horizontal_applied and phase2["decision"] == "intact":
                    applied_rules.append(f"vertical_{phase2['reason']}")

                detections.append(
                    {
                        "class_id": cls_id,
                        "image_name": output_name,
                        "det_index": det_index,
                        "base_conf": base_conf,
                        "conf": conf,
                        "x_center": float(xywhn[0]),
                        "y_center": float(xywhn[1]),
                        "x_center_px": (x1 + x2) / 2.0,
                        "y_center_px": (y1 + y2) / 2.0,
                        "width": float(xywhn[2]),
                        "height": float(xywhn[3]),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "edge_type": edge_type,
                        "actual_vertical_edge_type": actual_vertical_edge_type,
                        "phase2_decision": phase2["decision"],
                        "phase2_reason": phase2["reason"],
                        "phase2_aspect_ratio": phase2["aspect_ratio"],
                        "phase2_edge_span_score": phase2["edge_span_score"],
                        "actual_vertical_edge_span_score": actual_vertical_edge_span_score,
                        "horizontal_width_height_ratio": horizontal_eval["width_height_ratio"],
                        "horizontal_top_gap": horizontal_eval["top_gap"],
                        "horizontal_bottom_gap": horizontal_eval["bottom_gap"],
                        "horizontal_min_edge_gap": horizontal_eval["min_edge_gap"],
                        "horizontal_edge_span_score": horizontal_eval["edge_span_score"],
                        "horizontal_rule_edge_type": horizontal_eval["edge_type"],
                        "horizontal_edge_touch_px": horizontal_edge_touch_px,
                        "horizontal_flat_ratio_threshold_short": horizontal_flat_ratio_threshold_short,
                        "horizontal_flat_ratio_threshold": horizontal_flat_ratio_threshold,
                        "horizontal_edge_span_threshold_short": horizontal_edge_span_threshold_short,
                        "horizontal_edge_span_threshold": horizontal_edge_span_threshold,
                        "horizontal_penalty_factor": horizontal_penalty_factor,
                        "intact_aspect_ratio_threshold": vertical_intact_aspect_ratio_threshold,
                        "edge_span_threshold": vertical_edge_span_threshold,
                        "edge_span_threshold_thin": vertical_edge_span_threshold_thin,
                        "defective_penalty_factor": vertical_defective_penalty_factor,
                        "applied_rules": applied_rules,
                        "temporal_matched": False,
                        "temporal_kept": None,
                        "temporal_suppressed": False,
                        "temporal_partner_image_name": None,
                        "temporal_partner_det_index": None,
                        "temporal_partner_base_conf": None,
                        "temporal_overlap_ratio": None,
                        "temporal_expected_dx": None,
                        "temporal_expected_dy": None,
                    }
                )

        if same_class_contain_suppression:
            detections = suppress_same_class_contained_boxes(
                detections=detections,
                contain_ratio_threshold=contain_ratio_threshold,
            )

        frame_entries.append(
            {
                "result": result,
                "image": image,
                "original_image": original_image,
                "image_height": image_height,
                "image_width": image_width,
                "output_name": output_name,
                "output_path": output_path,
                "detections": detections,
            }
        )

    if adjacent_frame_dedup:
        apply_adjacent_frame_dedup(
            frame_entries,
            edge_priority_rule=adjacent_frame_edge_priority_rule,
            x_overlap_threshold=adjacent_frame_x_overlap_threshold,
            delta_y=adjacent_frame_delta_y,
            height_min=adjacent_frame_height_min,
            height_max=adjacent_frame_height_max,
            height_confidence_offset_px=adjacent_frame_height_confidence_offset_px,
            non_bottom_y2_correction=adjacent_frame_non_bottom_y2_correction,
            bottom_touch_margin_px=adjacent_frame_bottom_touch_margin_px,
            x_tolerance=adjacent_frame_x_tolerance,
        )

    for frame_entry in frame_entries:
        result = frame_entry["result"]
        image = frame_entry["image"]
        original_image = frame_entry["original_image"]
        image_height = int(frame_entry["image_height"])
        image_width = int(frame_entry["image_width"])
        output_name = str(frame_entry["output_name"])
        output_path = Path(frame_entry["output_path"])
        all_detections = list(frame_entry["detections"])
        detections = [det for det in all_detections if not bool(det.get("temporal_suppressed"))]

        if render_adjacent_frame_guides and adjacent_frame_dedup:
            render_adjacent_frame_guide_lines(
                image,
                delta_y=adjacent_frame_delta_y,
                height_max=adjacent_frame_height_max,
            )

        if save_txt:
            label_path = labels_dir / f"{Path(output_name).stem}.txt"
            lines = []
            for det in detections:
                fields = [
                    str(int(det["class_id"])),
                    f"{det['x_center']:.6f}",
                    f"{det['y_center']:.6f}",
                    f"{det['width']:.6f}",
                    f"{det['height']:.6f}",
                ]
                if save_conf:
                    fields.append(f"{det['conf']:.6f}")
                lines.append(" ".join(fields))
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        for det in detections:
            color = color_for_class(int(det["class_id"]))
            x1 = max(0, min(image_width - 1, round(float(det["x1"]))))
            y1 = max(0, min(image_height - 1, round(float(det["y1"]))))
            x2 = max(0, min(image_width - 1, round(float(det["x2"]))))
            y2 = max(0, min(image_height - 1, round(float(det["y2"]))))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            conf_text = f"{float(det['conf']):.2f}"
            label = f"{int(det['class_id'])} {conf_text}"
            if hasattr(result, "names") and int(det["class_id"]) in result.names:
                label = f"{result.names[int(det['class_id'])]} {conf_text}"
            decision = det["phase2_decision"]
            if decision:
                label += f" {decision}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_top = max(0, y1 - text_height - baseline - 8)
            text_bottom = min(image_height - 1, text_top + text_height + baseline + 8)
            text_right = min(image_width - 1, x1 + text_width + 12)
            cv2.rectangle(image, (x1, text_top), (text_right, text_bottom), color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                label,
                (x1 + 6, text_bottom - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for det in all_detections:
            summary.append(
                {
                    "image_name": output_name,
                    "det_index": det["det_index"],
                    "class_id": int(det["class_id"]),
                    "base_conf": det["base_conf"],
                    "final_conf": det["conf"],
                    "edge_type": det["edge_type"],
                    "actual_vertical_edge_type": det["actual_vertical_edge_type"],
                    "phase2_decision": det["phase2_decision"],
                    "phase2_reason": det["phase2_reason"],
                    "phase2_aspect_ratio": det["phase2_aspect_ratio"],
                    "phase2_edge_span_score": det["phase2_edge_span_score"],
                    "actual_vertical_edge_span_score": det["actual_vertical_edge_span_score"],
                    "edge_span_threshold_thin": det["edge_span_threshold_thin"],
                    "horizontal_width_height_ratio": det["horizontal_width_height_ratio"],
                    "horizontal_top_gap": det["horizontal_top_gap"],
                    "horizontal_bottom_gap": det["horizontal_bottom_gap"],
                    "horizontal_min_edge_gap": det["horizontal_min_edge_gap"],
                    "horizontal_edge_span_score": det["horizontal_edge_span_score"],
                    "horizontal_rule_edge_type": det["horizontal_rule_edge_type"],
                    "temporal_matched": det["temporal_matched"],
                    "temporal_kept": det["temporal_kept"],
                    "temporal_suppressed": det["temporal_suppressed"],
                    "temporal_partner_image_name": det["temporal_partner_image_name"],
                    "temporal_partner_det_index": det["temporal_partner_det_index"],
                    "temporal_partner_base_conf": det["temporal_partner_base_conf"],
                    "temporal_overlap_ratio": det["temporal_overlap_ratio"],
                    "temporal_expected_dx": det["temporal_expected_dx"],
                    "temporal_expected_dy": det["temporal_expected_dy"],
                    "adjacent_compare_self_height": det.get("adjacent_compare_self_height"),
                    "adjacent_compare_partner_height": det.get("adjacent_compare_partner_height"),
                    "adjacent_compare_self_width": det.get("adjacent_compare_self_width"),
                    "adjacent_compare_partner_width": det.get("adjacent_compare_partner_width"),
                    "applied_rules": det["applied_rules"],
                    "xyxy": [det["x1"], det["y1"], det["x2"], det["y2"]],
                }
            )
            if penalty_hits_dir is not None:
                for rule_name in det["applied_rules"]:
                    if rule_name not in {
                        "horizontal_edge_penalty",
                        "horizontal_short_edge_penalty",
                        "vertical_defective_penalty",
                    }:
                        continue
                    preview_name = (
                        f"{Path(output_name).stem}__det{int(det['det_index']):02d}"
                        f"__cls{int(det['class_id'])}__{rule_name}.png"
                    )
                    render_penalty_preview(
                        image=original_image,
                        det=det,
                        rule_name=rule_name,
                        output_path=penalty_hits_dir / rule_name / preview_name,
                    )

        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Failed to write prediction image: {output_path}")

    (save_dir / "phase2_postprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
