from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
import re
import shutil
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VERSION_PATTERN = re.compile(r"version(\d+)", re.IGNORECASE)
POSTPROCESS_RULE_KEYS = (
    "horizontal_rule",
    "same_class_contain_suppression",
    "contain_ratio_threshold",
    "adjacent_frame_dedup",
    "adjacent_frame_edge_priority_rule",
    "adjacent_frame_x_overlap_threshold",
    "adjacent_frame_delta_y",
    "adjacent_frame_height_min",
    "adjacent_frame_height_max",
    "adjacent_frame_height_confidence_offset_px",
    "adjacent_frame_non_bottom_y2_correction",
    "adjacent_frame_bottom_touch_margin_px",
    "adjacent_frame_x_tolerance",
    "horizontal_edge_touch_px",
    "horizontal_flat_ratio_threshold_short",
    "horizontal_flat_ratio_threshold",
    "horizontal_edge_span_threshold_short",
    "horizontal_edge_span_threshold",
    "horizontal_penalty_factor",
    "vertical_rule",
    "vertical_intact_aspect_ratio_threshold",
    "vertical_edge_span_threshold",
    "vertical_edge_span_threshold_thin",
    "vertical_defective_penalty_factor",
    "vertical_edge_margin_px",
)
POSTPROCESS_EXPORT_KEYS = (
    "export_penalty_hits",
    "penalty_hits_dirname",
    "render_adjacent_frame_guides",
)


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


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase2_postprocess_module(repo_root: Path):
    module_path = repo_root / "src" / "predict" / "youge" / "phase2_postprocess_common.py"
    spec = importlib.util.spec_from_file_location("phase2_postprocess_common", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load phase-2 postprocess module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction on the youge dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights.")
    parser.add_argument("--source", type=str, default=None, help="Prediction source, e.g. image, folder, video.")
    parser.add_argument("--split", type=str, choices=("train", "val", "both"), default=None, help="Dataset split shortcut under datasets/data/youge/images.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument(
        "--agnostic-nms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply class-agnostic NMS to suppress nearly identical cross-class boxes.",
    )
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name.")
    parser.add_argument("--version", type=str, default=None, help="Model version token, e.g. version001.")
    parser.add_argument(
        "--postprocess-version",
        type=str,
        default=None,
        help="Postprocess recipe version token, e.g. pp001.",
    )
    parser.add_argument(
        "--horizontal-rule",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="horizontal_rule",
        help="Enable confidence penalty for boxes touching the top or bottom edge.",
    )
    parser.add_argument(
        "--edge-penalty",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="edge_penalty",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--same-class-contain-suppression",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Suppress lower-confidence same-class boxes that are mostly contained inside a larger box.",
    )
    parser.add_argument(
        "--contain-ratio-threshold",
        type=float,
        default=None,
        help="Containment threshold defined as intersection_area / smaller_box_area.",
    )
    parser.add_argument(
        "--adjacent-frame-dedup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For adjacent frames, keep only the higher-confidence box when the same object appears twice.",
    )
    parser.add_argument(
        "--adjacent-frame-edge-priority-rule",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For adjacent-frame matches, prioritize non-edge boxes or the less-truncated edge box before falling back to confidence.",
    )
    parser.add_argument(
        "--adjacent-frame-x-overlap-threshold",
        type=float,
        default=None,
        help="Minimum horizontal overlap ratio used by adjacent-frame matching.",
    )
    parser.add_argument(
        "--adjacent-frame-delta-y",
        type=float,
        default=None,
        help="Fixed vertical offset from current-frame y1 to next-frame coordinates, in pixels.",
    )
    parser.add_argument(
        "--adjacent-frame-height-min",
        type=float,
        default=None,
        help="Minimum object height used by adjacent-frame dedup geometry.",
    )
    parser.add_argument(
        "--adjacent-frame-height-max",
        type=float,
        default=None,
        help="Maximum object height used by adjacent-frame dedup geometry.",
    )
    parser.add_argument(
        "--adjacent-frame-height-confidence-offset-px",
        type=float,
        default=None,
        help="If adjacent-frame matched box heights differ by less than this offset, fall back to base confidence.",
    )
    parser.add_argument(
        "--adjacent-frame-non-bottom-y2-correction",
        type=float,
        default=None,
        help="Extra correction added to the next-frame y2 upper bound when the current box is not touching the bottom edge.",
    )
    parser.add_argument(
        "--adjacent-frame-bottom-touch-margin-px",
        type=float,
        default=None,
        help="Bottom-edge touch margin in pixels used to decide whether the current box is truncated at the bottom.",
    )
    parser.add_argument(
        "--adjacent-frame-x-tolerance",
        type=float,
        default=None,
        help="Maximum allowed horizontal edge drift in pixels for adjacent-frame dedup.",
    )
    parser.add_argument(
        "--adjacent-frame-previous-min-y-center",
        type=float,
        default=None,
        help="Adjacent-frame dedup only considers previous-frame boxes whose normalized y_center is at least this value.",
    )
    parser.add_argument(
        "--adjacent-frame-next-max-y-center",
        type=float,
        default=None,
        help="Adjacent-frame dedup only considers next-frame boxes whose normalized y_center is at most this value.",
    )
    parser.add_argument(
        "--horizontal-edge-touch-px",
        "--edge-touch-px",
        dest="horizontal_edge_touch_px",
        type=int,
        default=None,
        help="Top/bottom edge-touch threshold in pixels for the horizontal penalty rule.",
    )
    parser.add_argument(
        "--horizontal-flat-ratio-threshold-short",
        dest="horizontal_flat_ratio_threshold_short",
        type=float,
        default=None,
        help="Lower bound for the second horizontal penalty band.",
    )
    parser.add_argument(
        "--horizontal-flat-ratio-threshold",
        "--flat-ratio-threshold",
        dest="horizontal_flat_ratio_threshold",
        type=float,
        default=None,
        help="Minimum width/height ratio for a box to be considered abnormally flat in the horizontal penalty rule.",
    )
    parser.add_argument(
        "--horizontal-edge-span-threshold-short",
        dest="horizontal_edge_span_threshold_short",
        type=float,
        default=None,
        help="Edge-span threshold used by the second horizontal penalty band.",
    )
    parser.add_argument(
        "--horizontal-edge-span-threshold",
        dest="horizontal_edge_span_threshold",
        type=float,
        default=None,
        help="Minimum edge-span score required before the horizontal penalty rule can trigger.",
    )
    parser.add_argument(
        "--horizontal-penalty-factor",
        "--edge-penalty-factor",
        dest="horizontal_penalty_factor",
        type=float,
        default=None,
        help="Confidence multiplier applied when the horizontal penalty rule triggers.",
    )
    parser.add_argument(
        "--vertical-rule",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable rule-based intact/defective judgement for top/bottom edge boxes.",
    )
    parser.add_argument(
        "--vertical-intact-aspect-ratio-threshold",
        "--intact-aspect-ratio-threshold",
        dest="vertical_intact_aspect_ratio_threshold",
        type=float,
        default=None,
        help="If h/w >= this value, force the edge box into intact.",
    )
    parser.add_argument(
        "--vertical-edge-span-threshold",
        "--edge-span-threshold",
        dest="vertical_edge_span_threshold",
        type=float,
        default=None,
        help="If intact override does not trigger, edge span score >= this value means defective.",
    )
    parser.add_argument(
        "--vertical-edge-span-threshold-thin",
        dest="vertical_edge_span_threshold_thin",
        type=float,
        default=None,
        help="Additional thin-box escape guard. Even when h/w reaches the intact override threshold, edge span score must stay below this value.",
    )
    parser.add_argument(
        "--vertical-defective-penalty-factor",
        "--defective-penalty-factor",
        dest="vertical_defective_penalty_factor",
        type=float,
        default=None,
        help="Confidence multiplier applied when the rule-based judge marks a box as defective.",
    )
    parser.add_argument(
        "--vertical-edge-margin-px",
        dest="vertical_edge_margin_px",
        type=float,
        default=None,
        help="Top/bottom edge margin used by the vertical rule.",
    )
    parser.add_argument(
        "--export-penalty-hits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export review images for every box that triggers a confidence penalty rule.",
    )
    parser.add_argument(
        "--penalty-hits-dirname",
        type=str,
        default=None,
        help="Directory name under the prediction output used to store penalty-hit previews.",
    )
    parser.add_argument(
        "--render-adjacent-frame-guides",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render the three adjacent-frame scanning guide lines on output images.",
    )
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory.")
    parser.add_argument("--save-txt", action="store_true", help="Save predictions to txt files.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in txt outputs.")
    return parser.parse_args()


def load_predict_config(config_path: Path | None) -> dict:
    defaults = {
        "weights": None,
        "source": None,
        "split": "val",
        "imgsz": None,
        "device": "0",
        "conf": 0.25,
        "iou": 0.7,
        "agnostic_nms": False,
        "project": "runs/predict",
        "name": "youge_predict",
        "version": None,
        "postprocess_version": None,
        "horizontal_rule": False,
        "adjacent_frame_dedup": False,
        "vertical_rule": True,
        "same_class_contain_suppression": False,
        "contain_ratio_threshold": 0.85,
        "adjacent_frame_x_overlap_threshold": 0.35,
        "adjacent_frame_edge_priority_rule": False,
        "adjacent_frame_delta_y": 250.0,
        "adjacent_frame_height_min": 35.64,
        "adjacent_frame_height_max": 119.95,
        "adjacent_frame_height_confidence_offset_px": 1.0,
        "adjacent_frame_non_bottom_y2_correction": 10.0,
        "adjacent_frame_bottom_touch_margin_px": 3.0,
        "adjacent_frame_x_tolerance": 12.0,
        "render_adjacent_frame_guides": False,
        "horizontal_edge_touch_px": 1,
        "horizontal_flat_ratio_threshold_short": 1.65,
        "horizontal_flat_ratio_threshold": 2.0,
        "horizontal_edge_span_threshold_short": 0.7,
        "horizontal_edge_span_threshold": 0.0,
        "horizontal_penalty_factor": 0.5,
        "vertical_intact_aspect_ratio_threshold": 1.1,
        "vertical_edge_span_threshold": 0.74,
        "vertical_edge_span_threshold_thin": 1.0,
        "vertical_defective_penalty_factor": 0.5,
        "vertical_edge_margin_px": 0.0,
        "export_penalty_hits": True,
        "penalty_hits_dirname": "penalty_hits",
        "exist_ok": True,
        "save_txt": False,
        "save_conf": False,
    }

    if config_path is None:
        return defaults

    with config_path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    legacy_map = {
        "edge_penalty": "horizontal_rule",
        "edge_touch_px": "horizontal_edge_touch_px",
        "flat_ratio_threshold": "horizontal_flat_ratio_threshold",
        "edge_penalty_factor": "horizontal_penalty_factor",
        "intact_aspect_ratio_threshold": "vertical_intact_aspect_ratio_threshold",
        "edge_span_threshold": "vertical_edge_span_threshold",
        "defective_penalty_factor": "vertical_defective_penalty_factor",
    }
    for legacy_key, new_key in legacy_map.items():
        if legacy_key in user_config and new_key not in user_config:
            user_config[new_key] = user_config[legacy_key]

    defaults.update(user_config)
    return defaults


def resolve_inference_imgsz(
    *,
    configured_imgsz: int | None,
    version_meta: dict | None,
) -> tuple[int, str]:
    if configured_imgsz is not None:
        return int(configured_imgsz), "config"

    if version_meta is not None:
        train_run_dir_value = version_meta.get("train_run_dir")
        if train_run_dir_value:
            run_summary_path = Path(str(train_run_dir_value)) / "run_summary.json"
            run_summary = load_json_if_exists(run_summary_path)
            if isinstance(run_summary, dict):
                train_config = run_summary.get("train_config")
                if isinstance(train_config, dict) and train_config.get("imgsz") is not None:
                    return int(train_config["imgsz"]), f"train_run_summary:{run_summary_path}"

        metadata_train_config = version_meta.get("train_config")
        if isinstance(metadata_train_config, dict) and metadata_train_config.get("imgsz") is not None:
            return int(metadata_train_config["imgsz"]), "model_metadata.train_config"

    return 640, "default"


def merge_args_into_config(args: argparse.Namespace, config: dict) -> dict:
    for key in (
        "weights",
        "source",
        "split",
        "imgsz",
        "device",
        "conf",
        "iou",
        "agnostic_nms",
        "project",
        "name",
        "version",
        "postprocess_version",
        "horizontal_rule",
        "same_class_contain_suppression",
        "contain_ratio_threshold",
        "adjacent_frame_dedup",
        "adjacent_frame_edge_priority_rule",
        "adjacent_frame_x_overlap_threshold",
        "adjacent_frame_delta_y",
        "adjacent_frame_height_min",
        "adjacent_frame_height_max",
        "adjacent_frame_height_confidence_offset_px",
        "adjacent_frame_non_bottom_y2_correction",
        "adjacent_frame_bottom_touch_margin_px",
        "adjacent_frame_x_tolerance",
        "horizontal_edge_touch_px",
        "horizontal_flat_ratio_threshold_short",
        "horizontal_flat_ratio_threshold",
        "horizontal_edge_span_threshold_short",
        "horizontal_edge_span_threshold",
        "horizontal_penalty_factor",
        "vertical_rule",
        "vertical_intact_aspect_ratio_threshold",
        "vertical_edge_span_threshold",
        "vertical_edge_span_threshold_thin",
        "vertical_defective_penalty_factor",
        "vertical_edge_margin_px",
        "export_penalty_hits",
        "penalty_hits_dirname",
        "render_adjacent_frame_guides",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if getattr(args, "edge_penalty", None) is not None and getattr(args, "horizontal_rule", None) is None:
        config["horizontal_rule"] = args.edge_penalty

    if args.exist_ok:
        config["exist_ok"] = True
    if args.save_txt:
        config["save_txt"] = True
    if args.save_conf:
        config["save_conf"] = True
    return config


def collect_cli_overrides(args: argparse.Namespace) -> dict:
    overrides: dict[str, object] = {}
    for key in (
        "config",
        "weights",
        "source",
        "split",
        "imgsz",
        "device",
        "conf",
        "iou",
        "agnostic_nms",
        "project",
        "name",
        "version",
        "postprocess_version",
        "horizontal_rule",
        "same_class_contain_suppression",
        "contain_ratio_threshold",
        "adjacent_frame_dedup",
        "adjacent_frame_edge_priority_rule",
        "adjacent_frame_x_overlap_threshold",
        "adjacent_frame_delta_y",
        "adjacent_frame_height_min",
        "adjacent_frame_height_max",
        "adjacent_frame_height_confidence_offset_px",
        "adjacent_frame_non_bottom_y2_correction",
        "adjacent_frame_bottom_touch_margin_px",
        "adjacent_frame_x_tolerance",
        "horizontal_edge_touch_px",
        "horizontal_flat_ratio_threshold_short",
        "horizontal_flat_ratio_threshold",
        "horizontal_edge_span_threshold_short",
        "horizontal_edge_span_threshold",
        "horizontal_penalty_factor",
        "vertical_rule",
        "vertical_intact_aspect_ratio_threshold",
        "vertical_edge_span_threshold",
        "vertical_edge_span_threshold_thin",
        "vertical_defective_penalty_factor",
        "vertical_edge_margin_px",
        "export_penalty_hits",
        "penalty_hits_dirname",
        "render_adjacent_frame_guides",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value

    if getattr(args, "edge_penalty", None) is not None and getattr(args, "horizontal_rule", None) is None:
        overrides["horizontal_rule"] = args.edge_penalty

    if args.exist_ok:
        overrides["exist_ok"] = True
    if args.save_txt:
        overrides["save_txt"] = True
    if args.save_conf:
        overrides["save_conf"] = True
    return overrides


def collect_source_images(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_versioned_weight_file(model_output_dir: Path, version: str | None = None) -> tuple[Path, str] | None:
    if not model_output_dir.exists():
        return None

    normalized_version = str(version).lower() if version else None
    candidates: list[tuple[int, Path, str]] = []
    for path in model_output_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".pt":
            continue
        lower_stem = path.stem.lower()
        if lower_stem.endswith("_last"):
            continue
        match = VERSION_PATTERN.search(path.stem)
        if not match:
            continue
        current_version = f"version{int(match.group(1)):03d}"
        if normalized_version and current_version.lower() != normalized_version:
            continue
        candidates.append((int(match.group(1)), path, current_version))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, selected_path, selected_version = candidates[0]
    return selected_path.resolve(), selected_version


def find_versioned_train_weight_file(train_runs_dir: Path, version: str | None = None) -> tuple[Path, str] | None:
    if not train_runs_dir.exists():
        return None

    normalized_version = str(version).lower() if version else None
    candidates: list[tuple[int, Path, str]] = []
    for path in train_runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = VERSION_PATTERN.search(path.name)
        if not match:
            continue
        current_version = f"version{int(match.group(1)):03d}"
        if normalized_version and current_version.lower() != normalized_version:
            continue
        best_path = path / "weights" / "best.pt"
        if not best_path.exists():
            continue
        candidates.append((int(match.group(1)), best_path, current_version))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, selected_path, selected_version = candidates[0]
    return selected_path.resolve(), selected_version


def resolve_source_input(dataset_root: Path, source_value: str | None, split_value: str, save_dir: Path) -> str:
    images_root = dataset_root / "images"
    shortcut = (source_value or split_value or "val").strip().lower()
    if shortcut in {"train", "val"}:
        source_dir = images_root / shortcut
        if not source_dir.exists():
            raise FileNotFoundError(f"Prediction source not found: {source_dir}")
        return str(source_dir)
    if shortcut == "both":
        source_manifest = save_dir / "source_manifest.txt"
        entries = [images_root / "train", images_root / "val"]
        missing = [str(path) for path in entries if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Prediction source not found: {', '.join(missing)}")
        image_paths: list[str] = []
        for directory in entries:
            image_paths.extend(str(path) for path in collect_source_images(directory))
        if not image_paths:
            raise FileNotFoundError(f"No source images found under: {', '.join(str(path) for path in entries)}")
        source_manifest.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        return str(source_manifest)

    source_path = Path(source_value).resolve() if source_value else images_root / split_value
    if not source_path.exists():
        raise FileNotFoundError(f"Prediction source not found: {source_path}")
    return str(source_path)


def build_postprocess_rules(config: dict) -> dict:
    return {key: config[key] for key in POSTPROCESS_RULE_KEYS}


def build_postprocess_export_options(config: dict) -> dict:
    return {key: config[key] for key in POSTPROCESS_EXPORT_KEYS}


def resolve_postprocess_recipe(
    *,
    repo_root: Path,
    species: str,
    model_version: str,
    requested_postprocess_version: str | None,
    rules: dict,
    export_options: dict,
    save_dir_name: str,
    weights_path: Path,
    source_hint: str | None,
    compute_postprocess_fingerprint,
    get_next_postprocess_version,
    get_postprocess_registry_dir,
    load_postprocess_index,
    save_postprocess_index,
    save_postprocess_recipe,
) -> dict:
    registry_dir = get_postprocess_registry_dir(repo_root, species, model_version)
    index_payload = load_postprocess_index(registry_dir)
    recipes = index_payload.setdefault("recipes", {})
    fingerprint = compute_postprocess_fingerprint(rules)

    existing_version = recipes.get(fingerprint)
    if existing_version and requested_postprocess_version and existing_version != requested_postprocess_version:
        raise ValueError(
            f"Postprocess rules already registered as {existing_version}, which conflicts with requested "
            f"postprocess_version={requested_postprocess_version}."
        )

    if requested_postprocess_version:
        recipe_version = requested_postprocess_version
        recipe_path = registry_dir / f"{recipe_version}.json"
        reused = True
        if recipe_path.exists():
            recipe_payload = json.loads(recipe_path.read_text(encoding="utf-8"))
            if recipe_payload.get("fingerprint") != fingerprint:
                raise ValueError(
                    f"Requested postprocess_version={recipe_version} already exists with different rules: {recipe_path}"
                )
        else:
            recipe_payload = {
                "species": species,
                "model_version": model_version,
                "postprocess_version": recipe_version,
                "fingerprint": fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rules": rules,
                "export_options": export_options,
                "examples": {
                    "predict_run_name": save_dir_name,
                    "weights": str(weights_path),
                    "source": source_hint,
                },
            }
            save_postprocess_recipe(registry_dir, recipe_payload)
            reused = False
        recipes[fingerprint] = recipe_version
        index_payload["model_version"] = model_version
        index_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_postprocess_index(registry_dir, index_payload)
        return {
            "postprocess_version": recipe_version,
            "fingerprint": fingerprint,
            "registry_dir": registry_dir,
            "recipe_path": recipe_path,
            "recipe_payload": recipe_payload,
            "reused": reused,
        }

    if existing_version:
        recipe_path = registry_dir / f"{existing_version}.json"
        if recipe_path.exists():
            recipe_payload = json.loads(recipe_path.read_text(encoding="utf-8"))
        else:
            recipe_payload = {
                "species": species,
                "model_version": model_version,
                "postprocess_version": existing_version,
                "fingerprint": fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rules": rules,
                "export_options": export_options,
                "examples": {
                    "predict_run_name": save_dir_name,
                    "weights": str(weights_path),
                    "source": source_hint,
                },
            }
            recipe_path = save_postprocess_recipe(registry_dir, recipe_payload)
        return {
            "postprocess_version": existing_version,
            "fingerprint": fingerprint,
            "registry_dir": registry_dir,
            "recipe_path": recipe_path,
            "recipe_payload": recipe_payload,
            "reused": True,
        }

    recipe_version = get_next_postprocess_version(registry_dir)
    recipe_payload = {
        "species": species,
        "model_version": model_version,
        "postprocess_version": recipe_version,
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rules": rules,
        "export_options": export_options,
        "examples": {
            "predict_run_name": save_dir_name,
            "weights": str(weights_path),
            "source": source_hint,
        },
    }
    recipe_path = save_postprocess_recipe(registry_dir, recipe_payload)
    recipes[fingerprint] = recipe_version
    index_payload["model_version"] = model_version
    index_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_postprocess_index(registry_dir, index_payload)
    return {
        "postprocess_version": recipe_version,
        "fingerprint": fingerprint,
        "registry_dir": registry_dir,
        "recipe_path": recipe_path,
        "recipe_payload": recipe_payload,
        "reused": False,
    }


def main() -> None:
    args = parse_args()
    cli_overrides = collect_cli_overrides(args)

    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from ultralytics import YOLO
    from src.youge_versioning import (
        build_predict_run_name,
        compute_postprocess_fingerprint,
        extract_version_from_text,
        get_model_output_dir,
        get_next_postprocess_version,
        get_postprocess_registry_dir,
        load_version_metadata,
        load_postprocess_index,
        normalize_version,
        normalize_postprocess_version,
        save_postprocess_index,
        save_postprocess_recipe,
    )
    phase2_postprocess = load_phase2_postprocess_module(repo_root)

    dataset_root = repo_root / "datasets" / "data" / "youge"
    config_path = Path(args.config).resolve() if args.config else script_path.with_name("predict_youge.json")
    predict_config = load_predict_config(config_path if config_path.exists() else None)
    predict_config = merge_args_into_config(args, predict_config)
    model_output_dir = get_model_output_dir(repo_root, "youge")
    train_runs_dir = repo_root / "src" / "train" / "youge" / "runs" / "train"
    version = normalize_version(predict_config.get("version"))
    requested_postprocess_version = normalize_postprocess_version(predict_config.get("postprocess_version"))
    version_meta = None
    if predict_config["weights"]:
        weights_path = Path(predict_config["weights"]).resolve()
        version = version or extract_version_from_text(weights_path.stem) or extract_version_from_text(str(weights_path))
    else:
        selected = find_versioned_train_weight_file(train_runs_dir, version)
        if selected is None:
            selected = find_versioned_weight_file(model_output_dir, version)
        if selected is None:
            if version:
                raise FileNotFoundError(
                    f"No versioned model weights found for {version} in: {train_runs_dir} or {model_output_dir}"
                )
            raise FileNotFoundError(f"No versioned model weights found in: {train_runs_dir} or {model_output_dir}")
        weights_path, version = selected
        version_meta = load_version_metadata(model_output_dir, version)

    if version_meta is None and version:
        version_meta = load_version_metadata(model_output_dir, version)

    resolved_imgsz, imgsz_source = resolve_inference_imgsz(
        configured_imgsz=predict_config.get("imgsz"),
        version_meta=version_meta,
    )
    predict_config["imgsz"] = resolved_imgsz

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if int(predict_config["horizontal_edge_touch_px"]) < 0:
        raise ValueError("horizontal_edge_touch_px must be >= 0")
    if float(predict_config["horizontal_flat_ratio_threshold_short"]) <= 0:
        raise ValueError("horizontal_flat_ratio_threshold_short must be > 0")
    if float(predict_config["horizontal_flat_ratio_threshold"]) <= 0:
        raise ValueError("horizontal_flat_ratio_threshold must be > 0")
    if float(predict_config["horizontal_flat_ratio_threshold_short"]) >= float(predict_config["horizontal_flat_ratio_threshold"]):
        raise ValueError("horizontal_flat_ratio_threshold_short must be < horizontal_flat_ratio_threshold")
    if not (0.0 <= float(predict_config["horizontal_edge_span_threshold_short"]) <= 1.0):
        raise ValueError("horizontal_edge_span_threshold_short must be in the range [0, 1]")
    if not (0.0 <= float(predict_config["horizontal_edge_span_threshold"]) <= 1.0):
        raise ValueError("horizontal_edge_span_threshold must be in the range [0, 1]")
    if not (0.0 < float(predict_config["horizontal_penalty_factor"]) <= 1.0):
        raise ValueError("horizontal_penalty_factor must be in the range (0, 1]")
    if not (0.0 <= float(predict_config["contain_ratio_threshold"]) <= 1.0):
        raise ValueError("contain_ratio_threshold must be in the range [0, 1]")
    if not (0.0 <= float(predict_config["adjacent_frame_x_overlap_threshold"]) <= 1.0):
        raise ValueError("adjacent_frame_x_overlap_threshold must be in the range [0, 1]")
    if float(predict_config["adjacent_frame_delta_y"]) <= 0:
        raise ValueError("adjacent_frame_delta_y must be > 0")
    if float(predict_config["adjacent_frame_height_min"]) <= 0:
        raise ValueError("adjacent_frame_height_min must be > 0")
    if float(predict_config["adjacent_frame_height_max"]) <= 0:
        raise ValueError("adjacent_frame_height_max must be > 0")
    if float(predict_config["adjacent_frame_height_min"]) > float(predict_config["adjacent_frame_height_max"]):
        raise ValueError("adjacent_frame_height_min must be <= adjacent_frame_height_max")
    if float(predict_config["adjacent_frame_x_tolerance"]) < 0:
        raise ValueError("adjacent_frame_x_tolerance must be >= 0")
    if float(predict_config["vertical_intact_aspect_ratio_threshold"]) <= 0:
        raise ValueError("vertical_intact_aspect_ratio_threshold must be > 0")
    if not (0.0 <= float(predict_config["vertical_edge_span_threshold"]) <= 1.0):
        raise ValueError("vertical_edge_span_threshold must be in the range [0, 1]")
    if not (0.0 <= float(predict_config["vertical_edge_span_threshold_thin"]) <= 1.0):
        raise ValueError("vertical_edge_span_threshold_thin must be in the range [0, 1]")
    if not (0.0 < float(predict_config["vertical_defective_penalty_factor"]) <= 1.0):
        raise ValueError("vertical_defective_penalty_factor must be in the range (0, 1]")
    if float(predict_config["vertical_edge_margin_px"]) < 0:
        raise ValueError("vertical_edge_margin_px must be >= 0")

    postprocess_rules = build_postprocess_rules(predict_config)
    postprocess_export_options = build_postprocess_export_options(predict_config)
    print(f"Using config: {config_path}")
    if cli_overrides:
        print(f"CLI overrides: {json.dumps(cli_overrides, ensure_ascii=False, sort_keys=True)}")
    effective_inference_params = {
        "imgsz": int(predict_config["imgsz"]),
        "conf": float(predict_config["conf"]),
        "iou": float(predict_config["iou"]),
        "agnostic_nms": bool(predict_config.get("agnostic_nms", False)),
        "save_txt": bool(predict_config.get("save_txt", False)),
        "save_conf": bool(predict_config.get("save_conf", False)),
    }
    print(f"Resolved inference imgsz from: {imgsz_source}")
    print(f"Effective inference params: {json.dumps(effective_inference_params, ensure_ascii=False, sort_keys=True)}")
    print(f"Effective postprocess rules: {json.dumps(postprocess_rules, ensure_ascii=False, sort_keys=True)}")
    provisional_name = build_predict_run_name(str(predict_config["name"]), version, requested_postprocess_version)
    recipe_info = resolve_postprocess_recipe(
        repo_root=repo_root,
        species="youge",
        model_version=version,
        requested_postprocess_version=requested_postprocess_version,
        rules=postprocess_rules,
        export_options=postprocess_export_options,
        save_dir_name=provisional_name,
        weights_path=weights_path,
        source_hint=predict_config.get("source"),
        compute_postprocess_fingerprint=compute_postprocess_fingerprint,
        get_next_postprocess_version=get_next_postprocess_version,
        get_postprocess_registry_dir=get_postprocess_registry_dir,
        load_postprocess_index=load_postprocess_index,
        save_postprocess_index=save_postprocess_index,
        save_postprocess_recipe=save_postprocess_recipe,
    )
    postprocess_version = recipe_info["postprocess_version"]

    predict_config["version"] = version
    predict_config["postprocess_version"] = postprocess_version
    predict_config["model_version"] = version
    predict_config["name"] = build_predict_run_name(str(predict_config["name"]), version, postprocess_version)
    recipe_predict_name = recipe_info["recipe_payload"].setdefault("examples", {}).get("predict_run_name")
    if recipe_predict_name != predict_config["name"]:
        recipe_info["recipe_payload"]["examples"]["predict_run_name"] = predict_config["name"]
        recipe_info["recipe_path"] = save_postprocess_recipe(recipe_info["registry_dir"], recipe_info["recipe_payload"])
    predict_config["project"] = resolve_output_dir(script_path.parent, predict_config["project"])
    save_dir = Path(predict_config["project"]) / predict_config["name"]
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve_source_input(
        dataset_root=dataset_root,
        source_value=predict_config["source"],
        split_value=str(predict_config.get("split") or "val").lower(),
        save_dir=save_dir,
    )

    run_config = dict(predict_config)
    run_config["model_version"] = version
    run_config["postprocess_version"] = postprocess_version
    run_config["postprocess_fingerprint"] = recipe_info["fingerprint"]
    run_config["postprocess_recipe_path"] = str(recipe_info["recipe_path"])
    run_config["postprocess_registry_dir"] = str(recipe_info["registry_dir"])
    run_config["postprocess_recipe_reused"] = bool(recipe_info["reused"])
    run_config["postprocess_rules"] = postprocess_rules
    run_config["postprocess_export_options"] = postprocess_export_options
    run_config["config_path"] = str(config_path)
    run_config["cli_overrides"] = cli_overrides
    run_config["weights"] = str(weights_path)
    run_config["source"] = source_path
    run_config["save_dir"] = str(save_dir)
    run_config["imgsz_source"] = imgsz_source
    if version_meta is not None:
        run_config["model_metadata"] = version_meta
    with (save_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    with (save_dir / "postprocess_recipe.json").open("w", encoding="utf-8") as f:
        json.dump(recipe_info["recipe_payload"], f, ensure_ascii=False, indent=2)

    model = YOLO(str(weights_path))
    if (
        predict_config["horizontal_rule"]
        or predict_config["vertical_rule"]
        or predict_config["same_class_contain_suppression"]
        or predict_config["adjacent_frame_dedup"]
    ):
        results = model.predict(
            source=source_path,
            imgsz=predict_config["imgsz"],
            device=predict_config["device"],
            conf=predict_config["conf"],
            iou=predict_config["iou"],
            agnostic_nms=bool(predict_config.get("agnostic_nms", False)),
            project=predict_config["project"],
            name=predict_config["name"],
            exist_ok=predict_config["exist_ok"],
            save=False,
            save_txt=False,
            save_conf=False,
        )
        phase2_postprocess.save_adjusted_predictions(
            results=results,
            save_dir=save_dir,
            save_txt=predict_config["save_txt"],
            save_conf=predict_config["save_conf"],
            horizontal_penalty_enabled=bool(predict_config["horizontal_rule"]),
            same_class_contain_suppression=bool(predict_config["same_class_contain_suppression"]),
            contain_ratio_threshold=float(predict_config["contain_ratio_threshold"]),
            adjacent_frame_dedup=bool(predict_config["adjacent_frame_dedup"]),
            adjacent_frame_edge_priority_rule=bool(predict_config["adjacent_frame_edge_priority_rule"]),
            adjacent_frame_x_overlap_threshold=float(predict_config["adjacent_frame_x_overlap_threshold"]),
            adjacent_frame_delta_y=float(predict_config["adjacent_frame_delta_y"]),
            adjacent_frame_height_min=float(predict_config["adjacent_frame_height_min"]),
            adjacent_frame_height_max=float(predict_config["adjacent_frame_height_max"]),
            adjacent_frame_height_confidence_offset_px=float(
                predict_config["adjacent_frame_height_confidence_offset_px"]
            ),
            adjacent_frame_non_bottom_y2_correction=float(predict_config["adjacent_frame_non_bottom_y2_correction"]),
            adjacent_frame_bottom_touch_margin_px=float(predict_config["adjacent_frame_bottom_touch_margin_px"]),
            adjacent_frame_x_tolerance=float(predict_config["adjacent_frame_x_tolerance"]),
            horizontal_edge_touch_px=int(predict_config["horizontal_edge_touch_px"]),
            horizontal_flat_ratio_threshold_short=float(predict_config["horizontal_flat_ratio_threshold_short"]),
            horizontal_flat_ratio_threshold=float(predict_config["horizontal_flat_ratio_threshold"]),
            horizontal_edge_span_threshold_short=float(predict_config["horizontal_edge_span_threshold_short"]),
            horizontal_edge_span_threshold=float(predict_config["horizontal_edge_span_threshold"]),
            horizontal_penalty_factor=float(predict_config["horizontal_penalty_factor"]),
            vertical_rule_enabled=bool(predict_config["vertical_rule"]),
            vertical_intact_aspect_ratio_threshold=float(predict_config["vertical_intact_aspect_ratio_threshold"]),
            vertical_edge_span_threshold=float(predict_config["vertical_edge_span_threshold"]),
            vertical_edge_span_threshold_thin=float(predict_config["vertical_edge_span_threshold_thin"]),
            vertical_defective_penalty_factor=float(predict_config["vertical_defective_penalty_factor"]),
            vertical_edge_margin_px=float(predict_config["vertical_edge_margin_px"]),
            export_penalty_hits=bool(predict_config.get("export_penalty_hits", True)),
            penalty_hits_dirname=str(predict_config.get("penalty_hits_dirname", "penalty_hits")),
            render_adjacent_frame_guides=bool(predict_config.get("render_adjacent_frame_guides", False)),
        )
    else:
        results = model.predict(
            source=source_path,
            imgsz=predict_config["imgsz"],
            device=predict_config["device"],
            conf=predict_config["conf"],
            iou=predict_config["iou"],
            agnostic_nms=bool(predict_config.get("agnostic_nms", False)),
            project=predict_config["project"],
            name=predict_config["name"],
            exist_ok=predict_config["exist_ok"],
            save=True,
            save_txt=predict_config["save_txt"],
            save_conf=predict_config["save_conf"],
        )

    save_dir = results[0].save_dir if results else save_dir
    print(
        f"Prediction finished. Version={version or 'unversioned'}, postprocess_version={postprocess_version}. "
        f"Results saved to: {save_dir}"
    )


if __name__ == "__main__":
    main()
