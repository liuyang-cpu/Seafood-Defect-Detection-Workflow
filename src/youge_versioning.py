from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

VERSION_PATTERN = re.compile(r"version(\d+)", re.IGNORECASE)
TRAILING_VERSION_PATTERN = re.compile(r"(?:[_-])?version\d+$", re.IGNORECASE)
POSTPROCESS_VERSION_PATTERN = re.compile(r"(?:pp|postprocess)(\d+)", re.IGNORECASE)
TRAILING_POSTPROCESS_VERSION_PATTERN = re.compile(r"(?:[_-])?(?:pp|postprocess)\d+$", re.IGNORECASE)


def get_model_output_dir(repo_root: Path, species: str = "youge") -> Path:
    return repo_root / "src" / "opt" / species / "model-output"


def get_postprocess_registry_root(repo_root: Path, species: str = "youge") -> Path:
    return repo_root / "src" / "predict" / species / "postprocess-registry"


def get_postprocess_registry_dir(repo_root: Path, species: str, version: str) -> Path:
    return get_postprocess_registry_root(repo_root, species) / version


def normalize_version(value: str | int | None, width: int = 3) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return f"version{int(text):0{width}d}"

    match = VERSION_PATTERN.search(text)
    if match:
        return f"version{int(match.group(1)):0{width}d}"

    raise ValueError(f"Unsupported version value: {value}")


def extract_version_from_text(value: str | None, width: int = 3) -> str | None:
    if not value:
        return None
    match = VERSION_PATTERN.search(str(value))
    if not match:
        return None
    return f"version{int(match.group(1)):0{width}d}"


def strip_trailing_version(name: str) -> str:
    stripped = TRAILING_VERSION_PATTERN.sub("", name).rstrip("_-")
    return stripped or name


def normalize_postprocess_version(value: str | int | None, width: int = 3) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return f"pp{int(text):0{width}d}"

    match = POSTPROCESS_VERSION_PATTERN.search(text)
    if match:
        return f"pp{int(match.group(1)):0{width}d}"

    raise ValueError(f"Unsupported postprocess_version value: {value}")


def extract_postprocess_version_from_text(value: str | None, width: int = 3) -> str | None:
    if not value:
        return None
    match = POSTPROCESS_VERSION_PATTERN.search(str(value))
    if not match:
        return None
    return f"pp{int(match.group(1)):0{width}d}"


def strip_trailing_postprocess_version(name: str) -> str:
    stripped = TRAILING_POSTPROCESS_VERSION_PATTERN.sub("", name).rstrip("_-")
    return stripped or name


def build_versioned_name(base_name: str, version: str | None) -> str:
    clean_base = strip_trailing_version(base_name)
    return f"{clean_base}_{version}" if version else clean_base


def build_predict_run_name(
    base_name: str,
    version: str | None,
    postprocess_version: str | None = None,
) -> str:
    clean_name = strip_trailing_postprocess_version(base_name)
    clean_name = strip_trailing_version(clean_name)
    clean_name = strip_trailing_postprocess_version(clean_name)
    name = build_versioned_name(clean_name, version)
    clean_name = strip_trailing_postprocess_version(name)
    return f"{clean_name}_{postprocess_version}" if postprocess_version else clean_name


def compute_postprocess_fingerprint(rules: dict) -> str:
    stable_text = json.dumps(rules, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(stable_text.encode('utf-8')).hexdigest()}"


def _postprocess_version_sort_key(value: str) -> int:
    normalized = normalize_postprocess_version(value)
    if normalized is None:
        return -1
    digits = "".join(ch for ch in normalized if ch.isdigit())
    return int(digits) if digits else -1


def get_next_postprocess_version(registry_dir: Path, width: int = 3) -> str:
    highest = 0
    if registry_dir.exists():
        for path in registry_dir.iterdir():
            if not path.is_file() or path.suffix.lower() != ".json":
                continue
            normalized = extract_postprocess_version_from_text(path.stem, width=width)
            if normalized:
                highest = max(highest, _postprocess_version_sort_key(normalized))
    return f"pp{highest + 1:0{width}d}"


def load_postprocess_index(registry_dir: Path) -> dict:
    index_path = registry_dir / "index.json"
    if not index_path.exists():
        return {"recipes": {}}
    return json.loads(index_path.read_text(encoding="utf-8"))


def save_postprocess_index(registry_dir: Path, payload: dict) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_postprocess_recipe(registry_dir: Path, payload: dict) -> Path:
    registry_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = registry_dir / f"{payload['postprocess_version']}.json"
    recipe_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (registry_dir / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return recipe_path


def get_next_version(model_output_dir: Path, width: int = 3) -> str:
    highest = 0
    if model_output_dir.exists():
        for path in model_output_dir.iterdir():
            match = VERSION_PATTERN.search(path.stem)
            if match:
                highest = max(highest, int(match.group(1)))

    latest_path = model_output_dir / "latest.json"
    if latest_path.exists():
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        match = VERSION_PATTERN.search(str(payload.get("version", "")))
        if match:
            highest = max(highest, int(match.group(1)))

    return f"version{highest + 1:0{width}d}"


def load_latest_metadata(model_output_dir: Path) -> dict | None:
    latest_path = model_output_dir / "latest.json"
    if not latest_path.exists():
        return None
    return json.loads(latest_path.read_text(encoding="utf-8"))


def load_version_metadata(model_output_dir: Path, version: str) -> dict | None:
    metadata_path = model_output_dir / f"{version}.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_version_metadata(model_output_dir: Path, version: str | None) -> dict | None:
    normalized = normalize_version(version) if version else None
    if normalized:
        return load_version_metadata(model_output_dir, normalized)
    return load_latest_metadata(model_output_dir)


def register_training_artifacts(
    *,
    repo_root: Path,
    species: str,
    version: str,
    train_run_name: str,
    train_run_dir: Path,
    source_model_name: str,
    overwrite_existing: bool = False,
) -> dict:
    model_output_dir = get_model_output_dir(repo_root, species)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    best_source = train_run_dir / "weights" / "best.pt"
    last_source = train_run_dir / "weights" / "last.pt"
    if not best_source.exists():
        raise FileNotFoundError(f"Training best weights not found: {best_source}")

    model_stem = Path(source_model_name).stem
    best_target = model_output_dir / f"{model_stem}_{version}.pt"
    if best_target.exists() and not overwrite_existing:
        raise FileExistsError(f"Versioned model already exists: {best_target}")
    shutil.copy2(best_source, best_target)

    last_target = None
    if last_source.exists():
        last_target = model_output_dir / f"{model_stem}_{version}_last.pt"
        shutil.copy2(last_source, last_target)

    payload = {
        "species": species,
        "version": version,
        "train_run_name": train_run_name,
        "train_run_dir": str(train_run_dir),
        "weights": str(best_target),
        "last_weights": str(last_target) if last_target else None,
        "source_model": source_model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    (model_output_dir / f"{version}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (model_output_dir / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
