from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from common import (
    build_trial_digest,
    build_search_space_payload,
    default_optuna_config_path,
    default_optuna_studies_root,
    default_refiner_runs_root,
    default_search_space_path,
    find_repo_root,
    format_run_stamp,
    infer_param_trends,
    load_json,
    sanitize_name,
    strip_payload_paths,
    validate_refiner_response,
    write_json,
)


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_PROVIDER = "openai"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_SCHEMA_NAME = "youge_optuna_space_refiner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use an LLM to refine the next-round Optuna search space from one study.")
    parser.add_argument("--study-name", type=str, required=True, help="Study name under src/train-optuna/runs/studies.")
    parser.add_argument("--config", type=str, default=None, help="Optional LLM config JSON path.")
    parser.add_argument("--provider", type=str, default=None, choices=("openai", "deepseek"), help="LLM provider override.")
    parser.add_argument("--model", type=str, default=None, help="Model id override.")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL override.")
    parser.add_argument("--optuna-config", type=str, default=None, help="Optional current optuna config JSON path.")
    parser.add_argument("--search-space", type=str, default=None, help="Optional current search-space JSON path.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top completed trials to summarize.")
    parser.add_argument("--failed-k", type=int, default=10, help="How many failed trials to summarize.")
    parser.add_argument("--dry-run", action="store_true", help="Build payload only, do not call the API.")
    return parser.parse_args()


def load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_settings(args: argparse.Namespace, config_payload: dict) -> dict:
    provider = args.provider or config_payload.get("provider") or DEFAULT_PROVIDER
    model = args.model or config_payload.get("model")
    base_url = args.base_url or config_payload.get("base_url")
    api_key = config_payload.get("api_key")
    api_key_env = config_payload.get("api_key_env")

    if provider == "deepseek":
        model = model or DEFAULT_DEEPSEEK_MODEL
        base_url = base_url or DEFAULT_DEEPSEEK_BASE_URL
    else:
        model = model or DEFAULT_MODEL

    if not api_key and api_key_env:
        api_key = os.environ.get(str(api_key_env))
    if not api_key and provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_messages(payload: dict, *, prompts_dir: Path) -> list[dict]:
    prompt_template = read_text_file(prompts_dir / "refiner_prompt_template.txt")
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    user_text = prompt_template.replace("{{PAYLOAD_JSON}}", payload_json)
    return [{"role": "user", "content": user_text}]


def call_openai_responses(
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    schema_payload: dict,
    messages: list[dict],
) -> tuple[dict, dict]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Install it with: pip install openai") from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.responses.create(
        model=model,
        input=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": DEFAULT_SCHEMA_NAME,
                "schema": schema_payload,
                "strict": True,
            }
        },
    )
    text = getattr(response, "output_text", None)
    if not text:
        raise RuntimeError("The model response did not include output_text.")
    return json.loads(text), response.model_dump() if hasattr(response, "model_dump") else json.loads(response.json())


def call_deepseek_chat(*, api_key: str, base_url: str, model: str, messages: list[dict]) -> tuple[dict, dict]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Install it with: pip install openai") from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        stream=False,
    )
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("The DeepSeek response did not include message content.")
    return json.loads(content), response.model_dump() if hasattr(response, "model_dump") else json.loads(response.json())


def build_refiner_payload(
    *,
    study_dir: Path,
    optuna_config: dict,
    search_space_payload: dict,
    top_k: int,
    failed_k: int,
) -> dict:
    report_meta = load_json(study_dir / "reports" / "latest" / "report_meta.json")
    resolved_config = load_json(study_dir / "resolved_config.json")
    best_trial = load_json(study_dir / "best_trial.json") if (study_dir / "best_trial.json").exists() else {}
    summary_json = load_json(study_dir / "reports" / "latest" / "summary.json")
    all_rows = list(summary_json.get("rows") or [])

    top_rows = [build_trial_digest(row) for row in all_rows if row.get("state") == "COMPLETE"][: max(1, top_k)]
    failed_rows = [build_trial_digest(row) for row in all_rows if row.get("state") != "COMPLETE"][: max(0, failed_k)]
    param_trends = infer_param_trends(all_rows, top_k=top_k)

    return {
        "task_context": {
            "project": "youge",
            "workflow": "optuna-search-space-refinement",
            "dataset_shape_hint": "1536x300",
            "primary_objective": "在控制试验预算的前提下，为下一轮 Optuna 收缩或重组搜索空间。",
            "business_priorities": [
                "边缘截断目标稳定性",
                "类别一致性",
                "总体 mAP50-95"
            ],
        },
        "study_context": {
            "study_name": study_dir.name,
            "study_dir": str(study_dir),
            "trials_total": report_meta.get("trials_total"),
            "trials_complete": report_meta.get("trials_complete"),
            "metric": resolved_config.get("metric"),
            "direction": resolved_config.get("direction"),
            "current_n_trials": optuna_config.get("n_trials"),
            "current_n_startup_trials": ((optuna_config.get("sampler") or {}).get("n_startup_trials")),
            "best_trial": strip_payload_paths(best_trial),
        },
        "current_optuna_config": optuna_config,
        "current_search_space": build_search_space_payload(search_space_payload),
        "search_runtime_summary": {
            "top_trials": top_rows,
            "failed_trials": failed_rows,
            "param_trends_top_k": param_trends,
        },
        "constraints": {
            "must_stay_within_existing_search_space": True,
            "must_output_enabled_params_subset": True,
            "must_keep_current_metric": True,
            "must_keep_single_gpu_execution": True,
            "prefer_conservative_space_refinement": True,
            "max_enabled_params_next": 12,
            "max_n_trials_next": 60,
            "min_n_trials_next": 8,
        },
    }


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    module_root = script_path.parent.parent
    repo_root = find_repo_root(script_path.parent)

    config_path = Path(args.config).resolve() if args.config else module_root / "config" / "llm_space_refiner.config.json"
    config_payload = load_config(config_path)
    settings = resolve_settings(args, config_payload)

    study_name = sanitize_name(args.study_name, fallback="study")
    study_dir = default_optuna_studies_root(repo_root) / study_name
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")

    optuna_config_path = Path(args.optuna_config).resolve() if args.optuna_config else default_optuna_config_path(repo_root)
    search_space_path = Path(args.search_space).resolve() if args.search_space else default_search_space_path(repo_root)
    schema_path = module_root / "docs" / "llm_space_refiner_response_schema.json"

    optuna_config = load_json(optuna_config_path)
    search_space_payload = load_json(search_space_path)
    payload = build_refiner_payload(
        study_dir=study_dir,
        optuna_config=optuna_config,
        search_space_payload=search_space_payload,
        top_k=args.top_k,
        failed_k=args.failed_k,
    )

    run_root = default_refiner_runs_root(module_root)
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = format_run_stamp(datetime.now())
    run_dir = run_root / f"refine_{stamp}_{study_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload_path = run_dir / "refiner_payload.json"
    write_json(payload_path, payload)

    if args.dry_run:
        print(f"Refiner payload written to: {payload_path}")
        return

    if not settings["api_key"]:
        raise RuntimeError(
            f"No API key resolved for provider={settings['provider']}. "
            f"Check {config_path} and/or the related environment variable."
        )

    schema_payload = load_json(schema_path)
    messages = build_messages(payload, prompts_dir=module_root / "prompts")
    if settings["provider"] == "deepseek":
        response_payload, raw_response = call_deepseek_chat(
            api_key=str(settings["api_key"]),
            base_url=str(settings["base_url"]),
            model=str(settings["model"]),
            messages=messages,
        )
    else:
        response_payload, raw_response = call_openai_responses(
            api_key=str(settings["api_key"]),
            base_url=str(settings["base_url"]) if settings["base_url"] else None,
            model=str(settings["model"]),
            schema_payload=schema_payload,
            messages=messages,
        )

    response_path = run_dir / "refiner_response.json"
    raw_response_path = run_dir / "raw_response.json"
    meta_path = run_dir / "refiner_meta.json"
    validation_errors = validate_refiner_response(
        response=response_payload,
        schema_payload=schema_payload,
        base_optuna_config=optuna_config,
        search_space_payload=search_space_payload,
    )
    write_json(response_path, response_payload)
    write_json(raw_response_path, raw_response)
    write_json(
        meta_path,
        {
            "created_at": stamp,
            "provider": settings["provider"],
            "model": settings["model"],
            "base_url": settings["base_url"],
            "config_path": str(config_path) if config_path.exists() else None,
            "study_dir": str(study_dir),
            "optuna_config_path": str(optuna_config_path),
            "search_space_path": str(search_space_path),
            "payload_path": str(payload_path),
            "response_path": str(response_path),
            "raw_response_path": str(raw_response_path),
            "validation_errors": validation_errors,
        },
    )
    print(f"Refiner payload written to: {payload_path}")
    print(f"Refiner response written to: {response_path}")
    print(f"Raw response written to: {raw_response_path}")
    if validation_errors:
        print(f"Validation errors written in meta: {meta_path}")
        raise RuntimeError("LLM response failed local validation.")


if __name__ == "__main__":
    main()
