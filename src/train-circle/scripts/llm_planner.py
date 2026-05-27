from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from planner_common import (
    DEFAULT_SCHEMA_NAME,
    build_planner_payload,
    choose_baseline_run,
    default_base_config_json,
    default_runs_train_dir,
    default_search_space_json,
    default_summary_csv,
    default_task_context,
    find_repo_root,
    load_history_from_run_summaries,
    load_json,
    load_summary_rows,
    write_json,
)

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_PROVIDER = "openai"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def default_planner_runs_root(script_path: Path) -> Path:
    return script_path.parent.parent / "plans" / "planner_runs"


def format_run_stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H_%M_%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call OpenAI Responses API to generate the next youge training plan.")
    parser.add_argument("--config", type=str, default=None, help="Optional planner config JSON path.")
    parser.add_argument(
        "--provider", type=str, default=None, choices=("openai", "deepseek"), help="LLM provider override."
    )
    parser.add_argument("--model", type=str, default=None, help="Model id override.")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL override for OpenAI-compatible clients.")
    parser.add_argument("--summary-csv", type=str, default=None, help="Optional summary.csv path.")
    parser.add_argument("--search-space", type=str, default=None, help="Optional search-space JSON path.")
    parser.add_argument("--base-config", type=str, default=None, help="Optional base config JSON path.")
    parser.add_argument("--baseline-version", type=str, default=None, help="Optional baseline version override.")
    parser.add_argument("--max-history", type=int, default=8, help="Maximum number of historical runs to include.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts payload only, do not call the API.")
    return parser.parse_args()


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_messages(payload: dict, *, prompts_dir: Path) -> list[dict]:
    prompt_template = read_text_file(prompts_dir / "planner_prompt_template.txt")
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    user_text = prompt_template.replace("{{PAYLOAD_JSON}}", payload_json)
    return [
        {"role": "user", "content": user_text},
    ]


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
        "api_key_env": api_key_env,
    }


def call_openai_responses(*, model: str, schema_payload: dict, messages: list[dict]) -> tuple[dict, dict]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Install it with: pip install openai") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)
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
    return json.loads(content), response.model_dump() if hasattr(response, "model_dump") else json.loads(
        response.json()
    )


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    config_path = (
        Path(args.config).resolve() if args.config else script_path.parent.parent / "config" / "llm_planner.config.json"
    )
    config_payload = load_config(config_path)
    settings = resolve_settings(args, config_payload)

    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else default_summary_csv(repo_root)
    search_space_path = Path(args.search_space).resolve() if args.search_space else default_search_space_json(repo_root)
    base_config_path = Path(args.base_config).resolve() if args.base_config else default_base_config_json(repo_root)
    output_root = Path(args.output_dir).resolve() if args.output_dir else default_planner_runs_root(script_path)
    output_root.mkdir(parents=True, exist_ok=True)

    search_space_payload = load_json(search_space_path)
    load_json(base_config_path)
    schema_payload = load_json(script_path.parent.parent / "docs" / "llm_planner_response_schema.json")

    all_history_payloads = load_history_from_run_summaries(default_runs_train_dir(repo_root), max_history=None)
    history_payloads = (
        all_history_payloads[: max(0, args.max_history)] if args.max_history is not None else list(all_history_payloads)
    )
    if not all_history_payloads:
        # fallback to summary rows if historical run summaries are unavailable
        summary_rows = load_summary_rows(summary_csv, args.max_history)
        history_payloads = [
            {
                "train_run_name": row.get("run_name"),
                "version": row.get("version"),
                "best_metrics": {
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "map50": row.get("map50"),
                    "map50_95": row.get("map50_95"),
                },
                "epochs_completed": row.get("epochs"),
                "best_epoch": row.get("best_epoch"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "train_config": row,
            }
            for row in summary_rows
        ]
        all_history_payloads = list(history_payloads)

    payload = build_planner_payload(
        task_context=default_task_context(),
        baseline_run=choose_baseline_run(all_history_payloads, args.baseline_version),
        search_space_payload=search_space_payload,
        history_payloads=history_payloads,
    )

    stamp = format_run_stamp(datetime.now())
    run_dir = output_root / f"llm_plan_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload_path = run_dir / "planner_payload.json"
    write_json(payload_path, payload)

    if args.dry_run:
        print(f"Planner payload written to: {payload_path}")
        return

    messages = build_messages(payload, prompts_dir=script_path.parent.parent / "prompts")
    if not settings["api_key"]:
        raise RuntimeError(
            f"No API key resolved for provider={settings['provider']}. "
            f"Check {config_path} and/or the related environment variable."
        )

    if settings["provider"] == "deepseek":
        planner_response, raw_response = call_deepseek_chat(
            api_key=str(settings["api_key"]),
            base_url=str(settings["base_url"]),
            model=str(settings["model"]),
            messages=messages,
        )
    else:
        planner_response, raw_response = call_openai_responses(
            model=str(settings["model"]),
            schema_payload=schema_payload,
            messages=messages,
        )

    planner_response_path = run_dir / "planner_response.json"
    raw_response_path = run_dir / "openai_response.json"
    write_json(planner_response_path, planner_response)
    write_json(raw_response_path, raw_response)
    write_json(
        run_dir / "planner_meta.json",
        {
            "created_at": stamp,
            "provider": settings["provider"],
            "model": settings["model"],
            "base_url": settings["base_url"],
            "config_path": str(config_path) if config_path.exists() else None,
            "summary_csv": str(summary_csv),
            "search_space_path": str(search_space_path),
            "base_config_path": str(base_config_path),
            "payload_path": str(payload_path),
            "planner_response_path": str(planner_response_path),
            "raw_response_path": str(raw_response_path),
        },
    )
    print(f"Planner payload written to: {payload_path}")
    print(f"Planner response written to: {planner_response_path}")
    print(f"Raw OpenAI response written to: {raw_response_path}")


if __name__ == "__main__":
    main()
