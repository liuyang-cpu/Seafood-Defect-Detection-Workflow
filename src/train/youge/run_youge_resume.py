from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import yaml


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "ultralytics").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate repo root from: {start}")


def get_runs_train_root(script_path: Path) -> Path:
    return script_path.parent / "runs" / "train"


def _read_args_yaml(run_dir: Path) -> dict:
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        return {}
    with args_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_last_epoch(results_csv: Path) -> int | None:
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    try:
        return int(float(rows[-1].get("epoch", "")))
    except (TypeError, ValueError):
        return None


def inspect_run(run_dir: Path) -> dict:
    args_payload = _read_args_yaml(run_dir)
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    run_summary = run_dir / "run_summary.json"
    results_csv = run_dir / "results.csv"

    target_epochs = args_payload.get("epochs")
    last_epoch = _read_last_epoch(results_csv)
    is_complete = bool(run_summary.exists())
    if last_epoch is not None and target_epochs is not None:
        is_complete = is_complete and int(last_epoch) >= int(target_epochs)

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "resume_checkpoint": str(last_pt.resolve()) if last_pt.exists() else None,
        "best_checkpoint": str(best_pt.resolve()) if best_pt.exists() else None,
        "target_epochs": target_epochs,
        "last_epoch": last_epoch,
        "device": args_payload.get("device"),
        "batch": args_payload.get("batch"),
        "imgsz": args_payload.get("imgsz"),
        "workers": args_payload.get("workers"),
        "has_run_summary": run_summary.exists(),
        "is_complete": bool(is_complete),
        "can_resume": last_pt.exists() and not bool(is_complete),
    }


def list_resume_candidates(runs_train_root: Path) -> list[dict]:
    if not runs_train_root.exists():
        return []
    runs = []
    for run_dir in sorted((path for path in runs_train_root.iterdir() if path.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        info = inspect_run(run_dir)
        if info["resume_checkpoint"]:
            runs.append(info)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume an interrupted youge training run.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory to resume from.")
    parser.add_argument("--resume-from", type=str, default=None, help="Explicit path to weights/last.pt.")
    parser.add_argument("--list", action="store_true", help="List resumable runs and exit.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument("--batch", type=int, default=None, help="Optional batch override.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional imgsz override.")
    parser.add_argument("--workers", type=int, default=None, help="Optional workers override.")
    return parser.parse_args()


def resolve_resume_checkpoint(*, script_path: Path, run_dir: str | None, resume_from: str | None) -> Path:
    if resume_from:
        checkpoint = Path(resume_from).resolve()
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")

    if run_dir:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = get_runs_train_root(script_path) / run_path
        checkpoint = run_path.resolve() / "weights" / "last.pt"
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")

    raise ValueError("Missing --run-dir or --resume-from.")


def resume_run(
    *,
    script_path: Path,
    run_dir: str | None = None,
    resume_from: str | None = None,
    device: str | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    workers: int | None = None,
    python_executable: str | None = None,
    log: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> Path:
    log_fn = log or print
    stop_check = stop_requested or (lambda: False)
    repo_root = find_repo_root(script_path.parent)
    checkpoint = resolve_resume_checkpoint(script_path=script_path, run_dir=run_dir, resume_from=resume_from)
    train_script = script_path.with_name("train_youge.py")
    command = [python_executable or sys.executable, str(train_script), "--resume", "--resume-from", str(checkpoint)]

    if device:
        command.extend(["--device", str(device)])
    if batch is not None:
        command.extend(["--batch", str(batch)])
    if imgsz is not None:
        command.extend(["--imgsz", str(imgsz)])
    if workers is not None:
        command.extend(["--workers", str(workers)])

    log_fn(f"续训检查点：{checkpoint}")
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_lines: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            output_lines.append(text)
            log_fn(text)
            if len(output_lines) > 400:
                output_lines = output_lines[-400:]
            if stop_check():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError("续训已被用户终止。")
        return_code = process.wait()
        if return_code != 0:
            tail = "\n".join(output_lines[-80:])
            raise RuntimeError(
                "续训训练子进程失败。\n"
                f"命令：{command}\n\n"
                "最后输出：\n"
                f"{tail}"
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return checkpoint.parent.parent


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    runs_train_root = get_runs_train_root(script_path)

    if args.list:
        candidates = list_resume_candidates(runs_train_root)
        if not candidates:
            print(f"No resumable runs found under: {runs_train_root}")
            return
        for item in candidates:
            status = "可续训" if item["can_resume"] else "已完成"
            print(
                f"{item['run_name']} | 状态={status} | "
                f"epoch={item['last_epoch']}/{item['target_epochs']} | "
                f"checkpoint={item['resume_checkpoint']}"
            )
        return

    resumed_run_dir = resume_run(
        script_path=script_path,
        run_dir=args.run_dir,
        resume_from=args.resume_from,
        device=args.device,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
    )
    print(f"Resume completed for: {resumed_run_dir}")


if __name__ == "__main__":
    main()
