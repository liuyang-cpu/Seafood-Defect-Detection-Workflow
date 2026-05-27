from __future__ import annotations

import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from run_youge_search_plan import (
    get_plan_selection,
    list_plan_sources,
    run_plan,
)


class SearchPlanRunnerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.script_path = Path(__file__).resolve()
        self.plan_map: dict[str, Path] = {}
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.plan_var = tk.StringVar()
        self.start_var = tk.StringVar(value="1")
        self.end_var = tk.StringVar()
        self.max_runs_var = tk.StringVar()
        self.batch_var = tk.StringVar()
        self.workers_var = tk.StringVar()
        self.dry_run_var = tk.BooleanVar(value=False)
        self.skip_report_var = tk.BooleanVar(value=False)

        self.root.title("Youge 训练计划执行器")
        self.root.geometry("980x680")

        self._build_ui()
        self._load_plans()
        self._schedule_log_poll()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text="训练计划").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        self.plan_combo = ttk.Combobox(frame, textvariable=self.plan_var, state="readonly")
        self.plan_combo.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        self.plan_combo.bind("<<ComboboxSelected>>", self._on_plan_changed)

        button_row = ttk.Frame(frame)
        button_row.grid(row=0, column=2, sticky="e", pady=(0, 8))
        ttk.Button(button_row, text="刷新", command=self._load_plans).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="预览", command=self._preview_selection).pack(side="left")

        options = ttk.LabelFrame(frame, text="执行范围", padding=10)
        options.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        for index in range(8):
            options.columnconfigure(index, weight=1 if index in (1, 3, 5, 7) else 0)

        ttk.Label(options, text="起始序号").grid(row=0, column=0, sticky="w")
        start_entry = ttk.Entry(options, textvariable=self.start_var, width=10)
        start_entry.grid(row=0, column=1, sticky="w", padx=(6, 16))

        ttk.Label(options, text="结束序号").grid(row=0, column=2, sticky="w")
        end_entry = ttk.Entry(options, textvariable=self.end_var, width=10)
        end_entry.grid(row=0, column=3, sticky="w", padx=(6, 16))

        ttk.Label(options, text="最多执行").grid(row=0, column=4, sticky="w")
        max_entry = ttk.Entry(options, textvariable=self.max_runs_var, width=10)
        max_entry.grid(row=0, column=5, sticky="w", padx=(6, 16))

        ttk.Label(options, text="统一Batch").grid(row=1, column=0, sticky="w", pady=(10, 0))
        batch_entry = ttk.Entry(options, textvariable=self.batch_var, width=10)
        batch_entry.grid(row=1, column=1, sticky="w", padx=(6, 16), pady=(10, 0))

        ttk.Label(options, text="统一Workers").grid(row=1, column=2, sticky="w", pady=(10, 0))
        workers_entry = ttk.Entry(options, textvariable=self.workers_var, width=10)
        workers_entry.grid(row=1, column=3, sticky="w", padx=(6, 16), pady=(10, 0))

        ttk.Checkbutton(options, text="仅预演，不执行训练", variable=self.dry_run_var).grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Checkbutton(options, text="跳过汇总报告", variable=self.skip_report_var).grid(
            row=2, column=1, sticky="w", pady=(10, 0)
        )

        info_frame = ttk.LabelFrame(frame, text="计划摘要", padding=10)
        info_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        info_frame.columnconfigure(0, weight=1)
        self.summary_label = ttk.Label(info_frame, text="", justify="left")
        self.summary_label.grid(row=0, column=0, sticky="w")

        action_frame = ttk.Frame(frame)
        action_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.run_button = ttk.Button(action_frame, text="执行当前计划", command=self._start_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(action_frame, text="终止训练", command=self._stop_run, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))

        params_frame = ttk.LabelFrame(frame, text="参数预览", padding=10)
        params_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(0, 8))
        params_frame.columnconfigure(0, weight=1)
        params_frame.rowconfigure(0, weight=1)
        self.params_text = ScrolledText(params_frame, wrap="word", font=("Consolas", 10), height=14)
        self.params_text.grid(row=0, column=0, sticky="nsew")
        self.params_text.configure(state="disabled")

        self.log_text = ScrolledText(frame, wrap="word", font=("Consolas", 10))
        self.log_text.grid(row=5, column=0, columnspan=3, sticky="nsew")
        self.log_text.configure(state="disabled")

        for variable in (self.start_var, self.end_var, self.max_runs_var, self.batch_var, self.workers_var):
            variable.trace_add("write", lambda *_args: self._refresh_summary())
        self.dry_run_var.trace_add("write", lambda *_args: self._refresh_summary())
        self.skip_report_var.trace_add("write", lambda *_args: self._refresh_summary())

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_params_text(self, text: str) -> None:
        self.params_text.configure(state="normal")
        self.params_text.delete("1.0", "end")
        self.params_text.insert("1.0", text)
        self.params_text.configure(state="disabled")

    def _schedule_log_poll(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(message)
        self.root.after(150, self._schedule_log_poll)

    def _load_plans(self) -> None:
        sources = list_plan_sources(self.script_path)
        self.plan_map = {display_name: plan_dir for display_name, plan_dir in sources}
        names = list(self.plan_map)
        self.plan_combo["values"] = names
        if names:
            if self.plan_var.get() not in names:
                self.plan_var.set(names[-1])
            self._refresh_summary()
        else:
            self.plan_var.set("")
            self.summary_label.configure(text="未找到可用训练计划。")
            self._set_params_text("")

    def _parse_optional_int(self, raw: str, field_name: str) -> int | None:
        text = raw.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{field_name}必须是整数。") from exc

    def _format_override_summary(self, batch: int | None, workers: int | None) -> str:
        parts = []
        if batch is not None:
            parts.append(f"batch={batch}")
        if workers is not None:
            parts.append(f"workers={workers}")
        return "，".join(parts) if parts else "无"

    def _read_form(self) -> dict:
        plan = self.plan_var.get().strip()
        if not plan:
            raise ValueError("请先选择一个训练计划。")
        plan_path = self.plan_map.get(plan)
        if plan_path is None:
            raise ValueError("当前选择的训练计划无效，请先刷新列表。")

        start_index = self._parse_optional_int(self.start_var.get(), "起始序号")
        end_index = self._parse_optional_int(self.end_var.get(), "结束序号")
        max_runs = self._parse_optional_int(self.max_runs_var.get(), "最多执行")
        batch = self._parse_optional_int(self.batch_var.get(), "统一Batch")
        workers = self._parse_optional_int(self.workers_var.get(), "统一Workers")
        return {
            "plan": str(plan_path),
            "plan_display_name": plan,
            "start_index": start_index or 1,
            "end_index": end_index,
            "max_runs": max_runs,
            "batch": batch,
            "workers": workers,
            "dry_run": self.dry_run_var.get(),
            "skip_report": self.skip_report_var.get(),
        }

    def _refresh_summary(self) -> None:
        try:
            values = self._read_form()
            plan_dir, manifest, selected_runs = get_plan_selection(
                script_path=self.script_path,
                plan=values["plan"],
                start_index=values["start_index"],
                end_index=values["end_index"],
                max_runs=values["max_runs"],
            )
        except Exception as exc:
            self.summary_label.configure(text=f"当前选择无效：{exc}")
            self._set_params_text("")
            return

        experiments = list(manifest.get("experiments") or [])
        preview_labels = ", ".join(f"[{run['index']}] {run['label']}" for run in selected_runs[:5])
        if len(selected_runs) > 5:
            preview_labels += ", ..."
        summary = (
            f"计划名：{plan_dir.name}\n"
            f"来源：{values['plan_display_name']}\n"
            f"实验总数：{len(experiments)}\n"
            f"本次选中：{len(selected_runs)}\n"
            f"统一覆盖：{self._format_override_summary(values['batch'], values['workers'])}\n"
            f"预览：{preview_labels}"
        )
        self.summary_label.configure(text=summary)
        self._refresh_params_preview(selected_runs)

    def _on_plan_changed(self, _event: object | None = None) -> None:
        self._refresh_summary()

    def _refresh_params_preview(self, selected_runs: list[dict]) -> None:
        blocks: list[str] = []
        for run in selected_runs:
            config_path = Path(run["config_path"]).resolve()
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                payload = {"_error": str(exc)}
            blocks.append(
                f"[实验 {run['index']}] {run['label']}\n"
                f"说明: {run.get('description') or '-'}\n"
                f"配置文件: {config_path}\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            )
        self._set_params_text("\n\n" + ("\n" + ("-" * 80) + "\n\n").join(blocks) if blocks else "")

    def _preview_selection(self) -> None:
        try:
            self._refresh_summary()
            values = self._read_form()
            _plan_dir, selected_runs = run_plan(
                script_path=self.script_path,
                plan=values["plan"],
                start_index=values["start_index"],
                end_index=values["end_index"],
                max_runs=values["max_runs"],
                batch=values["batch"],
                workers=values["workers"],
                dry_run=True,
                skip_report=values["skip_report"],
                log=self.log_queue.put,
            )
            self.log_queue.put(f"预览完成。本次选中了 {len(selected_runs)} 个实验。")
        except Exception as exc:
            messagebox.showerror("预览失败", str(exc))

    def _start_run(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("任务进行中", "当前已有训练计划在执行。")
            return

        try:
            values = self._read_form()
            self._refresh_summary()
        except Exception as exc:
            messagebox.showerror("输入有误", str(exc))
            return

        self.stop_event.clear()
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.log_queue.put("=" * 80)
        self.log_queue.put("开始执行训练计划...")

        def target() -> None:
            try:
                run_plan(
                    script_path=self.script_path,
                    plan=values["plan"],
                    start_index=values["start_index"],
                    end_index=values["end_index"],
                    max_runs=values["max_runs"],
                    batch=values["batch"],
                    workers=values["workers"],
                    dry_run=values["dry_run"],
                    skip_report=values["skip_report"],
                    python_executable=sys.executable,
                    log=self.log_queue.put,
                    stop_requested=self.stop_event.is_set,
                )
                self.log_queue.put("训练计划执行完成。")
            except Exception as exc:
                self.log_queue.put(f"执行失败：{exc}")
            finally:
                self.root.after(0, self._reset_buttons_after_run)

        self.worker_thread = threading.Thread(target=target, daemon=True)
        self.worker_thread.start()

    def _stop_run(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            messagebox.showinfo("没有运行任务", "当前没有正在执行的训练计划。")
            return
        self.stop_event.set()
        self.log_queue.put("已请求终止，等待当前训练进程退出...")
        self.stop_button.configure(state="disabled")

    def _reset_buttons_after_run(self) -> None:
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    app = SearchPlanRunnerApp(root)
    app._refresh_summary()
    root.mainloop()


if __name__ == "__main__":
    main()
