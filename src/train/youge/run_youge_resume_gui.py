from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from run_youge_resume import get_runs_train_root, list_resume_candidates, resume_run


class ResumeRunnerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.script_path = Path(__file__).resolve()
        self.runs_train_root = get_runs_train_root(self.script_path)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.run_map: dict[str, dict] = {}
        self.last_error_text = ""

        self.run_var = tk.StringVar()
        self.device_var = tk.StringVar()
        self.batch_var = tk.StringVar()
        self.imgsz_var = tk.StringVar()
        self.workers_var = tk.StringVar()
        self.advanced_var = tk.BooleanVar(value=False)

        self.root.title("Youge 断点续训")
        self.root.geometry("980x700")

        self._build_ui()
        self._load_runs()
        self._schedule_log_poll()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text="训练目录").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        self.run_combo = ttk.Combobox(frame, textvariable=self.run_var, state="readonly")
        self.run_combo.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        self.run_combo.bind("<<ComboboxSelected>>", self._on_run_changed)

        button_row = ttk.Frame(frame)
        button_row.grid(row=0, column=2, sticky="e", pady=(0, 8))
        ttk.Button(button_row, text="刷新", command=self._load_runs).pack(side="left")

        action_frame = ttk.LabelFrame(frame, text="直接操作", padding=10)
        action_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.run_button = ttk.Button(action_frame, text="开始续训当前任务", command=self._start_resume)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(action_frame, text="终止续训", command=self._stop_resume, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))

        options = ttk.LabelFrame(frame, text="高级选项", padding=10)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        for index in range(8):
            options.columnconfigure(index, weight=1 if index in (1, 3, 5) else 0)

        ttk.Label(
            options,
            text="默认会按原训练参数直接续训，只有在你想临时降显存或换设备时才需要展开并填写。",
        ).grid(row=0, column=0, columnspan=8, sticky="w")
        ttk.Checkbutton(
            options,
            text="展开高级覆盖参数",
            variable=self.advanced_var,
            command=self._toggle_advanced,
        ).grid(row=1, column=0, columnspan=8, sticky="w", pady=(8, 0))

        self.advanced_frame = ttk.Frame(options)
        self.advanced_frame.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(8, 0))
        for index in range(8):
            self.advanced_frame.columnconfigure(index, weight=1 if index in (1, 3, 5) else 0)

        ttk.Label(self.advanced_frame, text="设备").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.advanced_frame, textvariable=self.device_var, width=12).grid(
            row=0, column=1, sticky="w", padx=(6, 16)
        )

        ttk.Label(self.advanced_frame, text="Batch").grid(row=0, column=2, sticky="w")
        ttk.Entry(self.advanced_frame, textvariable=self.batch_var, width=12).grid(
            row=0, column=3, sticky="w", padx=(6, 16)
        )

        ttk.Label(self.advanced_frame, text="Imgsz").grid(row=0, column=4, sticky="w")
        ttk.Entry(self.advanced_frame, textvariable=self.imgsz_var, width=12).grid(
            row=0, column=5, sticky="w", padx=(6, 0)
        )

        ttk.Label(self.advanced_frame, text="Workers").grid(row=0, column=6, sticky="w", padx=(16, 0))
        ttk.Entry(self.advanced_frame, textvariable=self.workers_var, width=12).grid(
            row=0, column=7, sticky="w", padx=(6, 0)
        )

        info_frame = ttk.LabelFrame(frame, text="续训摘要", padding=10)
        info_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        info_frame.columnconfigure(0, weight=1)
        self.summary_label = ttk.Label(info_frame, text="", justify="left")
        self.summary_label.grid(row=0, column=0, sticky="w")

        self.preview_text = ScrolledText(frame, wrap="word", font=("Consolas", 10))
        self.preview_text.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(0, 8))
        self.preview_text.configure(state="disabled")

        self.log_text = ScrolledText(frame, wrap="word", font=("Consolas", 10), height=10)
        self.log_text.grid(row=5, column=0, columnspan=3, sticky="nsew")
        self.log_text.configure(state="disabled")
        self._toggle_advanced()

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_preview(self, text: str) -> None:
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", text)
        self.preview_text.configure(state="disabled")

    def _schedule_log_poll(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(message)
        self.root.after(150, self._schedule_log_poll)

    def _toggle_advanced(self) -> None:
        if self.advanced_var.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()

    def _show_error_dialog(self, title: str, detail: str) -> None:
        self.last_error_text = detail
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("980x620")
        dialog.transient(self.root)

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(
            frame,
            text="执行失败，完整错误信息如下。你可以滚动查看，并直接复制内容发给我。",
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        text = ScrolledText(frame, wrap="word", font=("Consolas", 10))
        text.grid(row=1, column=0, sticky="nsew")
        text.insert("1.0", detail)
        text.focus_set()

        button_row = ttk.Frame(frame)
        button_row.grid(row=2, column=0, sticky="e", pady=(10, 0))

        def copy_text() -> None:
            dialog.clipboard_clear()
            dialog.clipboard_append(detail)

        ttk.Button(button_row, text="复制错误内容", command=copy_text).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="关闭", command=dialog.destroy).pack(side="left")

    def _load_runs(self) -> None:
        candidates = list_resume_candidates(self.runs_train_root)
        self.run_map = {item["run_name"]: item for item in candidates}
        names = list(self.run_map)
        self.run_combo["values"] = names
        if names:
            preferred = next((name for name in names if self.run_map[name]["can_resume"]), names[0])
            if self.run_var.get() not in names:
                self.run_var.set(preferred)
            self._refresh_summary()
        else:
            self.run_var.set("")
            self.summary_label.configure(text="未找到可用训练目录。")
            self._set_preview("")

    def _parse_optional_int(self, raw: str, field_name: str) -> int | None:
        text = raw.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{field_name}必须是整数。") from exc

    def _read_form(self) -> dict:
        run_name = self.run_var.get().strip()
        if not run_name:
            raise ValueError("请先选择一个训练目录。")
        run_info = self.run_map.get(run_name)
        if not run_info:
            raise ValueError("当前训练目录无效，请刷新后重试。")
        return {
            "run_name": run_name,
            "run_info": run_info,
            "device": self.device_var.get().strip() or None,
            "batch": self._parse_optional_int(self.batch_var.get(), "Batch"),
            "imgsz": self._parse_optional_int(self.imgsz_var.get(), "Imgsz"),
            "workers": self._parse_optional_int(self.workers_var.get(), "Workers"),
        }

    def _refresh_summary(self) -> None:
        try:
            values = self._read_form()
        except Exception as exc:
            self.summary_label.configure(text=f"当前选择无效：{exc}")
            self._set_preview("")
            return

        info = values["run_info"]
        status = "可续训" if info["can_resume"] else "已完成/不建议续训"
        summary = (
            f"目录：{info['run_name']}\n"
            f"状态：{status}\n"
            f"当前进度：{info['last_epoch']}/{info['target_epochs']}\n"
            f"检查点：{info['resume_checkpoint']}"
        )
        self.summary_label.configure(text=summary)
        preview = (
            f"run_name: {info['run_name']}\n"
            f"run_dir: {info['run_dir']}\n"
            f"resume_checkpoint: {info['resume_checkpoint']}\n"
            f"target_epochs: {info['target_epochs']}\n"
            f"last_epoch: {info['last_epoch']}\n"
            f"default_device: {info['device']}\n"
            f"default_batch: {info['batch']}\n"
            f"default_imgsz: {info['imgsz']}\n"
            f"default_workers: {info['workers']}\n"
            f"has_run_summary: {info['has_run_summary']}\n"
            f"is_complete: {info['is_complete']}\n"
            f"can_resume: {info['can_resume']}\n"
        )
        self._set_preview(preview)

    def _on_run_changed(self, _event: object | None = None) -> None:
        self._refresh_summary()

    def _start_resume(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("任务进行中", "当前已有续训任务在执行。")
            return

        try:
            values = self._read_form()
            self._refresh_summary()
        except Exception as exc:
            messagebox.showerror("输入有误", str(exc))
            return

        if not values["run_info"]["can_resume"]:
            confirmed = messagebox.askyesno("继续确认", "该目录看起来像已完成训练，仍然要尝试续训吗？")
            if not confirmed:
                return

        self.stop_event.clear()
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.log_queue.put("=" * 80)
        self.log_queue.put("开始续训...")

        def target() -> None:
            try:
                resume_run(
                    script_path=self.script_path,
                    run_dir=values["run_name"],
                    device=values["device"],
                    batch=values["batch"],
                    imgsz=values["imgsz"],
                    workers=values["workers"],
                    python_executable=sys.executable,
                    log=self.log_queue.put,
                    stop_requested=self.stop_event.is_set,
                )
                self.log_queue.put("续训执行完成。")
            except Exception as exc:
                detail = f"{exc}\n\n{traceback.format_exc()}"
                self.log_queue.put(f"续训失败：{exc}")
                self.root.after(0, lambda: self._show_error_dialog("续训失败", detail))
            finally:
                self.root.after(0, self._reset_buttons_after_run)

        import sys

        self.worker_thread = threading.Thread(target=target, daemon=True)
        self.worker_thread.start()

    def _stop_resume(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            messagebox.showinfo("没有运行任务", "当前没有正在执行的续训任务。")
            return
        self.stop_event.set()
        self.log_queue.put("已请求终止，等待当前续训进程退出...")
        self.stop_button.configure(state="disabled")

    def _reset_buttons_after_run(self) -> None:
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._load_runs()


def main() -> None:
    root = tk.Tk()
    app = ResumeRunnerApp(root)
    app._refresh_summary()
    root.mainloop()


if __name__ == "__main__":
    main()
