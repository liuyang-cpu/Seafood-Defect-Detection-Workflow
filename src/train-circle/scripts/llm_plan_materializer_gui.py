from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from llm_plan_materializer import materialize_from_paths


def list_plan_dirs(plans_root: Path) -> list[Path]:
    candidates: list[Path] = []
    if plans_root.exists():
        candidates.extend(path for path in plans_root.iterdir() if path.is_dir() and path.name.startswith("llm_plan_"))
    legacy_root = plans_root.parent
    if legacy_root.exists() and legacy_root != plans_root:
        candidates.extend(path for path in legacy_root.iterdir() if path.is_dir() and path.name.startswith("llm_plan_"))
    unique = {path.resolve(): path.resolve() for path in candidates}
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime, reverse=True)


class MaterializerGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LLM 训练计划转换器")
        self.root.geometry("980x760")

        self.script_dir = Path(__file__).resolve().parent
        self.module_root = self.script_dir.parent
        self.planner_runs_root = self.module_root / "plans" / "planner_runs"
        self.materialized_root = self.module_root / "plans" / "materialized_plans"

        self.plan_var = tk.StringVar()
        self.response_var = tk.StringVar()
        self.payload_var = tk.StringVar()
        self.output_root_var = tk.StringVar(value=str(self.materialized_root))
        self.status_var = tk.StringVar(value="请选择一份 LLM 规划结果。")

        self.plan_map: dict[str, Path] = {}
        self._build_ui()
        self.refresh_plan_list()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        plan_frame = ttk.LabelFrame(main, text="选择规划结果", padding=10)
        plan_frame.pack(fill=tk.X)

        ttk.Label(plan_frame, text="计划目录").grid(row=0, column=0, sticky="w")
        self.plan_combo = ttk.Combobox(plan_frame, textvariable=self.plan_var, state="readonly", width=58)
        self.plan_combo.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.plan_combo.bind("<<ComboboxSelected>>", self.on_plan_selected)
        ttk.Button(plan_frame, text="刷新", command=self.refresh_plan_list).grid(row=0, column=2, sticky="ew")

        ttk.Label(plan_frame, text="响应文件").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(plan_frame, textvariable=self.response_var).grid(
            row=1, column=1, sticky="ew", padx=(8, 8), pady=(10, 0)
        )
        ttk.Button(plan_frame, text="浏览", command=self.choose_response_file).grid(
            row=1, column=2, sticky="ew", pady=(10, 0)
        )

        ttk.Label(plan_frame, text="载荷文件").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(plan_frame, textvariable=self.payload_var).grid(
            row=2, column=1, sticky="ew", padx=(8, 8), pady=(10, 0)
        )
        ttk.Button(plan_frame, text="浏览", command=self.choose_payload_file).grid(
            row=2, column=2, sticky="ew", pady=(10, 0)
        )

        ttk.Label(plan_frame, text="输出目录").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(plan_frame, textvariable=self.output_root_var).grid(
            row=3, column=1, sticky="ew", padx=(8, 8), pady=(10, 0)
        )
        ttk.Button(plan_frame, text="浏览", command=self.choose_output_dir).grid(
            row=3, column=2, sticky="ew", pady=(10, 0)
        )

        for i, weight in enumerate((0, 1, 0)):
            plan_frame.columnconfigure(i, weight=weight)

        preview_frame = ttk.LabelFrame(main, text="建议预览", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.preview_text = tk.Text(preview_frame, wrap=tk.WORD, height=18)
        self.preview_text.pack(fill=tk.BOTH, expand=True)

        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(action_frame, text="转换为训练计划", command=self.run_materializer).pack(side=tk.LEFT)
        ttk.Button(action_frame, text="打开输出目录", command=self.open_output_dir).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(action_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))

        log_frame = ttk.LabelFrame(main, text="执行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def refresh_plan_list(self) -> None:
        plan_dirs = list_plan_dirs(self.planner_runs_root)
        self.plan_map = {path.name: path for path in plan_dirs}
        self.plan_combo["values"] = list(self.plan_map)
        if plan_dirs and not self.plan_var.get():
            self.plan_var.set(plan_dirs[0].name)
            self.on_plan_selected()
        self.status_var.set(f"已加载 {len(plan_dirs)} 个规划目录。")

    def on_plan_selected(self, _event: object | None = None) -> None:
        plan_dir = self.plan_map.get(self.plan_var.get())
        if not plan_dir:
            return
        response_path = plan_dir / "planner_response.json"
        payload_path = plan_dir / "planner_payload.json"
        self.response_var.set(str(response_path))
        self.payload_var.set(str(payload_path) if payload_path.exists() else "")
        self.load_preview(response_path)

    def load_preview(self, response_path: Path) -> None:
        self.preview_text.delete("1.0", tk.END)
        if not response_path.exists():
            self.preview_text.insert(tk.END, "未找到 planner_response.json")
            return
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.preview_text.insert(tk.END, f"读取失败：{exc}")
            return

        observations = payload.get("observations") or []
        runs = payload.get("recommended_runs") or []
        lines: list[str] = []
        lines.append("观察结论：")
        for item in observations:
            lines.append(f"- {item}")
        lines.append("")
        lines.append(f"建议实验数：{len(runs)}")
        for index, run in enumerate(runs, start=1):
            lines.append("")
            lines.append(f"{index}. {run.get('label', '')}")
            lines.append(f"原因：{run.get('reason', '')}")
            lines.append("参数：")
            overrides = run.get("overrides") or {}
            for key, value in overrides.items():
                lines.append(f"  {key}: {value}")
        self.preview_text.insert(tk.END, "\n".join(lines))

    def choose_response_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 planner_response.json",
            initialdir=str(self.planner_runs_root),
            filetypes=[("JSON 文件", "*.json")],
        )
        if not path:
            return
        self.response_var.set(path)
        response_path = Path(path)
        guessed_payload = response_path.with_name("planner_payload.json")
        if guessed_payload.exists():
            self.payload_var.set(str(guessed_payload))
        self.load_preview(response_path)

    def choose_payload_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 planner_payload.json",
            initialdir=str(self.planner_runs_root),
            filetypes=[("JSON 文件", "*.json")],
        )
        if path:
            self.payload_var.set(path)

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(
            title="选择输出目录", initialdir=self.output_root_var.get() or str(self.materialized_root)
        )
        if path:
            self.output_root_var.set(path)

    def open_output_dir(self) -> None:
        output_dir = Path(self.output_root_var.get().strip())
        if not output_dir.exists():
            messagebox.showerror("目录不存在", f"未找到目录：{output_dir}")
            return
        import os

        os.startfile(str(output_dir))

    def run_materializer(self) -> None:
        response_text = self.response_var.get().strip()
        if not response_text:
            messagebox.showerror("缺少响应文件", "请先选择 planner_response.json。")
            return

        response_path = Path(response_text)
        payload_path = Path(self.payload_var.get().strip()) if self.payload_var.get().strip() else None
        output_root = Path(self.output_root_var.get().strip()) if self.output_root_var.get().strip() else None

        self.status_var.set("正在转换，请稍候...")
        self.log(f"开始转换：{response_path}")

        def worker() -> None:
            try:
                result = materialize_from_paths(
                    planner_response_path=response_path,
                    payload_path=payload_path,
                    output_root=output_root,
                )
            except Exception:
                self.root.after(0, lambda: self.on_error(exc))
                return
            self.root.after(0, lambda: self.on_success(result))

        threading.Thread(target=worker, daemon=True).start()

    def on_success(self, result: dict[str, object]) -> None:
        self.status_var.set("转换完成。")
        self.log(f"输出目录：{result['plan_dir']}")
        self.log(f"Manifest：{result['manifest_path']}")
        self.log(f"实验数量：{result['experiments_count']}")
        messagebox.showinfo(
            "转换完成",
            f"输出目录：\n{result['plan_dir']}\n\n实验数量：{result['experiments_count']}",
        )

    def on_error(self, exc: Exception) -> None:
        self.status_var.set("转换失败。")
        self.log(f"转换失败：{exc}")
        messagebox.showerror("转换失败", str(exc))


def main() -> None:
    root = tk.Tk()
    MaterializerGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
