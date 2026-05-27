from __future__ import annotations

import argparse
import html
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local HTTP launcher for labelImg.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def html_page(title: str, body: str) -> bytes:
    return (
        "\n".join(
            [
                "<!DOCTYPE html>",
                '<html lang="zh-CN">',
                "<head>",
                '  <meta charset="utf-8">',
                f"  <title>{html.escape(title)}</title>",
                "  <style>",
                "    body { font-family: 'Segoe UI', sans-serif; padding: 24px; background: #f8fafc; color: #0f172a; }",
                "    .card { max-width: 860px; background: #fff; border: 1px solid #dbe3ec; border-radius: 16px; padding: 20px; }",
                "    h1 { margin-top: 0; font-size: 24px; }",
                "    p, li { font-size: 14px; color: #334155; }",
                "    code { background: #eff6ff; padding: 2px 6px; border-radius: 6px; }",
                "  </style>",
                "</head>",
                "<body>",
                f'  <div class="card">{body}</div>',
                "</body>",
                "</html>",
            ]
        )
    ).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = "\n".join(
                [
                    "<h1>LabelImg Launcher</h1>",
                    "<p>服务已启动。报告里的 <code>打开 LabelImg</code> 和 <code>打开参数 GUI</code> 按钮会调用这个本地服务。</p>",
                    "<p>直接访问方式：</p>",
                    "<ul>",
                    "<li><code>/open?labelimg_exe=...&image_dir=...&class_file=...&save_dir=...</code></li>",
                    "<li><code>/open_inspector?predict_dir=...&image_name=...</code></li>",
                    "</ul>",
                ]
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_page("LabelImg Launcher", body))
            return

        if parsed.path == "/open_inspector":
            query = parse_qs(parsed.query)
            predict_dir = Path(query.get("predict_dir", [""])[0])
            image_name = query.get("image_name", [""])[0]
            if not str(predict_dir):
                self.send_error(400, "Missing query param: predict_dir")
                return
            if not image_name:
                self.send_error(400, "Missing query param: image_name")
                return
            if not predict_dir.exists():
                self.send_error(400, f"Path not found: predict_dir={predict_dir}")
                return

            helper_python = Path(sys.executable)
            helper_script = Path(__file__).resolve().parent.parent / "predict" / "youge" / "inspect_postprocess_gui.py"
            if not helper_script.exists():
                self.send_error(500, f"Inspector script not found: {helper_script}")
                return

            subprocess.Popen(
                [
                    str(helper_python),
                    str(helper_script),
                    "--predict-dir",
                    str(predict_dir),
                    "--image-name",
                    image_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                cwd=str(predict_dir),
            )

            body = "\n".join(
                [
                    "<h1>已启动参数 GUI</h1>",
                    f"<p><strong>predict_dir</strong>: {html.escape(str(predict_dir))}</p>",
                    f"<p><strong>image_name</strong>: {html.escape(image_name)}</p>",
                    "<p>可以关闭这个页面，回到报告继续点别的样本。</p>",
                ]
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_page("Inspector Launched", body))
            return

        if parsed.path != "/open":
            self.send_error(404, "Not Found")
            return

        query = parse_qs(parsed.query)
        labelimg_exe = Path(query.get("labelimg_exe", [""])[0])
        image_target = Path(query.get("image_dir", [""])[0])
        class_file = Path(query.get("class_file", [""])[0])
        save_dir = Path(query.get("save_dir", [""])[0])

        missing = [
            name
            for name, value in {
                "labelimg_exe": labelimg_exe,
                "image_dir": image_target,
                "class_file": class_file,
                "save_dir": save_dir,
            }.items()
            if not str(value)
        ]
        if missing:
            self.send_error(400, f"Missing query params: {', '.join(missing)}")
            return

        for name, path in {
            "labelimg_exe": labelimg_exe,
            "image_dir": image_target,
            "class_file": class_file,
            "save_dir": save_dir,
        }.items():
            if not path.exists():
                self.send_error(400, f"Path not found: {name}={path}")
                return

        labelimg_python = labelimg_exe.parent.parent / "python.exe"
        helper_script = Path(__file__).with_name("labelimg_open_target.py")
        if not labelimg_python.exists():
            self.send_error(400, f"Python not found for labelImg env: {labelimg_python}")
            return
        if not helper_script.exists():
            self.send_error(500, f"Helper script not found: {helper_script}")
            return

        subprocess.Popen(
            [
                str(labelimg_python),
                str(helper_script),
                "--image-path",
                str(image_target),
                "--class-file",
                str(class_file),
                "--save-dir",
                str(save_dir),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=str(image_target.parent if image_target.is_file() else image_target),
        )

        body = "\n".join(
            [
                "<h1>已启动 LabelImg</h1>",
                f"<p><strong>image_target</strong>: {html.escape(str(image_target))}</p>",
                f"<p><strong>class_file</strong>: {html.escape(str(class_file))}</p>",
                f"<p><strong>save_dir</strong>: {html.escape(str(save_dir))}</p>",
                "<p>可以关闭这个页面，回到报告继续点别的样本。</p>",
            ]
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html_page("LabelImg Launched", body))

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LabelImg launcher listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
