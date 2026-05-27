from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open labelImg on a directory and jump to a target image.")
    parser.add_argument("--image-path", required=True, type=str)
    parser.add_argument("--class-file", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path).resolve()
    class_file = Path(args.class_file).resolve()
    save_dir = Path(args.save_dir).resolve()

    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    from labelImg.labelImg import MainWindow

    app = QApplication(sys.argv)
    image_dir = image_path.parent
    window = MainWindow(
        default_filename=str(image_dir),
        default_prefdef_class_file=str(class_file),
        default_save_dir=str(save_dir),
    )
    window.show()

    target_path = os.path.abspath(str(image_path))

    def select_target_image() -> None:
        normalized_list = [os.path.abspath(path) for path in window.m_img_list]
        if target_path in normalized_list:
            index = normalized_list.index(target_path)
            window.cur_img_idx = index
            window.load_file(window.m_img_list[index])
        else:
            QTimer.singleShot(150, select_target_image)

    QTimer.singleShot(250, select_target_image)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
