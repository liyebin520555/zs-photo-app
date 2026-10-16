# -*- coding: utf-8 -*-
"""证件照工具 - 主程序入口"""
import sys
import os

# 将脚本所在目录加入 sys.path，确保导入 ui 模块时能找到
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from PySide6.QtWidgets import QApplication, QMainWindow
from ui.photo_tab import PhotoTab


class MainWindow(QMainWindow):
    """主窗口类：证件照工具"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("证件照工具")
        self.resize(1280, 820)

        # 计算屏幕可用区域并居中显示窗口
        screen = QApplication.primaryScreen().availableGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

        self.setCentralWidget(PhotoTab())


def main() -> None:
    """程序入口函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("证件照工具")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
