"""
量化交易系统

主程序入口 - 直接启动UI
"""

import sys


def main():
    """主函数 - 直接启动UI"""
    from UI.main_ui import main as ui_main
    ui_main()


if __name__ == "__main__":
    main()
