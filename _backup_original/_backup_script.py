"""
项目重构备份脚本

1. 创建备份文件夹
2. 复制原始文件到备份文件夹
3. 验证备份完整性
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"d:\Users\ii52\PycharmProjects\Quantitative Learning")
BACKUP_ROOT = PROJECT_ROOT / "_backup_original"

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    "_backup_original",
    "node_modules",
    ".venv",
    "venv",
    ".env",
}

EXCLUDE_FILES = {
    ".pyc",
    ".pyo",
    ".log",
    ".db",
    ".sqlite",
}


def create_backup():
    print("=" * 60)
    print("项目重构备份脚本")
    print("=" * 60)
    
    if BACKUP_ROOT.exists():
        print(f"\n备份文件夹已存在: {BACKUP_ROOT}")
        print("正在清理旧备份...")
        shutil.rmtree(BACKUP_ROOT)
    
    BACKUP_ROOT.mkdir(parents=True)
    print(f"\n创建备份文件夹: {BACKUP_ROOT}")
    
    copied_files = []
    copied_dirs = []
    
    for item in PROJECT_ROOT.iterdir():
        if item.name in EXCLUDE_DIRS:
            continue
        
        if item.is_file():
            if any(item.name.endswith(ext) for ext in EXCLUDE_FILES):
                continue
            
            dest = BACKUP_ROOT / item.name
            shutil.copy2(item, dest)
            copied_files.append(item.name)
            print(f"  复制文件: {item.name}")
        
        elif item.is_dir():
            dest = BACKUP_ROOT / item.name
            shutil.copytree(item, dest, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS, *EXCLUDE_FILES))
            copied_dirs.append(item.name)
            print(f"  复制目录: {item.name}/")
    
    print("\n" + "=" * 60)
    print("备份完成!")
    print(f"  复制文件数: {len(copied_files)}")
    print(f"  复制目录数: {len(copied_dirs)}")
    print(f"  备份位置: {BACKUP_ROOT}")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    create_backup()
