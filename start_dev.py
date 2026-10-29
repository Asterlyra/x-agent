"""本地开发启动器：同时启动 FastAPI 后端和 Vite 前端。"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"


def _start_process(command: list[str], cwd: Path, label: str) -> subprocess.Popen:
    """启动一个开发进程，并把输出直接显示在当前终端。"""
    print(f"[{label}] {' '.join(command)}", flush=True)
    return subprocess.Popen(command, cwd=cwd, start_new_session=True)


def main() -> int:
    """启动前后端并在退出时清理子进程。"""
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        print("未找到 pnpm，请先安装 pnpm。", file=sys.stderr)
        return 1
    if not FRONTEND_ROOT.is_dir():
        print(f"前端目录不存在: {FRONTEND_ROOT}", file=sys.stderr)
        return 1
    if not (FRONTEND_ROOT / "node_modules").is_dir():
        print("未发现前端依赖，先执行 pnpm install…", flush=True)
        result = subprocess.run([pnpm, "install"], cwd=FRONTEND_ROOT)
        if result.returncode != 0:
            return result.returncode

    processes = [
        _start_process(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.app:app",
                "--reload",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            PROJECT_ROOT,
            "backend",
        ),
        _start_process(
            [pnpm, "exec", "vite", "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
            FRONTEND_ROOT,
            "frontend",
        ),
    ]
    print("网页地址: http://127.0.0.1:5173", flush=True)
    print("按 Ctrl+C 同时停止前后端。", flush=True)

    try:
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    return return_code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在停止开发服务…", flush=True)
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
