# agent/tools/shell_tool.py —— 命令执行工具（对应 pi 的 bash 工具）
#
# cwd 限定 WORKSPACE_ROOT（无黑名单，安全交给宿主层钩子）；超时 + start_new_session
# + killpg（shell=True 产生子进程组，必须整组杀掉）；非零退出码不抛异常，
# exit code 非零时抛出异常，由 Tool Pipeline 投影为 error。

import os
import signal
import subprocess

from langchain_core.tools import tool

from agent.tools.file_tools import WORKSPACE_ROOT

DEFAULT_TIMEOUT = 30
MAX_OUTPUT_CHARS = 4000


def _truncate_tail(text: str) -> str:
    """超长输出截断：保留尾部（错误/关键信息常在结尾，对应 pi 保留尾部的策略）。"""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return f"...（已截断 {len(text) - MAX_OUTPUT_CHARS} 字符）\n{text[-MAX_OUTPUT_CHARS:]}"


@tool
def run_command(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """在 workspace/ 工作区内执行 shell 命令（默认超时 30 秒）。适合文件操作、环境查询等 shell 任务；非零退出码不报错，输出与退出码一并返回由你判断；耗时命令请调大 timeout，超时会整组杀掉进程；命令走 shell 执行，注意引号与转义。"""
    if timeout <= 0:
        raise ValueError(f"timeout 必须为正数: {timeout}")
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=WORKSPACE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # 独立进程组：超时时才能杀整组
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # 杀整组，防 shell 子进程残留
        proc.wait()
        raise TimeoutError(f"命令超时（>{timeout} 秒）: {command}")

    parts = []
    if stdout:
        parts.append(f"[stdout]\n{_truncate_tail(stdout.rstrip())}")
    if stderr:
        parts.append(f"[stderr]\n{_truncate_tail(stderr.rstrip())}")
    if not parts:
        parts.append("（无输出）")
    result = f"退出码 {proc.returncode}\n" + "\n".join(parts)
    if proc.returncode != 0:
        raise RuntimeError(result)
    return result
