"""只读访问当前项目源码的协作角色工具。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MAX_FILES = 300
DEFAULT_MAX_LINES = 240
DEFAULT_MAX_CHARS = 12_000

_DENIED_DIRECTORY_NAMES = frozenset({".git", ".venv", "node_modules"})


def _relative_parts(path: Path, root: Path) -> tuple[str, ...]:
    """返回相对路径组成，项目根本身返回空元组。"""
    if path == root:
        return ()
    return path.relative_to(root).parts


def _is_denied(relative: Path) -> bool:
    """判断项目内相对路径是否属于敏感或运行时目录。"""
    parts = relative.parts
    if any(part in _DENIED_DIRECTORY_NAMES for part in parts):
        return True
    if any(part == ".env" or part.startswith(".env.") for part in parts):
        return True
    return len(parts) >= 2 and parts[0] == "data" and parts[1] == "history"


def _resolve_repo_path(path: str) -> Path:
    """解析项目根内路径，并拒绝敏感目录和符号链接。"""
    root = PROJECT_ROOT.resolve()
    raw = Path(path).expanduser()
    candidate = raw if raw.is_absolute() else PROJECT_ROOT / raw
    unresolved = candidate.absolute()
    try:
        relative_unresolved = unresolved.relative_to(PROJECT_ROOT.absolute())
    except ValueError as exc:
        raise ValueError(f"路径超出项目根（{PROJECT_ROOT}）: {unresolved}") from exc
    if _is_denied(relative_unresolved):
        raise ValueError(f"路径属于禁止读取范围: {relative_unresolved}")

    cursor = PROJECT_ROOT.absolute()
    for part in relative_unresolved.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"项目源码工具拒绝符号链接路径: {relative_unresolved}")

    resolved = candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(f"路径超出项目根（{PROJECT_ROOT}）: {resolved}")
    relative = Path(*_relative_parts(resolved, root))
    if _is_denied(relative):
        raise ValueError(f"路径属于禁止读取范围: {relative}")
    return resolved


def _safe_relative_file(candidate: Path, root: Path) -> Path | None:
    """把可读取的普通文件转成项目相对路径。"""
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    if _is_denied(relative) or candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_file() or not resolved.is_relative_to(root):
        return None
    return relative


def _iter_safe_files(target: Path, root: Path, *, recursive: bool) -> Iterator[Path]:
    """按稳定顺序遍历文件，并在进入敏感目录前完成剪枝。"""
    try:
        entries = sorted(target.iterdir(), key=lambda item: item.name)
    except OSError:
        return
    for candidate in entries:
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_symlink() or _is_denied(relative):
            continue
        try:
            if candidate.is_file():
                yield candidate
            elif recursive and candidate.is_dir():
                yield from _iter_safe_files(candidate, root, recursive=True)
        except OSError:
            continue


@tool
def list_repo_files(
    directory: str = ".",
    recursive: bool = True,
    limit: int = DEFAULT_MAX_FILES,
) -> str:
    """研究当前 x-agent 项目源码、测试或文档时调用（只读）。只列项目根内文件；自动排除密钥、Git、虚拟环境、依赖和会话历史；最多返回 limit 条。"""
    target = _resolve_repo_path(directory)
    if not target.exists():
        raise FileNotFoundError(f"目录不存在: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"不是目录: {target}")
    safe_limit = max(1, min(int(limit), DEFAULT_MAX_FILES))
    root = PROJECT_ROOT.resolve()
    files: list[str] = []
    truncated = False
    for candidate in _iter_safe_files(target, root, recursive=recursive):
        relative = _safe_relative_file(candidate, root)
        if relative is None:
            continue
        if len(files) >= safe_limit:
            truncated = True
            break
        files.append(str(relative))
    if not files:
        return "（目录中没有可读的项目文件）"
    suffix = f"\n[最多显示 {safe_limit} 条]" if truncated else ""
    return "\n".join(files) + suffix


@tool
def read_repo_file(path: str, offset: int = 1, limit: int = DEFAULT_MAX_LINES) -> str:
    """读取当前 x-agent 项目中的 UTF-8 源码、测试或文档（只读）。输出带行号并支持 offset/limit 分页；密钥、Git、虚拟环境、依赖、会话历史、符号链接和项目外路径会被拒绝。"""
    target = _resolve_repo_path(path)
    if not target.exists():
        raise FileNotFoundError(f"文件不存在: {target}")
    if not target.is_file():
        raise IsADirectoryError(f"不是文件: {target}")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("项目源码工具只读取 UTF-8 文本文件") from exc
    if "\x00" in content:
        raise ValueError("项目源码工具拒绝二进制文件")

    lines = content.splitlines()
    start = max(0, int(offset) - 1)
    safe_limit = max(1, min(int(limit), DEFAULT_MAX_LINES))
    selected = lines[start : start + safe_limit]
    rendered: list[str] = []
    used = 0
    for index, line in enumerate(selected, start=start + 1):
        item = f"{index}: {line}"
        if rendered and used + len(item) + 1 > DEFAULT_MAX_CHARS:
            break
        rendered.append(item)
        used += len(item) + 1
    text = "\n".join(rendered)
    consumed = len(rendered)
    if start + consumed < len(lines):
        text += (
            f"\n[已截断，共 {len(lines)} 行，继续读取请用 "
            f"offset={start + consumed + 1}]"
        )
    return text


__all__ = ["list_repo_files", "read_repo_file"]
