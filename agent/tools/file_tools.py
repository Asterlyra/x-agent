# agent/tools/file_tools.py —— 本地文件工具集（read/write/edit）
#
# 安全边界：路径必须落在 WORKSPACE_ROOT（项目根/workspace）内，防目录穿越；
# 测试通过 monkeypatch WORKSPACE_ROOT 隔离。

from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

DEFAULT_MAX_LINES = 200
DEFAULT_MAX_CHARS = 4000
DEFAULT_MAX_FILES = 200


def _resolve_path(path: str) -> Path:
    """把用户给的路径解析成工作区内的绝对路径，越界直接报错。"""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p
    p = p.resolve()
    root = WORKSPACE_ROOT.resolve()
    if p != root and not p.is_relative_to(root):
        raise ValueError(f"路径超出工作区（{WORKSPACE_ROOT}）: {p}")
    return p


@tool
def list_files(directory: str = ".", recursive: bool = True, limit: int = DEFAULT_MAX_FILES) -> str:
    """用户要研究工作区目录结构或寻找文件时调用（只读）。只列出 workspace 内的相对文件路径；recursive=false 只看当前层；最多返回 limit 条，目录越界或不是目录会报错。"""
    root = _resolve_path(directory)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {root}")
    safe_limit = max(1, min(int(limit), DEFAULT_MAX_FILES))
    candidates = root.rglob("*") if recursive else root.glob("*")
    files: list[str] = []
    workspace = WORKSPACE_ROOT.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file() or not resolved.is_relative_to(workspace):
            continue
        files.append(str(resolved.relative_to(workspace)))
        if len(files) >= safe_limit:
            break
    files.sort()
    if not files:
        return "（目录中没有可读文件）"
    suffix = f"\n[最多显示 {safe_limit} 条]" if len(files) >= safe_limit else ""
    return "\n".join(files) + suffix


@tool
def read_file(path: str, offset: int = 1, limit: int = DEFAULT_MAX_LINES) -> str:
    """用户要查看工作区文件内容时调用（只读，不会修改文件）。大文件按 offset/limit 分页读取，输出带截断提示时按提示的 offset 续读；文件不存在会报错，工作区外的路径会被拒绝。"""
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    start = max(0, offset - 1)
    selected = lines[start : start + limit]
    text = "\n".join(selected)
    if start + len(selected) < len(lines):
        text += f"\n[已截断，共 {len(lines)} 行，继续读取请用 offset={start + len(selected) + 1}]"
    return text


@tool
def write_file(path: str, content: str) -> str:
    """创建新文件或整体覆盖写文件（仅限工作区内）。content 为完整新内容，原内容会被整体替换——局部修改请用 edit_file；父目录不存在时自动创建。"""
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {p.relative_to(WORKSPACE_ROOT)}（{len(content)} 字节）"


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """局部修改文件：把第一处 old_string 精确替换为 new_string（原样匹配，含空白）。替换前先 read_file 确认目标文本存在；old_string 未找到会报错，此时缩小匹配片段或换内容重试，不要重复相同调用。"""
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    text = p.read_text(encoding="utf-8")
    if old_string not in text:
        raise ValueError(f"未在文件中找到目标文本: {old_string!r}")
    p.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"已替换第一处匹配（{p.relative_to(WORKSPACE_ROOT)}）"
