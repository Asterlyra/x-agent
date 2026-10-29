# agent/tools/todo_tool.py —— todo 管理器（状态化工具示例）
#
# 教学点：工具持有状态（模块级内存）但重启即丢——工具状态 vs 会话状态是两回事。

from langchain_core.tools import tool

# 内存存储：list[dict]，每条 {"task": str, "done": bool}
_todos: list[dict] = []


@tool
def todo_add(task: str) -> str:
    """用户要"记下来/添加待办事项"时调用。是内存存储，进程重启后清空。示例: 用户说"帮我记一下买牛奶" → todo_add("买牛奶")。"""
    _todos.append({"task": task, "done": False})
    return f"已添加待办 #{len(_todos)}: {task}"


@tool
def todo_list() -> str:
    """用户问"我记了什么/有哪些待办"时调用；或在标记完成前先查编号（只读，不修改数据）。编号从 1 开始。"""
    if not _todos:
        return "待办列表为空"
    return "\n".join(
        f"{i}. [{'x' if t['done'] else ' '}] {t['task']}" for i, t in enumerate(_todos, 1)
    )


@tool
def todo_done(index: int) -> str:
    """把第 index 条待办标记为完成。index 为 todo_list 中的编号（从 1 开始），先查列表再调用；编号越界会报错。"""
    if not 1 <= index <= len(_todos):
        raise ValueError(f"待办编号越界: {index}（共 {len(_todos)} 条）")
    _todos[index - 1]["done"] = True
    return f"已标记完成 #{index}: {_todos[index - 1]['task']}"


def todo_snapshot() -> list[dict]:
    """返回当前待办列表副本（只读，供状态栏等代码读取——不暴露内部可变引用）。"""
    return [dict(item) for item in _todos]


def reset_todos() -> None:
    """清空进程内待办，供隔离评测和测试建立干净环境。"""
    _todos.clear()
