# agent/tools/__init__.py —— 工具注册表（对应 pi 的 tools Map）
#
# 显式列表而非装饰器注册：加工具只改这里、图不动；顺序即绑定顺序。

from agent.tools.calculate_tool import calculate
from agent.tools.datetime_tool import get_current_date, get_current_time
from agent.tools.file_tools import edit_file, list_files, read_file, write_file
from agent.tools.todo_tool import todo_add, todo_done, todo_list
from agent.tools.web_tools import fetch_url, web_search

TOOL_REGISTRY: list = [
    # 时间
    get_current_time,
    get_current_date,
    # 计算
    calculate,
    # 待办（内存状态，重启即丢）
    todo_add,
    todo_list,
    todo_done,
    # 本地文件（限定 workspace/ 内）
    list_files,
    read_file,
    write_file,
    edit_file,
    # 网页（网络工具，依赖网络环境）
    fetch_url,
    web_search,
]
