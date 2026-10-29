# agent/tools/datetime_tool.py —— 时间工具（Phase 2 示例，验证工具闭环）
#
# @tool 装饰器从函数签名 + docstring 生成 schema 和描述——描述就是给 LLM 的说明书。

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

_CST = ZoneInfo("Asia/Shanghai")


@tool
def get_current_time() -> str:
    """用户问"现在几点/当前时间"时调用。返回精确到秒的时刻（Asia/Shanghai 时区）；只问日期时用 get_current_date，系统提示词已含日期且用户不问具体时刻时无需调用。示例: 问"现在几点了"。"""
    return datetime.now(_CST).strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_current_date() -> str:
    """用户问"今天几号/日期"时调用。返回当前日期（Asia/Shanghai 时区，格式 YYYY-MM-DD）；需要时分秒时用 get_current_time。示例: 问"今天几号"。"""
    return datetime.now(_CST).strftime("%Y-%m-%d")
