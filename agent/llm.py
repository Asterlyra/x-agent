# agent/llm.py —— 模型工厂 + 可观测性开关（对应 pi 的 pi-ai 模型层）
#
# 全项目只有这里知道"模型怎么构造"：切 provider 只改这一个文件；
# api_key 优先读取进程环境变量，再退项目根 .env，绝不硬编码。
# Phase 12：只保留 DeepSeek provider；外部 tracing 已移除。

import os
from pathlib import Path

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEMPERATURE = 0.7
# 工具图暂时只支持显式 non-thinking；恢复其他档位前必须保留 reasoning_content。
THINKING_LEVELS = ("off",)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _read_key_from_env_file(key: str) -> str | None:
    """从项目根 .env 读取指定键（标准库手写解析，不引 python-dotenv）。

    规则：只认 KEY=VALUE 行，跳过空行和 # 注释；文件不存在/没找到 → None。
    """
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, value = line.partition("=")
        if k.strip() == key:
            return value.strip()
    return None


def _read_env_or_fallback(key: str) -> str | None:
    """环境变量优先，找不到时退回项目根 `.env`。"""
    return os.environ.get(key) or _read_key_from_env_file(key)


def create_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    reasoning_effort: str | None = None,
    **kwargs,
) -> ChatOpenAI:
    """创建 DeepSeek 模型对象。

    api_key 查找顺序：显式参数 → 环境变量 → `.env`；
    当前工具图只允许 off，并显式发送 DeepSeek 的 thinking disabled；
    base_url 仅用于兼容 DeepSeek 的测试代理，不承诺任意 OpenAI 兼容服务。
    """
    if api_key is None:
        api_key = _read_env_or_fallback("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY：请在项目根目录 .env 里写 `DEEPSEEK_API_KEY=...`，"
            "或先 `export DEEPSEEK_API_KEY=...`"
        )
    if reasoning_effort not in {None, "off"}:
        raise ValueError(
            f"当前工具图只支持关闭思考模式，收到: {reasoning_effort!r}"
        )
    extra_body = dict(kwargs.pop("extra_body", {}) or {})
    configured_thinking = extra_body.get("thinking")
    if configured_thinking not in (None, "disabled", {"type": "disabled"}):
        raise ValueError("当前工具图不允许通过 extra_body 启用思考模式")
    extra_body["thinking"] = {"type": "disabled"}

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body,
        **kwargs,
    )
