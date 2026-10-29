# agent/skills/skill_tools.py —— 技能渐进式披露工具（Phase 8，E2）
#
# skills_list（索引）/ skill_view（按名加载全文）——模型按 description 路由，
# 需要时主动调 skill_view 读全文，替代"全文注入"（token 可控且模型可发现）。
# 结构化契约：返回 {"success", "data"/"error"} 不抛异常（对齐 web_tools）。

import json

from langchain_core.tools import tool

from agent.skills.format import format_skill_content, format_skills_index


def make_skill_tools(skills: list) -> list:
    """按给定技能列表构造 skill_list / skill_view 两个工具（闭包注入技能）。

    graph.py 在配置了 skills 时条件注册——默认注册表不含技能工具，
    模型 schema 只暴露 name 参数（skills 经闭包捕获，不进 schema）。
    """

    @tool
    def skills_list() -> str:
        """列出当前可用的技能（名称 + 用途描述），供决定是否加载全文。

        当任务可能匹配某个技能描述时，先调用本工具查看有哪些技能可用，
        再用 skill_view 加载匹配技能的完整内容。技能包含专业流程与约定，
        比泛化做法更可靠。"""
        index = format_skills_index(skills)
        return json.dumps(
            {"success": True, "data": {"skills": index}},
            ensure_ascii=False,
        )

    @tool
    def skill_view(name: str) -> str:
        """按名称加载一篇技能的完整内容（含核心原则 / 禁止清单 / 参考资料）。

        技能名取自 skills_list 的索引。仅在确实要执行该技能覆盖的任务时调用，
        把全文内容用作执行规范；不匹配时不要加载（避免上下文浪费）。"""
        hit = next((s for s in skills if s.name == name), None)
        if hit is None:
            available = "、".join(s.name for s in skills) or "（无）"
            return json.dumps(
                {"success": False, "error": f"未找到技能 {name}，可用: {available}"},
                ensure_ascii=False,
            )
        return json.dumps(
            {"success": True, "data": {"content": format_skill_content(hit)}},
            ensure_ascii=False,
        )

    return [skills_list, skill_view]
