# agent/skills/format.py —— 技能渲染（Phase 8，E2）
#
# 系统提示词只列索引（name + description，模型据此路由）；全文由 skill_view 工具
# 或 /skill 命令按需加载。对齐 Hermes prompt_builder.py 的 available_skills 索引段。

from xml.sax.saxutils import escape

from agent.skills.loader import Skill


def format_skills_index(skills: list[Skill]) -> str:
    """`<available_skills>` 索引段：只列模型可主动调用的技能，XML 转义防注入。

    返回空串表示无可用技能（不污染系统提示词）；disable-model-invocation
    的技能不出现在索引里（只走 /skill 显式注入）。
    """
    visible = [s for s in skills if s.model_invokable]
    if not visible:
        return ""
    lines = ["<available_skills>"]
    for s in sorted(visible, key=lambda s: s.name):
        lines.append(f"- {escape(s.name)}: {escape(s.description)}")
    lines.append("</available_skills>")
    return "\n".join(lines)


def format_skill_content(skill: Skill) -> str:
    """把整篇技能封装为 `<skill name location>` 块（注入上下文用）。

    location 供模型溯源；正文按原样嵌入（技能作者已按四部分规范书写）。
    """
    return (
        f'<skill name="{escape(skill.name)}" location="{escape(skill.location)}">\n'
        f"{skill.content}\n"
        f"</skill>"
    )
