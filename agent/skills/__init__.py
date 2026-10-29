# agent/skills/__init__.py —— 技能系统（Phase 8：SKILL.md 加载 + 渐进式披露）
#
# loader：递归加载 + frontmatter 校验（失败只诊断不抛错）；
# format：<available_skills> 索引段 / <skill> 内容封装；
# skill_tools：skills_list / skill_view 渐进式披露工具（闭包注入技能列表）。
