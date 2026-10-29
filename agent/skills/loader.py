# agent/skills/loader.py —— SKILL.md 加载与校验（Phase 8，E1）
#
# 递归发现 skills/ 目录下的 SKILL.md，解析 frontmatter（pyyaml），产出 Skill 数据类。
# 安全原则（书 4.8）：Skill 是把外部内容当指令的制度化形式——单文件失败只记诊断、
# 永不抛错（坏技能不能崩 agent，对齐 web 工具的结构化契约）。

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SKILL_FILENAME = "SKILL.md"
# 索引约束：description 超长会占满上下文，超出部分截断并记诊断（Hermes 同款检查）
MAX_DESCRIPTION_LEN = 60


@dataclass(frozen=True)
class Skill:
    """一份已加载的技能。

    model_invokable：frontmatter `disable-model-invocation: true` 时 False——
    该技能只允许 /skill 显式注入，不出现在模型可见的索引里（防模型乱触发）。
    """

    name: str
    description: str
    content: str
    location: str
    model_invokable: bool = True


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 `---` 包裹的 YAML frontmatter；无 frontmatter 返回 ({}, 全文)。"""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, text
    return meta, parts[2].strip()


def _load_one(path: Path) -> tuple[Skill | None, str | None]:
    """加载单个 SKILL.md；返回 (Skill, None) 或 (None, 诊断文本)。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return None, f"{path}: 读取失败（{e}）"
    meta, body = _parse_frontmatter(raw)

    name = meta.get("name") or path.parent.name
    description = meta.get("description")
    if not name:
        return None, f"{path}: 缺少 name（frontmatter 或目录名兜底为空）"
    if not description:
        return None, f"{path}: 缺少 description（索引路由条件，必填）"

    # 超出索引上限：截断用于展示，但仍加载（诊断提示作者修）
    diagnostic = None
    if len(description) > MAX_DESCRIPTION_LEN:
        description = description[:MAX_DESCRIPTION_LEN - 1] + "…"
        diagnostic = (
            f"{path}: description 超过 {MAX_DESCRIPTION_LEN} 字符，已截断用于索引"
        )
    if not body.strip():
        return None, f"{path}: 正文为空"

    return (
        Skill(
            name=name,
            description=description,
            content=body,
            location=str(path),
            model_invokable=not bool(meta.get("disable-model-invocation", False)),
        ),
        diagnostic,
    )


def load_skills(root: str | Path) -> tuple[list[Skill], list[str]]:
    """递归加载 root 下所有 SKILL.md；返回 (技能列表, 诊断列表)。

    诊断只警告不抛错：索引截断 / 文件损坏 / frontmatter 缺失都跳过继续。
    """
    root = Path(root)
    skills: list[Skill] = []
    diagnostics: list[str] = []
    if not root.is_dir():
        return skills, [f"{root}: 技能目录不存在"]
    for path in sorted(root.rglob(SKILL_FILENAME)):
        skill, diagnostic = _load_one(path)
        if skill is not None:
            skills.append(skill)
        if diagnostic:
            diagnostics.append(diagnostic)
    return skills, diagnostics
