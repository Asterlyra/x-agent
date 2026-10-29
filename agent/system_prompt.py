# agent/system_prompt.py —— 系统提示词体系（Phase 4，D3）
#
# 静态 PERSONA + 每轮动态构建：call_model 每轮重建 SystemMessage 注入 LLM 调用，
# 不进会话历史（动态段随工具变化，落盘无意义）。

# 静态人设：身份 / 能力边界 / 行为约定——模型据此定位自己
PERSONA = """你是 x-agent，一个个人助理。目标是高效、准确地帮助用户。

行为约定：
- 默认用中文交流，回答简洁直接。
- 不确定的信息明确说明，不编造事实。
- 需要实时信息（时间、天气、网页内容）时，先调用工具获取结果，再基于结果回答。
- 涉及文件读写时只操作工作区（workspace/）内的文件。"""

def format_tool_list(tools) -> str:
    """把工具列表格式化为"名称: 描述首行"清单——模型由此知道可用工具。

    无工具时返回空串（不污染提示词）。描述只取首行，避免长描述占满上下文。
    """
    if not tools:
        return ""
    lines = ["可用工具:"]
    for t in tools:
        desc = (t.description or "").splitlines()[0] if t.description else "无描述"
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


def build_system_prompt(tools, persona: str | None = None, skills=None,
                        memory: str | None = None) -> str:
    """每轮动态构建完整系统提示词：人设 + 技能索引 + 记忆快照 + 工具清单。

    tools：本轮绑定的工具列表（来自图注入）。当前时间已迁至 Agent 状态栏
    （Phase 7.5 修正：动态段在开头会破坏前缀缓存，状态栏在上下文末尾代价更低）。
    persona：可插拔人设（Phase 7.8，垂类预留接口①）；None = 默认 PERSONA。
    skills：Phase 8 技能索引——只列 name+description（渐进式披露，全文经
    skill_view / /skill 按需加载）；None/空 = 不注入索引段。
    memory：Phase 10 长期记忆**冻结快照**（MEMORY.md 内容，会话内恒定——写入
    不改变它，保前缀缓存）；None/空 = 不注入记忆段。
    """
    sections = [persona if persona is not None else PERSONA]
    # 技能索引放动态段最前（会话内稳定，不破坏前缀缓存；全文不进系统提示词）
    if skills:
        from agent.skills.format import format_skills_index

        skills_section = format_skills_index(skills)
        if skills_section:
            sections.append("\n" + skills_section)
    # 记忆快照在技能之后、工具清单之前（DESIGN.md §2.5 volatile 顺序：
    # skills 索引 → 记忆快照 → 用户画像 → 时间戳；快照本身会话内稳定不破缓存）
    if memory and memory.strip():
        sections.append(
            f"\n<long_term_memory>\n（会话启动时的冻结快照；会话内新写入经 "
            f"remember 后下一会话生效）\n{memory.strip()}\n</long_term_memory>"
        )
    tool_list = format_tool_list(tools)
    if tool_list:
        sections.append("\n" + tool_list)
    return "\n\n".join(sections)
