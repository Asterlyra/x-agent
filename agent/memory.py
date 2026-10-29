"""跨会话长期记忆：冻结快照 + 可控写入。"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


_ENTRY_RE = re.compile(r"^- \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] .+")
_DANGEROUS = ("ignore previous instructions", "忽略之前的指令", "system prompt", "系统提示词")


class Memory:
    """维护 `MEMORY.md`，会话中使用启动时冻结的快照。"""

    def __init__(self, path: str | Path, char_limit: int = 2200):
        self.path = Path(path)
        self.char_limit = char_limit
        self._current = self._load()
        self._frozen = self._current

    def _load(self) -> str:
        """读取记忆文件。"""
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def frozen_snapshot(self) -> str:
        """返回本轮固定的记忆快照。"""
        return self._frozen

    def refreeze(self) -> None:
        """重新读取磁盘并开始新的快照。"""
        self._current = self._load()
        self._frozen = self._current

    def remember(self, content: str, old_text: str | None = None) -> dict:
        """追加或唯一替换一条记忆，并返回结构化结果。"""
        content = " ".join((content or "").split())
        if not content:
            return {"success": False, "error": "内容为空"}
        lowered = content.casefold()
        if any(pattern in lowered for pattern in _DANGEROUS):
            return {"success": False, "error": "写入被内容审查拦截"}
        current = self._load()
        lines = [line for line in current.splitlines() if line.strip()]
        if any(not _ENTRY_RE.match(line) for line in lines):
            return {"success": False, "error": "记忆文件包含无法识别的行，请先整理为条目格式"}
        entry = f"- [{datetime.now():%Y-%m-%d %H:%M}] {content}"
        if old_text:
            matches = [index for index, line in enumerate(lines) if old_text in line]
            if len(matches) > 1:
                return {"success": False, "error": "old_text 匹配多条记忆，请提供更精确文本"}
            if matches:
                lines[matches[0]] = entry
            else:
                lines.append(entry)
        else:
            lines.append(entry)
        updated = "\n".join(lines) + ("\n" if lines else "")
        if len(updated) > self.char_limit:
            return {"success": False, "error": f"超出记忆字符上限 {self.char_limit}"}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(updated, encoding="utf-8")
        os.replace(temporary, self.path)
        self._current = updated
        return {"success": True, "data": {"entry": entry, "entry_count": len(lines)}}
