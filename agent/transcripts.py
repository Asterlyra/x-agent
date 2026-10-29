"""当前对话与自动归档的轻量持久化。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


class TranscriptFileError(ValueError):
    """表示当前对话文件无法安全读取。"""


@dataclass(frozen=True)
class HistoryHit:
    """一条归档检索命中。"""

    source: str
    index: int
    content: str


@dataclass(frozen=True)
class SessionSummary:
    """浏览器会话列表使用的轻量摘要。"""

    id: str
    created_at: str
    updated_at: str
    message_count: int
    title: str
    active: bool = False


class TranscriptStore:
    """维护一个当前对话，并把旧对话自动归档。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.current_path = self.root / "current.jsonl"
        self.current_meta_path = self.root / "current.meta.json"
        self.archive_dir = self.root / "archive"

    @staticmethod
    def new_session_id() -> str:
        """生成可读且不会碰撞的会话 id。"""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return f"session-{stamp}-{uuid4().hex[:8]}"

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        """拒绝路径穿越和空会话 id。"""
        value = (session_id or "").strip()
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError("非法会话 id")
        return value

    def current_session_id(self) -> str:
        """读取当前会话 id；旧数据没有元文件时使用稳定兼容值。"""
        try:
            payload = json.loads(self.current_meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return "current"
        try:
            return self._validate_session_id(str(payload["session_id"]))
        except (KeyError, TypeError, ValueError):
            return "current"

    def _metadata(self) -> dict:
        """读取当前会话元数据；坏元数据按空对象处理。"""
        try:
            payload = json.loads(self.current_meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def set_current_session_id(self, session_id: str, created_at: str | None = None) -> None:
        """原子保存当前会话 id，不把元信息混入消息 JSONL。"""
        session_id = self._validate_session_id(session_id)
        previous = self._metadata()
        now = datetime.now().isoformat(timespec="seconds")
        previous_id = previous.get("session_id")
        payload = {
            "session_id": session_id,
            "created_at": created_at or (str(previous.get("created_at")) if previous_id == session_id else now),
            "updated_at": now,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.current_meta_path.with_name(f".{self.current_meta_path.name}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.current_meta_path)

    @staticmethod
    def _serialize(messages: list[BaseMessage]) -> str:
        """把 LangChain 消息编码为每行一条 JSON。"""
        lines = [json.dumps(message_to_dict(message), ensure_ascii=False) for message in messages]
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _read(path: Path) -> list[BaseMessage]:
        """读取并验证一个 transcript 文件。"""
        try:
            raw_lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise TranscriptFileError(f"无法读取 {path}: {exc}") from exc

        payloads = []
        for index, line in enumerate(raw_lines, 1):
            if not line.strip():
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TranscriptFileError(f"{path} 第 {index} 行不是合法 JSON") from exc
        try:
            return messages_from_dict(payloads)
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptFileError(f"{path} 包含无法识别的消息") from exc

    def load_current(self) -> list[BaseMessage]:
        """载入当前对话；文件不存在时返回空列表。"""
        return self._read(self.current_path)

    def save_current(self, messages: list[BaseMessage]) -> None:
        """原子保存完整当前对话，损坏文件不会被静默覆盖。"""
        if self.current_path.exists():
            self._read(self.current_path)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.current_path.with_name(f".{self.current_path.name}.{uuid4().hex}.tmp")
        temporary.write_text(self._serialize(messages), encoding="utf-8")
        os.replace(temporary, self.current_path)

    def load_session(self, session_id: str) -> list[BaseMessage]:
        """按会话 id 读取当前副本或归档副本。"""
        session_id = self._validate_session_id(session_id)
        if session_id == self.current_session_id():
            return self.load_current()
        path = self.archive_dir / f"{session_id}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"会话归档不存在: {session_id}")
        return self._read(path)

    def save_archive(self, session_id: str, messages: list[BaseMessage]) -> None:
        """原子写入指定会话归档，供切换和 fork 使用。"""
        session_id = self._validate_session_id(session_id)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        target = self.archive_dir / f"{session_id}.jsonl"
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        temporary.write_text(self._serialize(messages), encoding="utf-8")
        os.replace(temporary, target)

    @staticmethod
    def _title(messages: list[BaseMessage]) -> str:
        """从第一条用户消息生成列表标题。"""
        for message in messages:
            if getattr(message, "type", "") == "human":
                text = " ".join(str(message.content).split())
                if text:
                    return text[:40] + ("…" if len(text) > 40 else "")
        return "空会话"

    def _summary(self, session_id: str, messages: list[BaseMessage], path: Path, active: bool) -> SessionSummary:
        """从消息文件和文件时间生成会话摘要。"""
        stat = path.stat() if path.exists() else None
        timestamp = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds") if stat else datetime.now().isoformat(timespec="seconds")
        metadata = self._metadata() if active else {}
        created_at = str(metadata.get("created_at") or timestamp)
        return SessionSummary(
            id=session_id,
            created_at=created_at,
            updated_at=timestamp,
            message_count=len(messages),
            title=self._title(messages),
            active=active,
        )

    def list_sessions(self) -> list[SessionSummary]:
        """列出当前会话和有效归档，坏归档跳过。"""
        current_id = self.current_session_id()
        current_messages = self.load_current()
        summaries = [self._summary(current_id, current_messages, self.current_path, True)]
        if self.archive_dir.is_dir():
            for path in sorted(self.archive_dir.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True):
                if path.stem == current_id:
                    continue
                try:
                    messages = self._read(path)
                except TranscriptFileError:
                    continue
                summaries.append(self._summary(path.stem, messages, path, False))
        return sorted(summaries, key=lambda item: (item.active, item.updated_at), reverse=True)

    def archive_current(self, source: str | None = None) -> str | None:
        """把非空当前对话移入归档，并返回归档来源名。"""
        messages = self.load_current()
        if not messages:
            self.clear_current()
            return None
        source = self._validate_session_id(source or self.new_session_id())
        self.save_archive(source, messages)
        self.save_current([])
        return source

    def delete_archive(self, session_id: str) -> None:
        """删除一个归档会话；不存在时保持幂等。"""
        session_id = self._validate_session_id(session_id)
        try:
            (self.archive_dir / f"{session_id}.jsonl").unlink()
        except FileNotFoundError:
            pass

    def clear_current(self) -> None:
        """清空当前对话，不创建归档。"""
        self.save_current([])

    def search_archives(self, query: str) -> list[HistoryHit]:
        """按大小写不敏感子串顺序扫描有效归档。"""
        needle = (query or "").strip().casefold()
        if not needle or not self.archive_dir.is_dir():
            return []
        hits: list[HistoryHit] = []
        for path in sorted(self.archive_dir.glob("*.jsonl"), reverse=True):
            try:
                messages = self._read(path)
            except TranscriptFileError:
                continue
            for index, message in enumerate(messages):
                content = str(message.content)
                if needle in content.casefold():
                    hits.append(HistoryHit(path.stem, index, content))
        return hits
