import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.transcripts import TranscriptFileError, TranscriptStore


def test_transcript_round_trip_keeps_langchain_message_shape(tmp_path):
    store = TranscriptStore(tmp_path / "history")
    messages = [HumanMessage(content="你好", id="u1"), AIMessage(content="你好！", id="a1")]
    store.save_current(messages)
    loaded = store.load_current()
    assert [message.type for message in loaded] == ["human", "ai"]
    assert [message.id for message in loaded] == ["u1", "a1"]


def test_transcript_rejects_invalid_json(tmp_path):
    path = tmp_path / "history" / "current.jsonl"
    path.parent.mkdir()
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(TranscriptFileError, match="不是合法 JSON"):
        TranscriptStore(path.parent).load_current()


def test_transcript_reset_truncates_current_file(tmp_path):
    store = TranscriptStore(tmp_path / "history")
    store.save_current([HumanMessage(content="暂存", id="u1")])
    store.clear_current()
    assert store.load_current() == []


def test_archive_search_skips_corrupt_archive(tmp_path):
    store = TranscriptStore(tmp_path / "history")
    store.save_current([HumanMessage(content="LangGraph", id="u1")])
    source = store.archive_current()
    assert source
    assert store.search_archives("langgraph")[0].source == source
    (tmp_path / "history" / "archive" / "broken.jsonl").write_text("{}\n", encoding="utf-8")
    assert len(store.search_archives("langgraph")) == 1


def test_missing_session_is_not_treated_as_an_empty_archive(tmp_path):
    store = TranscriptStore(tmp_path / "history")
    store.save_current([])
    store.set_current_session_id("current-session")

    with pytest.raises(FileNotFoundError, match="归档不存在"):
        store.load_session("missing-session")
