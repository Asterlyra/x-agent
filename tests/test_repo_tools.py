from pathlib import Path

import pytest


def test_repo_tools_read_source_and_hide_sensitive_paths(tmp_path: Path, monkeypatch):
    from agent.tools import repo_tools

    root = tmp_path / "project"
    (root / "agent").mkdir(parents=True)
    (root / "agent" / "graph.py").write_text("第一行\n第二行\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=value", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("secret", encoding="utf-8")
    (root / "data" / "history").mkdir(parents=True)
    (root / "data" / "history" / "current.jsonl").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(repo_tools, "PROJECT_ROOT", root)

    listing = repo_tools.list_repo_files.invoke({"directory": "."})
    content = repo_tools.read_repo_file.invoke({"path": "agent/graph.py", "offset": 2})

    assert listing == "agent/graph.py"
    assert content == "2: 第二行"
    for path in (".env", ".git/config", "data/history/current.jsonl"):
        with pytest.raises(ValueError, match="禁止读取"):
            repo_tools.read_repo_file.invoke({"path": path})


def test_repo_tools_reject_escape_symlink_and_binary(tmp_path: Path, monkeypatch):
    from agent.tools import repo_tools

    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (root / "escape").symlink_to(outside)
    (root / "binary.bin").write_bytes(b"a\x00b")
    monkeypatch.setattr(repo_tools, "PROJECT_ROOT", root)

    with pytest.raises(ValueError, match="项目根"):
        repo_tools.read_repo_file.invoke({"path": str(outside)})
    with pytest.raises(ValueError, match="符号链接"):
        repo_tools.read_repo_file.invoke({"path": "escape"})
    with pytest.raises(ValueError, match="二进制"):
        repo_tools.read_repo_file.invoke({"path": "binary.bin"})


def test_repo_listing_is_stable_and_only_marks_actual_truncation(tmp_path: Path, monkeypatch):
    from agent.tools import repo_tools

    root = tmp_path / "project"
    root.mkdir()
    (root / "b.py").write_text("b", encoding="utf-8")
    (root / "a.py").write_text("a", encoding="utf-8")
    monkeypatch.setattr(repo_tools, "PROJECT_ROOT", root)

    assert repo_tools.list_repo_files.invoke({"limit": 1}) == "a.py\n[最多显示 1 条]"
    assert repo_tools.list_repo_files.invoke({"limit": 2}) == "a.py\nb.py"
