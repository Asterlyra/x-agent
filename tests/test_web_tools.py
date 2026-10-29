import pytest

from agent.tools.web_tools import _decode_payload, _page_title, _validate_public_url


def test_decode_payload_uses_html_gb18030_charset():
    html = '<meta charset="gb18030"><title>价格查询_价格信息（2026年）</title>'
    decoded = _decode_payload(html.encode("gb18030"), "text/html")
    assert "价格查询" in decoded
    assert _page_title(decoded) == "价格查询_价格信息（2026年）"


def test_decode_payload_keeps_utf8_chinese():
    html = "<title>中文标题</title>"
    assert _page_title(_decode_payload(html.encode("utf-8"), "text/html; charset=utf-8")) == "中文标题"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "http://127.0.0.1/",
    "http://[::1]/",
    "https://localhost/",
    "https://example.com:8443/",
])
def test_public_url_validator_rejects_local_or_nonstandard_targets(url):
    with pytest.raises(ValueError):
        _validate_public_url(url)
