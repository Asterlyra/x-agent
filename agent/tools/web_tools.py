# agent/tools/web_tools.py —— 网页查询工具集（fetch_url + web_search，Bing 单引擎）
#
# 成功结果带 description 摘要供模型判断相关性；失败抛给 Tool Pipeline 统一标记 error。
# 快失败：8s 超时 + 1 次瞬态重试。
# 网络相关不进自动化测试（慢、不稳定），解析函数保持纯函数可单测。

import html
import ipaddress
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit

from langchain_core.tools import tool

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
REQUEST_TIMEOUT = 8        # 单次请求超时（秒）：快失败，不长时间挂起
MAX_FETCH_BYTES = 300_000
_RETRY_COUNT = 1           # URLError（含 SSL EOF 瞬态）重试次数：1 次 ≈ 最坏 16.5s
_RETRY_DELAY = 0.5
# 串行化搜索引擎请求：并发请求会被反爬拦截（LangGraph 并行调用两个 web_search 就会触发）
_SEARCH_LOCK = threading.Lock()
# 查询语言启发式：含 CJK 走中文市场，否则英文市场（Bing 对英文 query 用 en 页结果更准）
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_HEAD_RATIO = 0.75          # 截断时头部占比（Hermes 同为 75% head + 25% tail）
_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([\w.-]+)", re.I)


def _validate_public_url(url: str) -> str:
    """只允许解析到公开地址的标准 HTTP(S) URL。"""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"URL 无法解析: {url}") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"仅支持公开 http/https URL: {url}")
    if parsed.username or parsed.password:
        raise ValueError("URL 不允许携带用户名或密码")
    default_port = 80 if parsed.scheme == "http" else 443
    if port not in {None, default_port}:
        raise ValueError(f"URL 只允许标准端口 {default_port}")
    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise ValueError("URL 不允许访问本机或局域网主机")
    try:
        addresses = {item[4][0].split("%", 1)[0] for item in socket.getaddrinfo(hostname, port or default_port)}
    except socket.gaierror as exc:
        raise ValueError(f"URL 主机无法解析: {hostname}") from exc
    if not addresses:
        raise ValueError(f"URL 主机没有可用地址: {hostname}")
    for address in addresses:
        try:
            resolved = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError(f"URL 主机返回了无法识别的地址: {address}") from exc
        if not resolved.is_global:
            raise ValueError(f"URL 不允许访问非公开地址: {address}")
    return url


class _PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    """在每次 HTTP 重定向前重复公开地址校验。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _decode_payload(payload: bytes, content_type: str = "") -> str:
    """根据响应头或 HTML 声明解码网页，兼容 UTF-8 与中文旧编码。"""
    candidates: list[str] = []
    bom_encodings = (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    )
    for bom, encoding in bom_encodings:
        if payload.startswith(bom):
            candidates.append(encoding)
            break

    header_match = _CHARSET_RE.search(content_type)
    if header_match:
        candidates.append(header_match.group(1))

    # 只检查开头少量字节，避免为识别 meta charset 扫描整页。
    head = payload[:8192].decode("ascii", errors="ignore")
    meta_match = _CHARSET_RE.search(head)
    if meta_match:
        candidates.append(meta_match.group(1))

    # 没有声明时先尝试 UTF-8，再尝试中文网页常见的 GB18030。
    candidates.extend(("utf-8", "gb18030", "big5", "latin-1"))
    seen: set[str] = set()
    best: tuple[int, str] | None = None
    for encoding in candidates:
        normalized = encoding.lower().replace("_", "-")
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            text = payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
        replacements = text.count("\ufffd")
        if best is None or replacements < best[0]:
            best = (replacements, text)
        if replacements == 0:
            return text
    return best[1] if best is not None else payload.decode("utf-8", errors="replace")


def _fetch_text(url: str, timeout: float = REQUEST_TIMEOUT) -> str:
    """抓取网页原始 HTML（浏览器 UA + 超时 + 限量读取），URLError 重试 1 次。

    为什么要重试：国内站点过代理时 SSL EOF 是瞬态错误（实测时好时坏），
    重试能显著提高成功率；HTTP 状态异常（如 202 反爬）不算网络错误，不重试。
    """
    _validate_public_url(url)
    opener = urllib.request.build_opener(_PublicRedirectHandler())
    for attempt in range(_RETRY_COUNT + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with opener.open(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                return _decode_payload(resp.read(MAX_FETCH_BYTES), content_type)
        except urllib.error.URLError as e:
            if attempt < _RETRY_COUNT:
                time.sleep(_RETRY_DELAY)
                continue
            raise


def _html_to_text(raw: str) -> str:
    """粗略的 HTML → 纯文本：剥掉 script/style 与标签，折叠空白。"""
    raw = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text)


def _page_title(raw: str) -> str:
    """提取页面 <title>（抓取结果的标题，供 fetch_url 展示）。"""
    m = re.search(r"<title[^>]*>([\s\S]*?)</title>", raw, re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip())[:200] if m else ""


def _is_nginx_placeholder(raw: str) -> bool:
    """识别 nginx 默认占位页：真实存在但无内容的域名（模型应换 URL）。"""
    return "Welcome to nginx" in raw and "If you see this page" in raw


def _parse_bing_results(page: str, max_results: int) -> list[dict]:
    """从 Bing 结果页解析结构化结果列表（<li class="b_algo"> 块）。

    每项含 title / url / description / position 四个字段（Hermes 契约的缩放版：
    description 让模型先判断相关性再决定是否 fetch_url）。纯函数，不碰网络。
    2026-08 实测：Bing 改版后 <h2> 带 class 属性（<h2 class="">），
    标题链接也移到 <a target="_blank" ... href=...>——正则需允许 h2/a 带任意属性。
    """
    results = []
    for m in re.finditer(r'<li class="b_algo[^"]*"([\s\S]*?)</li>', page):
        block = m.group(1)
        link = re.search(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a></h2>', block)
        if not link:
            continue
        desc = re.search(r'<p[^>]*class="b_lineclamp[^"]*"[^>]*>([\s\S]*?)</p>', block)
        if not desc:
            desc = re.search(r'<div class="b_caption"[^>]*>[\s\S]*?<p[^>]*>([\s\S]*?)</p>', block)
        results.append({
            "title": html.unescape(re.sub(r"<[^>]+>", "", link.group(2)).strip()),
            "url": html.unescape(link.group(1)),
            "description": (
                html.unescape(re.sub(r"<[^>]+>", "", desc.group(1))).strip()[:300]
                if desc else ""
            ),
            "position": len(results) + 1,
        })
        if len(results) >= max_results:
            break
    return results


def _truncate_head_tail(text: str, max_chars: int) -> str:
    """长文 head+tail 截断（Hermes 同款：75% 头 + 25% 尾，行边界对齐）。

    返回截断后的文本；全文在预算内则原样返回。截断时附省略说明，模型知道丢了什么。
    """
    if len(text) <= max_chars:
        return text
    head_budget = int(max_chars * _HEAD_RATIO)
    head = text[:head_budget]
    tail = text[-(max_chars - head_budget):]
    nl = head.rfind("\n")              # 头部切回行边界
    if nl > head_budget * 0.5:
        head = head[:nl]
    nl = tail.find("\n")               # 尾部切到行边界
    if 0 <= nl < (max_chars - head_budget) * 0.5:
        tail = tail[nl + 1:]
    return (
        f"{head}\n\n[... 中间省略：页面共 {len(text)} 字符，"
        f"只显示首尾约 {len(head) + len(tail)} 字符，如需完整内容请缩小抓取范围 ...]\n\n{tail}"
    )


def _search_bing(query: str, mkt: str, max_results: int) -> tuple[list[dict], str]:
    """单引擎搜索：Bing 一次调用（快失败，不跨引擎死磕）。

    返回 (results, error)：error 为空表示成功（results 可能为空=无结果），
    工具入口负责把 error 转成异常交给 Tool Pipeline。
    """
    url = f"https://cn.bing.com/search?q={quote(query)}&mkt={mkt}"
    try:
        raw = _fetch_text(url)
    except Exception as e:
        return [], f"Bing 搜索失败: {type(e).__name__}: {e}"
    if _is_nginx_placeholder(raw):
        return [], "Bing 返回了占位页，可能被临时限制，请稍后再试"
    return _parse_bing_results(raw, max_results), ""


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """查询实时信息/查找未知事实时调用（Bing 搜索，无需 API key），返回结构化 JSON：标题、链接、描述摘要列表——先看描述判断相关性，需要正文再配合 fetch_url 抓取（只读）。

何时用：需要最新信息（新闻、比分、价格、政策）、验证未知事实、找网页时。
边界：不搜本地文件或会话历史；只返回链接和摘要，不返回正文（正文用 fetch_url）。
失败模式：搜索失败会成为工具错误；结果为空时换关键词再试（可加 site:域名、-排除词、"精确短语"等操作符，Bing 可能支持）。
参数示例：query="2026世界杯冠军"，max_results=5。"""
    try:
        max_results = min(max(1, int(max_results)), 10)
    except (TypeError, ValueError):
        max_results = 5
    mkt = "zh-CN" if _CJK_RE.search(query) else "en-US"
    # 全程持锁串行化请求（并发必被反爬拦截）
    with _SEARCH_LOCK:
        results, error = _search_bing(query, mkt, max_results)
    if error:
        raise RuntimeError(error)
    if not results:
        return json.dumps({
            "success": True,
            "data": {"web": []},
            "message": "没有找到相关结果，建议更换关键词或改用英文/中文再试",
        }, ensure_ascii=False)
    return json.dumps({"success": True, "data": {"web": results}}, ensure_ascii=False)


@tool
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """抓取网页标题与正文纯文本（前 max_chars 字符；长页只显示首尾并说明省略），供阅读链接内容用（只读）。

何时用：用户要看某个链接的具体内容，或 web_search 的描述摘要不够时。
边界：仅支持 http/https；不搜索（搜索用 web_search）。
失败模式：本机/私网/非标准端口会被拒绝；超时、占位页和空正文作为工具错误返回。"""
    _validate_public_url(url)
    try:
        max_chars = min(max(500, int(max_chars)), 20000)
    except (TypeError, ValueError):
        max_chars = 4000
    try:
        raw = _fetch_text(url)
    except Exception as e:
        raise RuntimeError(f"无法访问该页面: {type(e).__name__}: {e}") from e
    if _is_nginx_placeholder(raw):
        raise RuntimeError("该 URL 是 nginx 默认占位页（域名可能已停用或路径不对）")
    title = _page_title(raw)
    body = _html_to_text(raw).strip()
    if not body:
        raise RuntimeError(f"该页面没有可提取的文本内容: {url}")
    content = _truncate_head_tail(body, max_chars)
    return f"标题: {title}\n\n{content}" if title else content
