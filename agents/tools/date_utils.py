import re
from datetime import date, timedelta

_RELATIVE = [
    ("今天", 0), ("今日", 0), ("当天", 0),
    ("昨天", -1), ("昨日", -1),
    ("明天", 1), ("明日", 1),
]

_YMD_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_MD_RE = re.compile(r"(?<![\d年])(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")


def _build(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def detect_date(text: str) -> str | None:
    """从文本中提取用户明确指定的日期，返回 'YYYY-MM-DD'；未指定返回 None"""
    if not text:
        return None
    today = date.today()

    m = _YMD_RE.search(text)
    if m:
        return _build(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = _MD_RE.search(text)
    if m:
        return _build(today.year, int(m.group(1)), int(m.group(2)))

    m = _ISO_RE.search(text)
    if m:
        return _build(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    for word, offset in _RELATIVE:
        if word in text:
            return (today + timedelta(days=offset)).isoformat()

    return None


def to_chinese(iso_date: str) -> str:
    """'2026-08-06' -> '2026年8月6日'"""
    try:
        d = date.fromisoformat(iso_date)
    except ValueError:
        raise ValueError(f"非法 ISO 日期: {iso_date!r}") from None
    return f"{d.year}年{d.month}月{d.day}日"


def date_window(iso_date: str, before: int = 1, after: int = 1) -> tuple[str, str]:
    """返回围绕指定日期的搜索窗口 [D-before, D+after]，均为 'YYYY-MM-DD'"""
    d = date.fromisoformat(iso_date)
    return (d - timedelta(days=before)).isoformat(), (d + timedelta(days=after)).isoformat()
