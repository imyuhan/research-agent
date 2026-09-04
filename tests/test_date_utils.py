from datetime import date, timedelta

import pytest

from agents.tools.date_utils import date_window, detect_date, to_chinese


def test_detect_date_supports_chinese_date():
    assert detect_date("请研究 2026年8月6日 的发布情况") == "2026-08-06"


def test_detect_date_supports_iso_date():
    assert detect_date("发生在 2026-08-06 的事件") == "2026-08-06"


def test_detect_date_supports_month_and_day_without_year():
    current_year = date.today().year
    assert detect_date("请查询 8月6日 的新闻") == f"{current_year}-08-06"


@pytest.mark.parametrize(
    ("word", "offset"),
    [("今天", 0), ("昨天", -1), ("明天", 1), ("今日", 0), ("昨日", -1)],
)
def test_detect_date_supports_relative_dates(word, offset):
    expected = date.today() + timedelta(days=offset)
    assert detect_date(f"{word} 的研究结果") == expected.isoformat()


def test_detect_date_returns_none_for_missing_or_invalid_date():
    assert detect_date("") is None
    assert detect_date("2026年2月30日") is None
    assert detect_date("这是一段没有日期的问题") is None


def test_to_chinese_and_date_window():
    assert to_chinese("2026-08-06") == "2026年8月6日"
    assert date_window("2026-08-06", before=2, after=3) == (
        "2026-08-04",
        "2026-08-09",
    )


def test_to_chinese_rejects_invalid_iso_date():
    with pytest.raises(ValueError, match="非法 ISO 日期"):
        to_chinese("not-a-date")
