from agents.writer import _dedupe_citations


def test_dedupe_citations_prefers_first_url_and_removes_duplicates():
    citations = [
        {"index": 1, "title": "A", "url": "https://example.com/a"},
        {"index": 2, "title": "A duplicate", "url": "https://example.com/a"},
        {"index": 3, "title": "B", "url": "https://example.com/b"},
    ]

    deduped = _dedupe_citations(citations)

    assert len(deduped) == 2
    assert deduped[0]["index"] == 1
    assert deduped[1]["index"] == 3
