from agents.writer import _append_references, _build_references


def test_build_references_uses_supplied_indexes_and_urls():
    citations = [
        {"index": 1, "title": "来源一", "url": "https://example.com/one"},
        {"index": 2, "title": "来源二", "url": "https://example.com/two"},
    ]

    result = _build_references(citations)

    assert "[1] 来源一 - https://example.com/one" in result
    assert "[2] 来源二 - https://example.com/two" in result


def test_append_references_adds_web_and_evidence_sections():
    citations = [
        {"index": 1, "title": "网络来源", "url": "https://example.com/report"},
    ]
    evidence_pack = [
        {
            "evidence_id": "E-1",
            "kind": "kb",
            "title": "本地文档",
            "source": "C:/docs/local.txt",
        },
        {
            "evidence_id": "W-1",
            "kind": "web",
            "title": "网络来源",
            "url": "https://example.com/report",
        },
    ]

    result = _append_references("## 结论\n结论内容", citations, evidence_pack)

    assert "## 参考资料" in result
    assert "[1] 网络来源 - https://example.com/report" in result
    assert "## 证据追踪" in result
    assert "[E-1] KB | 本地文档 | C:/docs/local.txt" in result
    assert "[W-1] WEB | 网络来源 | https://example.com/report" in result


def test_append_references_does_not_duplicate_existing_sections():
    draft = (
        "## 结论\n已有结论\n\n"
        "## 参考资料\n[1] 已有来源 - https://example.com/one\n\n"
        "## 证据追踪\n[E-1] KB | 文档 | local.txt"
    )
    citations = [{"index": 1, "title": "已有来源", "url": "https://example.com/one"}]
    evidence_pack = [
        {"evidence_id": "E-1", "kind": "kb", "title": "文档", "source": "local.txt"}
    ]

    assert _append_references(draft, citations, evidence_pack) == draft


def test_append_references_leaves_draft_unchanged_without_sources():
    draft = "## 结论\n没有外部来源"
    assert _append_references(draft, [], []) == draft
