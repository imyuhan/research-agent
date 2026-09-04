from agents.evidence_utils import select_balanced_evidence


def test_select_balanced_evidence_interleaves_kb_and_web_items():
    evidence = [
        {"evidence_id": "KB-1", "kind": "kb"},
        {"evidence_id": "KB-2", "kind": "kb"},
        {"evidence_id": "KB-3", "kind": "kb"},
        {"evidence_id": "W-1", "kind": "web"},
        {"evidence_id": "W-2", "kind": "web"},
    ]

    selected = select_balanced_evidence(evidence, max_items=4)

    assert [item["evidence_id"] for item in selected] == ["KB-1", "W-1", "KB-2", "W-2"]
