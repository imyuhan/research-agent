import pytest

from infrastructure.config import settings
from workflow import route_after_review


def test_approved_report_ends_workflow():
    assert route_after_review({"review_passed": True}) == "approved"


def test_max_revisions_takes_precedence_over_research_route(monkeypatch):
    monkeypatch.setattr(settings, "MAX_REVISIONS", 3)
    state = {
        "review_passed": False,
        "revision_count": 3,
        "needs_new_search": True,
        "route_reason": "needs_research",
    }
    assert route_after_review(state) == "max_retries"


@pytest.mark.parametrize(
    "state",
    [
        {"review_passed": False, "revision_count": 1, "route_reason": "needs_research"},
        {"review_passed": False, "revision_count": 1, "needs_new_search": True},
    ],
)
def test_material_gaps_route_back_to_research(state):
    assert route_after_review(state) == "needs_research"


def test_normal_review_failure_routes_back_to_writer():
    state = {
        "review_passed": False,
        "revision_count": 1,
        "route_reason": "needs_rewrite",
    }
    assert route_after_review(state) == "needs_rewrite"


def test_missing_review_fields_use_safe_rewrite_default():
    assert route_after_review({}) == "needs_rewrite"
