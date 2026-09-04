from agents._token_tracker import _extract_usage, merge_state_usage, total


def test_extract_usage_prefers_token_usage_block():
    usage = _extract_usage(
        {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            }
        }
    )

    assert usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }


def test_merge_state_usage_accumulates_nodes_without_overwriting():
    prev = {
        "planner": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    new = {
        "planner": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "writer": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
    }

    merged = merge_state_usage(prev, new)

    assert merged["planner"]["total_tokens"] == 20
    assert merged["writer"]["total_tokens"] == 50
    assert total(merged)["total_tokens"] == 70
