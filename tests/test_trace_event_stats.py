"""Tests for trace-event-stats."""

import json
import tempfile
from pathlib import Path

import pytest

from trace_event_stats import DistStats, EventStats, compute_stats, stats_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_events(*overrides):
    """Build a list of events from keyword dicts."""
    return list(overrides)


BASE = {"tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01, "duration_ms": 200}


# ---------------------------------------------------------------------------
# compute_stats — basic
# ---------------------------------------------------------------------------

def test_empty_events():
    stats = compute_stats([])
    assert stats.event_count == 0
    assert stats.total_tokens == 0
    assert stats.total_cost_usd == 0.0
    assert stats.error_count == 0

def test_single_event():
    stats = compute_stats([BASE])
    assert stats.event_count == 1
    assert stats.total_tokens_in == 100
    assert stats.total_tokens_out == 50
    assert stats.total_tokens == 150

def test_total_cost():
    events = [{"cost_usd": 0.01}, {"cost_usd": 0.02}]
    stats = compute_stats(events)
    assert stats.total_cost_usd == pytest.approx(0.03)

def test_tokens_totals():
    events = [
        {"tokens_in": 100, "tokens_out": 50},
        {"tokens_in": 200, "tokens_out": 100},
    ]
    stats = compute_stats(events)
    assert stats.total_tokens_in == 300
    assert stats.total_tokens_out == 150
    assert stats.total_tokens == 450

def test_event_count():
    events = [BASE, BASE, BASE]
    stats = compute_stats(events)
    assert stats.event_count == 3

# ---------------------------------------------------------------------------
# Variant field names accepted
# ---------------------------------------------------------------------------

def test_accepts_input_tokens_variant():
    stats = compute_stats([{"input_tokens": 77}])
    assert stats.total_tokens_in == 77

def test_accepts_prompt_tokens_variant():
    stats = compute_stats([{"prompt_tokens": 88}])
    assert stats.total_tokens_in == 88

def test_accepts_completion_tokens_variant():
    stats = compute_stats([{"completion_tokens": 30}])
    assert stats.total_tokens_out == 30

def test_accepts_output_tokens_variant():
    stats = compute_stats([{"output_tokens": 40}])
    assert stats.total_tokens_out == 40

def test_accepts_cost_variant():
    stats = compute_stats([{"cost": 0.005}])
    assert stats.total_cost_usd == pytest.approx(0.005)

def test_accepts_latency_ms_variant():
    stats = compute_stats([{"latency_ms": 300}])
    assert stats.mean_duration_ms == pytest.approx(300.0)

def test_accepts_elapsed_ms_variant():
    stats = compute_stats([{"elapsed_ms": 400}])
    assert stats.duration_ms.count == 1

# ---------------------------------------------------------------------------
# kind + model
# ---------------------------------------------------------------------------

def test_kinds_collected():
    events = [
        {"kind": "llm_call", "tokens_in": 10},
        {"kind": "tool_call", "tokens_in": 5},
        {"kind": "llm_call", "tokens_in": 20},
    ]
    stats = compute_stats(events)
    assert "llm_call" in stats.kinds
    assert "tool_call" in stats.kinds
    assert len(stats.kinds) == 2

def test_kinds_sorted():
    events = [{"kind": "z"}, {"kind": "a"}, {"kind": "m"}]
    stats = compute_stats(events)
    assert stats.kinds == ["a", "m", "z"]

def test_models_collected():
    events = [{"model": "claude-sonnet-4-5"}, {"model": "gpt-5.4"}]
    stats = compute_stats(events)
    assert "claude-sonnet-4-5" in stats.models
    assert "gpt-5.4" in stats.models

def test_accepts_event_type_as_kind():
    events = [{"event_type": "llm_call"}]
    stats = compute_stats(events)
    assert "llm_call" in stats.kinds

def test_accepts_model_id_variant():
    events = [{"model_id": "claude-haiku"}]
    stats = compute_stats(events)
    assert "claude-haiku" in stats.models

# ---------------------------------------------------------------------------
# error counting
# ---------------------------------------------------------------------------

def test_error_count():
    events = [
        {"error": "timeout"},
        {"error": None},
        {"tokens_in": 5},
    ]
    stats = compute_stats(events)
    assert stats.error_count == 1

def test_error_via_err_field():
    events = [{"err": "connection refused"}]
    stats = compute_stats(events)
    assert stats.error_count == 1

def test_no_errors():
    stats = compute_stats([BASE])
    assert stats.error_count == 0

def test_errors_returns_list():
    events = [{"error": "bad"}, BASE]
    stats = compute_stats(events)
    errs = stats.errors()
    assert len(errs) == 1

# ---------------------------------------------------------------------------
# DistStats
# ---------------------------------------------------------------------------

def test_dist_stats_empty():
    d = DistStats.from_values([])
    assert d.count == 0
    assert d.total == 0.0
    assert d.min is None
    assert d.mean is None

def test_dist_stats_single():
    d = DistStats.from_values([42.0])
    assert d.count == 1
    assert d.total == 42.0
    assert d.min == 42.0
    assert d.max == 42.0
    assert d.mean == 42.0

def test_dist_stats_mean():
    d = DistStats.from_values([10.0, 20.0, 30.0])
    assert d.mean == pytest.approx(20.0)

def test_dist_stats_p50():
    d = DistStats.from_values([1.0, 2.0, 3.0])
    assert d.p50 == pytest.approx(2.0)

def test_dist_stats_p95_basic():
    values = list(range(1, 101, 1))  # 1..100
    d = DistStats.from_values([float(v) for v in values])
    assert d.p95 == pytest.approx(95.05, abs=1.0)

def test_dist_stats_min_max():
    d = DistStats.from_values([5.0, 1.0, 9.0, 3.0])
    assert d.min == 1.0
    assert d.max == 9.0

# ---------------------------------------------------------------------------
# by_kind + by_model filtering
# ---------------------------------------------------------------------------

def test_by_kind_filters():
    events = [
        {"kind": "llm_call", "tokens_in": 100},
        {"kind": "tool_call", "tokens_in": 5},
    ]
    stats = compute_stats(events)
    llm_stats = stats.by_kind("llm_call")
    assert llm_stats.event_count == 1
    assert llm_stats.total_tokens_in == 100

def test_by_kind_empty():
    events = [{"kind": "llm_call", "tokens_in": 100}]
    stats = compute_stats(events)
    none_stats = stats.by_kind("nonexistent")
    assert none_stats.event_count == 0

def test_by_model_filters():
    events = [
        {"model": "claude-sonnet-4-5", "cost_usd": 0.01},
        {"model": "gpt-5.4", "cost_usd": 0.02},
    ]
    stats = compute_stats(events)
    claude_stats = stats.by_model("claude-sonnet-4-5")
    assert claude_stats.total_cost_usd == pytest.approx(0.01)

# ---------------------------------------------------------------------------
# convenience properties
# ---------------------------------------------------------------------------

def test_mean_duration_ms_none_when_empty():
    stats = compute_stats([{"tokens_in": 5}])  # no duration field
    assert stats.mean_duration_ms is None

def test_p95_duration_ms():
    events = [{"duration_ms": float(i)} for i in range(1, 101)]
    stats = compute_stats(events)
    assert stats.p95_duration_ms is not None
    assert stats.p95_duration_ms > 90

# ---------------------------------------------------------------------------
# summary string
# ---------------------------------------------------------------------------

def test_summary_not_empty():
    stats = compute_stats([BASE])
    s = stats.summary()
    assert "events" in s
    assert "tokens_in" in s

def test_summary_contains_cost():
    stats = compute_stats([{"cost_usd": 0.123456}])
    s = stats.summary()
    assert "0.123456" in s

# ---------------------------------------------------------------------------
# stats_file
# ---------------------------------------------------------------------------

def test_stats_file_basic():
    events = [BASE, {"tokens_in": 50, "cost_usd": 0.005}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        path = f.name
    stats = stats_file(path)
    assert stats.event_count == 2

def test_stats_file_skips_blank_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(BASE) + "\n")
        f.write("\n")
        f.write(json.dumps(BASE) + "\n")
        path = f.name
    stats = stats_file(path)
    assert stats.event_count == 2

def test_stats_file_skips_malformed():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(BASE) + "\n")
        f.write("NOT JSON\n")
        f.write(json.dumps(BASE) + "\n")
        path = f.name
    stats = stats_file(path)
    assert stats.event_count == 2
