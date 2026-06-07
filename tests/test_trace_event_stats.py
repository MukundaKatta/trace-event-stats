"""Tests for trace-event-stats.

Uses the Python standard-library ``unittest`` framework only — no
third-party test dependencies. Run with::

    python -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make the ``src`` layout package importable without an editable/pip install,
# so the suite runs under a bare ``python -m unittest discover -s tests``.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trace_event_stats import DistStats, EventStats, compute_stats, stats_file


BASE = {"tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01, "duration_ms": 200}


class ComputeStatsBasicTests(unittest.TestCase):
    def test_empty_events(self):
        stats = compute_stats([])
        self.assertEqual(stats.event_count, 0)
        self.assertEqual(stats.total_tokens, 0)
        self.assertEqual(stats.total_cost_usd, 0.0)
        self.assertEqual(stats.error_count, 0)
        self.assertEqual(stats.kinds, [])
        self.assertEqual(stats.models, [])

    def test_single_event(self):
        stats = compute_stats([BASE])
        self.assertEqual(stats.event_count, 1)
        self.assertEqual(stats.total_tokens_in, 100)
        self.assertEqual(stats.total_tokens_out, 50)
        self.assertEqual(stats.total_tokens, 150)

    def test_total_cost(self):
        stats = compute_stats([{"cost_usd": 0.01}, {"cost_usd": 0.02}])
        self.assertAlmostEqual(stats.total_cost_usd, 0.03)

    def test_tokens_totals(self):
        events = [
            {"tokens_in": 100, "tokens_out": 50},
            {"tokens_in": 200, "tokens_out": 100},
        ]
        stats = compute_stats(events)
        self.assertEqual(stats.total_tokens_in, 300)
        self.assertEqual(stats.total_tokens_out, 150)
        self.assertEqual(stats.total_tokens, 450)

    def test_event_count(self):
        stats = compute_stats([BASE, BASE, BASE])
        self.assertEqual(stats.event_count, 3)

    def test_returns_event_stats_instance(self):
        self.assertIsInstance(compute_stats([BASE]), EventStats)

    def test_numeric_string_values_are_coerced(self):
        # _first_float coerces numeric strings (e.g. CSV-derived traces).
        stats = compute_stats([{"cost_usd": "0.01"}, {"cost_usd": 0.02}])
        self.assertAlmostEqual(stats.total_cost_usd, 0.03)

    def test_non_numeric_values_are_ignored(self):
        stats = compute_stats([{"tokens_in": "lots"}])
        self.assertEqual(stats.tokens_in.count, 0)
        self.assertEqual(stats.total_tokens_in, 0)


class VariantFieldNameTests(unittest.TestCase):
    def test_accepts_input_tokens_variant(self):
        self.assertEqual(compute_stats([{"input_tokens": 77}]).total_tokens_in, 77)

    def test_accepts_prompt_tokens_variant(self):
        self.assertEqual(compute_stats([{"prompt_tokens": 88}]).total_tokens_in, 88)

    def test_accepts_completion_tokens_variant(self):
        self.assertEqual(compute_stats([{"completion_tokens": 30}]).total_tokens_out, 30)

    def test_accepts_output_tokens_variant(self):
        self.assertEqual(compute_stats([{"output_tokens": 40}]).total_tokens_out, 40)

    def test_accepts_cost_variant(self):
        self.assertAlmostEqual(compute_stats([{"cost": 0.005}]).total_cost_usd, 0.005)

    def test_accepts_price_usd_variant(self):
        self.assertAlmostEqual(compute_stats([{"price_usd": 0.007}]).total_cost_usd, 0.007)

    def test_accepts_latency_ms_variant(self):
        self.assertAlmostEqual(compute_stats([{"latency_ms": 300}]).mean_duration_ms, 300.0)

    def test_accepts_elapsed_ms_variant(self):
        self.assertEqual(compute_stats([{"elapsed_ms": 400}]).duration_ms.count, 1)

    def test_canonical_name_wins_over_variant(self):
        # tokens_in is checked before input_tokens, so it takes precedence.
        stats = compute_stats([{"tokens_in": 100, "input_tokens": 999}])
        self.assertEqual(stats.total_tokens_in, 100)


class KindModelTests(unittest.TestCase):
    def test_kinds_collected(self):
        events = [
            {"kind": "llm_call", "tokens_in": 10},
            {"kind": "tool_call", "tokens_in": 5},
            {"kind": "llm_call", "tokens_in": 20},
        ]
        stats = compute_stats(events)
        self.assertIn("llm_call", stats.kinds)
        self.assertIn("tool_call", stats.kinds)
        self.assertEqual(len(stats.kinds), 2)

    def test_kinds_sorted(self):
        stats = compute_stats([{"kind": "z"}, {"kind": "a"}, {"kind": "m"}])
        self.assertEqual(stats.kinds, ["a", "m", "z"])

    def test_models_collected(self):
        stats = compute_stats([{"model": "claude-sonnet-4-5"}, {"model": "gpt-5.4"}])
        self.assertIn("claude-sonnet-4-5", stats.models)
        self.assertIn("gpt-5.4", stats.models)

    def test_accepts_event_type_as_kind(self):
        self.assertIn("llm_call", compute_stats([{"event_type": "llm_call"}]).kinds)

    def test_accepts_type_as_kind(self):
        self.assertIn("span", compute_stats([{"type": "span"}]).kinds)

    def test_accepts_model_id_variant(self):
        self.assertIn("claude-haiku", compute_stats([{"model_id": "claude-haiku"}]).models)

    def test_non_string_kind_ignored(self):
        self.assertEqual(compute_stats([{"kind": 123}]).kinds, [])


class ErrorCountingTests(unittest.TestCase):
    def test_error_count(self):
        events = [{"error": "timeout"}, {"error": None}, {"tokens_in": 5}]
        self.assertEqual(compute_stats(events).error_count, 1)

    def test_error_via_err_field(self):
        self.assertEqual(compute_stats([{"err": "connection refused"}]).error_count, 1)

    def test_error_via_exception_field(self):
        self.assertEqual(compute_stats([{"exception": "ValueError"}]).error_count, 1)

    def test_no_errors(self):
        self.assertEqual(compute_stats([BASE]).error_count, 0)

    def test_errors_returns_list(self):
        stats = compute_stats([{"error": "bad"}, BASE])
        self.assertEqual(len(stats.errors()), 1)

    def test_falsy_error_not_counted(self):
        self.assertEqual(compute_stats([{"error": ""}, {"error": 0}]).error_count, 0)


class DistStatsTests(unittest.TestCase):
    def test_empty(self):
        d = DistStats.from_values([])
        self.assertEqual(d.count, 0)
        self.assertEqual(d.total, 0.0)
        self.assertIsNone(d.min)
        self.assertIsNone(d.max)
        self.assertIsNone(d.mean)
        self.assertIsNone(d.p50)
        self.assertIsNone(d.p99)

    def test_single(self):
        d = DistStats.from_values([42.0])
        self.assertEqual(d.count, 1)
        self.assertEqual(d.total, 42.0)
        self.assertEqual(d.min, 42.0)
        self.assertEqual(d.max, 42.0)
        self.assertEqual(d.mean, 42.0)
        # With a single data point every percentile collapses to that value.
        self.assertEqual(d.p50, 42.0)
        self.assertEqual(d.p95, 42.0)

    def test_mean(self):
        self.assertAlmostEqual(DistStats.from_values([10.0, 20.0, 30.0]).mean, 20.0)

    def test_p50_is_median(self):
        self.assertAlmostEqual(DistStats.from_values([1.0, 2.0, 3.0]).p50, 2.0)

    def test_p50_even_count_interpolates(self):
        # Linear-interpolation median of [1,2,3,4] sits halfway between 2 and 3.
        self.assertAlmostEqual(DistStats.from_values([1.0, 2.0, 3.0, 4.0]).p50, 2.5)

    def test_p95_basic(self):
        values = [float(v) for v in range(1, 101)]  # 1..100
        self.assertAlmostEqual(DistStats.from_values(values).p95, 95.05, delta=1.0)

    def test_p99_high(self):
        values = [float(v) for v in range(1, 101)]
        d = DistStats.from_values(values)
        self.assertGreater(d.p99, 98.0)
        self.assertLessEqual(d.p99, 100.0)

    def test_min_max(self):
        d = DistStats.from_values([5.0, 1.0, 9.0, 3.0])
        self.assertEqual(d.min, 1.0)
        self.assertEqual(d.max, 9.0)

    def test_total_sums_values(self):
        self.assertAlmostEqual(DistStats.from_values([1.5, 2.5, 4.0]).total, 8.0)

    def test_percentiles_are_monotonic(self):
        d = DistStats.from_values([float(v) for v in range(1, 1001)])
        self.assertLessEqual(d.p50, d.p90)
        self.assertLessEqual(d.p90, d.p95)
        self.assertLessEqual(d.p95, d.p99)


class FilteringTests(unittest.TestCase):
    def test_by_kind_filters(self):
        events = [
            {"kind": "llm_call", "tokens_in": 100},
            {"kind": "tool_call", "tokens_in": 5},
        ]
        llm = compute_stats(events).by_kind("llm_call")
        self.assertEqual(llm.event_count, 1)
        self.assertEqual(llm.total_tokens_in, 100)

    def test_by_kind_empty(self):
        stats = compute_stats([{"kind": "llm_call", "tokens_in": 100}])
        self.assertEqual(stats.by_kind("nonexistent").event_count, 0)

    def test_by_kind_returns_event_stats(self):
        stats = compute_stats([{"kind": "llm_call"}])
        self.assertIsInstance(stats.by_kind("llm_call"), EventStats)

    def test_by_model_filters(self):
        events = [
            {"model": "claude-sonnet-4-5", "cost_usd": 0.01},
            {"model": "gpt-5.4", "cost_usd": 0.02},
        ]
        claude = compute_stats(events).by_model("claude-sonnet-4-5")
        self.assertAlmostEqual(claude.total_cost_usd, 0.01)

    def test_by_model_empty(self):
        stats = compute_stats([{"model": "claude-sonnet-4-5"}])
        self.assertEqual(stats.by_model("unknown").event_count, 0)


class ConveniencePropertyTests(unittest.TestCase):
    def test_mean_duration_ms_none_when_empty(self):
        self.assertIsNone(compute_stats([{"tokens_in": 5}]).mean_duration_ms)

    def test_p95_duration_ms(self):
        events = [{"duration_ms": float(i)} for i in range(1, 101)]
        stats = compute_stats(events)
        self.assertIsNotNone(stats.p95_duration_ms)
        self.assertGreater(stats.p95_duration_ms, 90)

    def test_total_tokens_in_is_int(self):
        self.assertIsInstance(compute_stats([{"tokens_in": 100}]).total_tokens_in, int)


class SummaryTests(unittest.TestCase):
    def test_summary_not_empty(self):
        s = compute_stats([BASE]).summary()
        self.assertIn("events", s)
        self.assertIn("tokens_in", s)

    def test_summary_contains_cost(self):
        s = compute_stats([{"cost_usd": 0.123456}]).summary()
        self.assertIn("0.123456", s)

    def test_summary_includes_duration_lines_when_present(self):
        s = compute_stats([BASE]).summary()
        self.assertIn("dur_ms p50", s)

    def test_summary_omits_duration_lines_when_absent(self):
        s = compute_stats([{"tokens_in": 5}]).summary()
        self.assertNotIn("dur_ms", s)


class StatsFileTests(unittest.TestCase):
    def _write_jsonl(self, lines):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self.addCleanup(os.remove, path)
        return path

    def test_basic(self):
        path = self._write_jsonl(
            [json.dumps(BASE), json.dumps({"tokens_in": 50, "cost_usd": 0.005})]
        )
        self.assertEqual(stats_file(path).event_count, 2)

    def test_skips_blank_lines(self):
        path = self._write_jsonl([json.dumps(BASE), "", json.dumps(BASE)])
        self.assertEqual(stats_file(path).event_count, 2)

    def test_skips_malformed(self):
        path = self._write_jsonl([json.dumps(BASE), "NOT JSON", json.dumps(BASE)])
        self.assertEqual(stats_file(path).event_count, 2)

    def test_aggregates_values_from_file(self):
        path = self._write_jsonl(
            [json.dumps({"cost_usd": 0.01}), json.dumps({"cost_usd": 0.02})]
        )
        self.assertAlmostEqual(stats_file(path).total_cost_usd, 0.03)


if __name__ == "__main__":
    unittest.main()
