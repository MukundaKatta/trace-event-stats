"""Aggregate stats from agent JSONL trace events.

Computes totals and percentile distributions for tokens, cost, duration,
and event counts — without pandas or numpy.

Zero dependencies — standard library only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Numeric distribution stats
# ---------------------------------------------------------------------------

@dataclass
class DistStats:
    """Descriptive statistics for a numeric field across events.

    All values are None if no data points were collected.
    """

    count: int
    total: float
    min: float | None
    max: float | None
    mean: float | None
    p50: float | None  # median
    p90: float | None
    p95: float | None
    p99: float | None

    @classmethod
    def from_values(cls, values: list[float]) -> "DistStats":
        if not values:
            return cls(count=0, total=0.0, min=None, max=None, mean=None, p50=None, p90=None, p95=None, p99=None)
        sorted_v = sorted(values)
        n = len(sorted_v)
        total = sum(sorted_v)
        return cls(
            count=n,
            total=total,
            min=sorted_v[0],
            max=sorted_v[-1],
            mean=total / n,
            p50=_percentile(sorted_v, 50),
            p90=_percentile(sorted_v, 90),
            p95=_percentile(sorted_v, 95),
            p99=_percentile(sorted_v, 99),
        )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear interpolation percentile (sorted input required)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = (pct / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_values[-1]
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


# ---------------------------------------------------------------------------
# EventStats — the main result type
# ---------------------------------------------------------------------------

@dataclass
class EventStats:
    """Aggregate statistics over a collection of agent trace events.

    Fields follow the canonical names from trace-field-normalize:
    tokens_in, tokens_out, cost_usd, duration_ms.
    """

    event_count: int
    kinds: list[str]                    # sorted unique values of 'kind' field
    models: list[str]                   # sorted unique values of 'model' field

    tokens_in: DistStats
    tokens_out: DistStats
    total_tokens: int                   # tokens_in.total + tokens_out.total
    cost_usd: DistStats
    duration_ms: DistStats

    error_count: int                    # events where 'error' field is present and truthy
    _raw_events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def by_kind(self, kind: str) -> "EventStats":
        """Return EventStats restricted to events with the given kind."""
        filtered = [e for e in self._raw_events if e.get("kind") == kind]
        return compute_stats(filtered)

    def by_model(self, model: str) -> "EventStats":
        """Return EventStats restricted to events with the given model."""
        filtered = [e for e in self._raw_events if e.get("model") == model]
        return compute_stats(filtered)

    def errors(self) -> list[dict[str, Any]]:
        """Return events that have an error field."""
        return [e for e in self._raw_events if e.get("error")]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        return self.cost_usd.total

    @property
    def total_tokens_in(self) -> int:
        return int(self.tokens_in.total)

    @property
    def total_tokens_out(self) -> int:
        return int(self.tokens_out.total)

    @property
    def mean_duration_ms(self) -> float | None:
        return self.duration_ms.mean

    @property
    def p95_duration_ms(self) -> float | None:
        return self.duration_ms.p95

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"events:    {self.event_count}",
            f"errors:    {self.error_count}",
            f"kinds:     {', '.join(self.kinds) or '—'}",
            f"models:    {', '.join(self.models) or '—'}",
            f"tokens_in: {self.total_tokens_in:,}",
            f"tokens_out:{self.total_tokens_out:,}",
            f"cost_usd:  ${self.total_cost_usd:.6f}",
        ]
        if self.duration_ms.count:
            lines += [
                f"dur_ms p50:{self.duration_ms.p50:.1f}",
                f"dur_ms p95:{self.duration_ms.p95:.1f}",
                f"dur_ms max:{self.duration_ms.max:.1f}",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def compute_stats(events: list[dict[str, Any]]) -> EventStats:
    """Compute aggregate statistics over a list of event dicts.

    Recognizes canonical field names (tokens_in, tokens_out, cost_usd,
    duration_ms, kind, model, error) as output by trace-field-normalize.
    Also accepts raw variant names for convenience.
    """
    tok_in: list[float] = []
    tok_out: list[float] = []
    costs: list[float] = []
    durations: list[float] = []
    kinds: set[str] = set()
    models: set[str] = set()
    error_count = 0

    for e in events:
        # tokens_in — accept a few common variants
        v = _first_float(e, "tokens_in", "input_tokens", "prompt_tokens")
        if v is not None:
            tok_in.append(v)

        # tokens_out
        v = _first_float(e, "tokens_out", "output_tokens", "completion_tokens")
        if v is not None:
            tok_out.append(v)

        # cost
        v = _first_float(e, "cost_usd", "cost", "price_usd")
        if v is not None:
            costs.append(v)

        # duration
        v = _first_float(e, "duration_ms", "latency_ms", "elapsed_ms")
        if v is not None:
            durations.append(v)

        # kind
        kind = e.get("kind") or e.get("event_type") or e.get("type")
        if kind and isinstance(kind, str):
            kinds.add(kind)

        # model
        model = e.get("model") or e.get("model_id") or e.get("model_name")
        if model and isinstance(model, str):
            models.add(model)

        # errors
        if e.get("error") or e.get("err") or e.get("exception"):
            error_count += 1

    tokens_in_dist = DistStats.from_values(tok_in)
    tokens_out_dist = DistStats.from_values(tok_out)

    result = EventStats(
        event_count=len(events),
        kinds=sorted(kinds),
        models=sorted(models),
        tokens_in=tokens_in_dist,
        tokens_out=tokens_out_dist,
        total_tokens=int(tokens_in_dist.total + tokens_out_dist.total),
        cost_usd=DistStats.from_values(costs),
        duration_ms=DistStats.from_values(durations),
        error_count=error_count,
    )
    result._raw_events = list(events)
    return result


def stats_file(path: str | Path) -> EventStats:
    """Load a JSONL trace file and compute stats over all events."""
    p = Path(path)
    events: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # skip malformed lines
    return compute_stats(events)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_float(
    d: dict[str, Any], *keys: str
) -> float | None:
    """Return the first non-None numeric value from the given keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None
