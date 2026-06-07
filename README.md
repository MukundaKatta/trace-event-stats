# trace-event-stats

[![CI](https://github.com/MukundaKatta/trace-event-stats/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/trace-event-stats/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Aggregate stats from agent JSONL trace events — tokens, cost, duration
percentiles, error counts, and by-kind / by-model filtering. **Zero
dependencies**, standard library only (`json`, `math`, `dataclasses`,
`pathlib`).

When an agent or LLM pipeline emits a stream of trace events (one JSON
object per line), this library turns that stream into a compact set of
descriptive statistics: totals, means, and `p50 / p90 / p95 / p99`
percentile distributions — without pulling in pandas or numpy.

## Install

```bash
pip install trace-event-stats
```

## Quick start

```python
from trace_event_stats import compute_stats

events = [
    {"kind": "llm_call", "model": "claude-sonnet-4-5",
     "tokens_in": 1200, "tokens_out": 350, "cost_usd": 0.018, "duration_ms": 820},
    {"kind": "tool_call", "tokens_in": 0, "tokens_out": 0, "duration_ms": 45},
    {"kind": "llm_call", "model": "claude-sonnet-4-5",
     "tokens_in": 900, "tokens_out": 120, "cost_usd": 0.011, "duration_ms": 610,
     "error": "rate_limited"},
]

stats = compute_stats(events)
print(stats.summary())
```

Output:

```text
events:    3
errors:    1
kinds:     llm_call, tool_call
models:    claude-sonnet-4-5
tokens_in: 2,100
tokens_out:470
cost_usd:  $0.029000
dur_ms p50:610.0
dur_ms p95:799.0
dur_ms max:820.0
```

Reach into individual numbers via convenience properties and the
underlying [`DistStats`](#diststats):

```python
stats.event_count        # 3
stats.total_tokens_in    # 2100
stats.total_tokens_out   # 470
stats.total_cost_usd     # 0.029
stats.p95_duration_ms    # 799.0
stats.duration_ms.max    # 820.0
stats.error_count        # 1
```

### From a JSONL file

```python
from trace_event_stats import stats_file

stats = stats_file("traces.jsonl")
print(stats.total_cost_usd)
```

Each line in the file is one JSON object. Blank lines and malformed JSON
lines are skipped rather than raising, so a partially written trace file
still produces usable stats.

### Filtering by kind or model

`by_kind` and `by_model` return a fresh `EventStats` restricted to the
matching events, so every property and percentile recomputes for that
subset:

```python
llm  = stats.by_kind("llm_call")    # only llm_call events
tool = stats.by_kind("tool_call")   # only tool_call events
claude = stats.by_model("claude-sonnet-4-5")

print(llm.event_count)        # 2
print(claude.total_cost_usd)  # 0.029

# Inspect the events that recorded an error:
for ev in stats.errors():
    print(ev["error"])
```

## Event format

An event is a plain `dict`. Every field is optional — events that omit a
field simply don't contribute to that field's statistics. Canonical field
names are recognized first, with a few common aliases accepted for
convenience:

| Statistic     | Canonical field | Accepted aliases                       |
| ------------- | --------------- | -------------------------------------- |
| input tokens  | `tokens_in`     | `input_tokens`, `prompt_tokens`        |
| output tokens | `tokens_out`    | `output_tokens`, `completion_tokens`   |
| cost (USD)    | `cost_usd`      | `cost`, `price_usd`                     |
| duration (ms) | `duration_ms`   | `latency_ms`, `elapsed_ms`             |
| kind          | `kind`          | `event_type`, `type`                   |
| model         | `model`         | `model_id`, `model_name`               |
| error         | `error`         | `err`, `exception`                     |

The canonical names match the output of `trace-field-normalize`. Numeric
fields accept ints, floats, or numeric strings (e.g. `"0.01"`); an event
is counted as an error whenever its error field is present and truthy.

## API reference

### `compute_stats(events: list[dict]) -> EventStats`

Compute aggregate statistics over a list of event dicts.

### `stats_file(path: str | Path) -> EventStats`

Load a JSONL trace file (one JSON object per line) and compute stats over
all parseable events. Blank and malformed lines are skipped.

### `EventStats`

Result of an aggregation.

| Member             | Type        | Description                                          |
| ------------------ | ----------- | ---------------------------------------------------- |
| `event_count`      | `int`       | Number of events aggregated.                         |
| `kinds`            | `list[str]` | Sorted unique `kind` values.                         |
| `models`           | `list[str]` | Sorted unique `model` values.                        |
| `tokens_in`        | `DistStats` | Distribution of input tokens.                        |
| `tokens_out`       | `DistStats` | Distribution of output tokens.                       |
| `total_tokens`     | `int`       | Sum of input + output tokens.                        |
| `cost_usd`         | `DistStats` | Distribution of per-event cost.                      |
| `duration_ms`      | `DistStats` | Distribution of per-event duration.                  |
| `error_count`      | `int`       | Number of events with a truthy error field.          |

Convenience properties: `total_cost_usd`, `total_tokens_in`,
`total_tokens_out`, `mean_duration_ms`, `p95_duration_ms`.

Methods:

- `by_kind(kind)` → `EventStats` restricted to that kind.
- `by_model(model)` → `EventStats` restricted to that model.
- `errors()` → `list[dict]` of events with a truthy error field.
- `summary()` → human-readable multi-line `str`.

### `DistStats`

Descriptive statistics for one numeric field across events. All numeric
members are `None` when no data points were collected.

| Member  | Description                              |
| ------- | ---------------------------------------- |
| `count` | Number of data points.                   |
| `total` | Sum of values.                           |
| `min`   | Smallest value.                          |
| `max`   | Largest value.                           |
| `mean`  | Arithmetic mean.                         |
| `p50`   | Median (50th percentile).                |
| `p90`   | 90th percentile.                         |
| `p95`   | 95th percentile.                         |
| `p99`   | 99th percentile.                         |

Percentiles use linear interpolation between the two nearest ranks, the
same method as numpy's default (`linear`).

`DistStats.from_values(values: list[float]) -> DistStats` builds a
distribution directly from a list of numbers.

## Development

The test suite uses only the standard-library `unittest` framework — no
third-party test dependencies:

```bash
python -m unittest discover -s tests
```

## Zero dependencies

Standard library only: `json`, `math`, `dataclasses`, `pathlib`. Nothing
else.

## License

[MIT](LICENSE)
