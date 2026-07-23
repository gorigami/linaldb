# linaldb (Python client)

Python client for [LINALDB](../../README.md) — a SQL-inspired analytical
engine treating vectors, matrices, and tensors as first-class citizens.

**Status**: scaffolding only (checkpoint 0 of
[`PYTHON_R_INTEROP_PLAN.md`](../../PYTHON_R_INTEROP_PLAN.md)) — not
functional yet, not published to PyPI.

Talks to a running `linal serve` instance over its HTTP API
(`POST /execute` for ad-hoc DSL, `/delivery/*` for real Parquet dataset
export) — see [`../CONTRACT.md`](../CONTRACT.md) for the exact wire
contract this client implements against. No compiled extension; requires
`linal serve` running an engine version `>= 0.1.72` (the version that
fixed Vector/Matrix columns to encode as native Arrow types in delivered
Parquet rather than JSON strings).

## Planned usage (not yet implemented)

```python
import linaldb

client = linaldb.connect("http://localhost:8080")
df = client.query("SELECT id, embedding FROM docs WHERE score > 0.8")

dataset = client.dataset("my_dataset")
df = dataset.to_pandas()          # requires the `pandas` extra
table = dataset.to_arrow()        # pyarrow.Table, no extra required
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
