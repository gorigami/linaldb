# linaldb (Python client)

Python client for [LINALDB](../../README.md) — a SQL-inspired analytical
engine treating vectors, matrices, and tensors as first-class citizens.

**Status**: functional through checkpoint 5 of
[`PYTHON_R_INTEROP_PLAN.md`](../../PYTHON_R_INTEROP_PLAN.md) — `/execute`
and `/delivery` are both implemented and tested against a real server (22
passing `pytest` tests, plus a real end-to-end example, see below). Not
yet published to PyPI.

Talks to a running `linal serve` instance over its HTTP API
(`POST /execute` for ad-hoc DSL, `/delivery/*` for real Parquet dataset
export) — see [`../CONTRACT.md`](../CONTRACT.md) for the exact wire
contract this client implements against. No compiled extension; requires
`linal serve` running an engine version `>= 0.1.74` (the version that
fixed `USE <database>` sent to `/execute` to actually persist, and fixed
`/delivery` to honor a non-default database).

## Usage

```python
import linaldb

client = linaldb.connect("http://localhost:8080")
df = client.query("SELECT id, embedding FROM docs WHERE score > 0.8")

dataset = client.dataset("my_dataset")
df = dataset.to_pandas()          # requires the `pandas` extra
table = dataset.to_arrow()        # pyarrow.Table, no extra required
```

See [`examples/digit_classification.py`](examples/digit_classification.py)
for a complete real-data walkthrough: it starts a real `linal serve`,
replays a real UCI handwritten-digits classification workflow through
this client, queries the result via `/execute`, exports the same data via
`/delivery`, and independently recomputes the classification from the raw
exported vectors to confirm both paths agree exactly.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Requires `cargo build --bin linal` to have been run in the repo root
first — the integration tests and the example script both launch a real
`linal serve` from the built binary (skipped, not failed, if it's
missing).
