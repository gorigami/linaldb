# linaldb-server (Python client)

Python client for [LINALDB](../../README.md) — a SQL-inspired analytical
engine treating vectors, matrices, and tensors as first-class citizens.

**Status**: two releases on PyPI. `0.1.0` (2026-09-09) was the initial
HTTP client — `/execute` and `/delivery` both implemented and tested
against a real server, plus a real end-to-end example (see below). `0.1.1`
(2026-09-16) fixed a real, silent data-corruption bug found via
real-world testing in the `linal-hub` sibling project:
`TensorResult.to_numpy()` ignored the wire payload's own `strides`/
`offset` fields, so a non-contiguous tensor result (e.g. `SHOW` of a
zero-copy `TRANSPOSE`) came back with plausible-looking but wrong values,
no exception — `to_numpy()` now reconstructs correctly, and a
`UserWarning` was added for the concurrent-session footgun described
below. See [`CHANGELOG.md`](CHANGELOG.md) for full detail. Verified
against engine `0.1.82`. Published on PyPI as
[`linaldb-server`](https://pypi.org/project/linaldb-server/):

```bash
pip install linaldb-server
```

Talks to a running `linal serve` instance over its HTTP API
(`POST /execute` for ad-hoc DSL, `/delivery/*` for real Parquet dataset
export) — see [`../CONTRACT.md`](../CONTRACT.md) for the exact wire
contract this client implements against. No compiled extension; requires
`linal serve` running an engine version `>= 0.1.74` (the version that
fixed `USE <database>` sent to `/execute` to actually persist, and fixed
`/delivery` to honor a non-default database).

## Usage

```python
import linaldb_server as linaldb

client = linaldb.connect("http://localhost:8080")
df = client.query("SELECT id, embedding FROM docs WHERE score > 0.8")

dataset = client.dataset("my_dataset")
df = dataset.to_pandas()          # requires the `pandas` extra
table = dataset.to_arrow()        # pyarrow.Table, no extra required
```

A `Client`/`connect()` call with no `database=` operates on the server's
*shared*, process-wide active database — matching the embedded CLI/REPL,
where `USE <db>` persists for the whole session. Pass `database="..."` for
an isolated session instead (every request then carries `X-Linal-Database`
and is fully scoped, regardless of what any other client does concurrently).
A bare `USE <db>` sent by a `Client` with no `database=` raises a
`UserWarning`, since that statement mutates state shared with every other
client connected to the same server.

See [`examples/digit_classification.py`](examples/digit_classification.py)
for a complete real-data walkthrough: it starts a real `linal serve`,
replays a real UCI handwritten-digits classification workflow through
this client, queries the result via `/execute`, exports the same data via
`/delivery`, and independently recomputes the classification from the raw
exported vectors to confirm both paths agree exactly.

## About

[LINALDB](https://github.com/gorigami/linaldb) is built by
[Gorigami](https://gorigami.xyz), a software company based in Colombia, and
maintained by Nicolás Balaguera. See the [project README](../../README.md)
and [LICENSE](../../LICENSE) for the full picture and licensing terms.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Requires `cargo build --bin linal` to have been run in the repo root
first — the integration tests and the example script both launch a real
`linal serve` from the built binary (skipped, not failed, if it's
missing).
