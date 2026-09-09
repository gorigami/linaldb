# linaldb (Python native bindings)

Embedded native Python bindings for [LINALDB](../../README.md) — a PyO3
extension linking the same synchronous engine the CLI/REPL runs
(`TensorDb`/`execute_line`, `src/engine/db.rs` + `src/dsl/mod.rs`)
directly into your Python process. **No server, no network** — this is
the "use it like SQLite" story, as opposed to [`clients/python`](../python)
(a thin HTTP client for a running `linal serve`). Not yet published to
PyPI; see [`../EMBEDDED_CONTRACT.md`](../EMBEDDED_CONTRACT.md) for the
exact result shapes this module implements.

## Usage

```python
import linaldb

db = linaldb.Db()  # persists to ./data by default, exactly like the CLI
db.execute("CREATE DATASET t COLUMNS (id: Int, score: Float)")
db.execute("INSERT INTO t VALUES (1, 0.9), (2, 0.4)")

result = db.execute("SELECT * FROM t WHERE score > 0.5")
print(result.columns, result.rows)   # ExecuteResult
df = result.to_pandas()              # requires the `pandas` extra

db.execute("SAVE DATASET t")
dataset = db.dataset("t")
df = dataset.to_pandas()             # reads data.parquet directly off disk
```

See
[`examples/digit_classification_embedded.py`](examples/digit_classification_embedded.py)
and the Jupyter notebook
[`examples/digit_classification_embedded.ipynb`](examples/digit_classification_embedded.ipynb)
for a complete real-data walkthrough — the same real UCI
handwritten-digits classification workflow
[`clients/python/examples/digit_classification.py`](../python/examples/digit_classification.py)
runs over HTTP, ported to embedded mode: no `linal serve` subprocess at
all, just an in-process `Db()`, an in-process classification query, and
an independent numpy recomputation from the dataset export, cross-checked
against the SQL result.

## Build

This crate vendors HDF5 and pulls in arrow/tokio/axum/zarrs as
dependencies of the root `linal` crate (`Cargo.toml` here has a `path`
dependency on `../..`) — the first build compiles all of that, same as
building the CLI itself; a `.cargo/config.toml` here points the build at
the repo-root `target/` dir to avoid a second full vendor build if you've
already built the root crate.

Requires a Rust toolchain (`cmake` + a C toolchain for HDF5, per the root
`CLAUDE.md`) and [`maturin`](https://www.maturin.rs/):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop           # builds the extension, installs it into .venv
```

If your Python interpreter is newer than PyO3's currently-supported
maximum CPython version, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` for
the build (`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop`) — this
repo was built and verified against CPython 3.14 that way.

## Development

```bash
source .venv/bin/activate
pip install -e ".[dev]"
maturin develop
pytest tests/
```

No live server needed for anything here (that's the whole point of
embedded mode) — tests and examples all run against an in-process `Db()`.
