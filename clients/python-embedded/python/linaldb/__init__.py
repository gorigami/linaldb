"""Embedded native Python bindings for LINALDB.

Unlike `clients/python` (a thin HTTP client for a running `linal serve`),
this package links the engine directly into the Python process via a PyO3
extension (`._native`) — no server, no network, same synchronous
`TensorDb`/`execute_line` the CLI/REPL runs. See
`clients/EMBEDDED_CONTRACT.md` in the parent repository for the exact
result shapes this module implements, and `clients/CONTRACT.md` for how
that compares to the HTTP client's wire contract.

    import linaldb

    db = linaldb.Db()  # in-memory-backed, persists to ./data like the CLI
    db.execute("CREATE DATASET t COLUMNS (id: Int)")
    result = db.execute("SELECT * FROM t")   # -> ExecuteResult
    df = result.to_pandas()

This module is kept thin on the Rust side (`src/lib.rs` here only
converts `DslOutput`/`Value` into plain Python primitives) — everything
ergonomic (pandas/pyarrow conversion, the `Dataset` file-export handle)
lives in this pure-Python layer, mirroring how `clients/python/linaldb_server`
splits `wire.py` (raw unwrap) from `client.py` (ergonomics).
"""

from __future__ import annotations

import json
from pathlib import Path

from ._native import Db as _NativeDb
from ._native import LinalError

__version__ = "0.1.0"

__all__ = [
    "Db",
    "Dataset",
    "ExecuteResult",
    "TensorResult",
    "LinalError",
]


class ExecuteResult:
    """A table-shaped `execute()` result: `.columns` (list of names, in
    order) and `.rows` (list of lists of already-native Python values,
    one list per row, in column order). Produced for a DSL `Table`/
    `TensorTable` result; other result kinds (`Message`, `None`) are
    returned directly as `str`/`None` from `Db.execute()`, matching the
    HTTP client's `Client.execute()` shape (`clients/python/linaldb_server/client.py`).
    """

    def __init__(self, columns: list[str], rows: list[list]):
        self.columns = columns
        self.rows = rows

    def __repr__(self) -> str:
        return f"ExecuteResult(columns={self.columns!r}, rows={len(self.rows)})"

    def to_pandas(self):
        """Build a `pandas.DataFrame` directly from this result's rows —
        a convenience with no HTTP-client equivalent, since the embedded
        path has no analog to `/delivery`'s Parquet export for ad-hoc
        query results (only saved datasets have an on-disk package; see
        `Db.dataset()`).
        """
        import pandas as pd

        return pd.DataFrame(self.rows, columns=self.columns)


class TensorResult:
    """A standalone `Tensor` result (e.g. `SHOW <name>` on a `LET`-bound
    tensor). `.to_numpy()` assumes a contiguous, zero-offset tensor (the
    common case for a freshly computed result); a non-trivial
    stride/offset view may not reshape correctly -- check `.strides`/
    `.offset` first if unsure. Mirrors
    `clients/python/linaldb_server/wire.py`'s `TensorResult`.
    """

    def __init__(self, shape: list[int], data: list[float], strides=None, offset: int = 0):
        self.shape = shape
        self.data = data
        self.strides = strides
        self.offset = offset

    def __repr__(self) -> str:
        return f"TensorResult(shape={self.shape!r}, len(data)={len(self.data)})"

    def to_numpy(self):
        import numpy as np

        arr = np.asarray(self.data, dtype="float64")
        if self.shape:
            arr = arr.reshape(self.shape)
        return arr


class Dataset:
    """A handle to a saved dataset's on-disk package
    (`{data_dir}/{db}/datasets/{name}/`, `Db.dataset_dir(name)`) —
    `.schema()`/`.stats()`/`.manifest()`/`.to_arrow()`/`.to_pandas()`.

    This is the embedded-mode counterpart of
    `clients/python/linaldb_server/dataset.py`'s `Dataset` (which fetches the
    same files over `/delivery`); here they're just read straight off
    disk, since the engine and this code share a filesystem.
    """

    def __init__(self, db: "Db", name: str):
        self._db = db
        self.name = name

    def _path(self, filename: str) -> Path:
        return Path(self._db.dataset_dir(self.name)) / filename

    def _read_json(self, filename: str) -> dict:
        path = self._path(filename)
        if not path.exists():
            raise LinalError(
                f"{path} does not exist — has dataset '{self.name}' been "
                f"saved yet (SAVE DATASET {self.name})?"
            )
        return json.loads(path.read_text())

    def manifest(self) -> dict:
        return self._read_json("manifest.json")

    def schema(self) -> dict:
        return self._read_json("schema.json")

    def stats(self) -> dict:
        return self._read_json("stats.json")

    def to_arrow(self):
        """Read `data.parquet` as a `pyarrow.Table`. Requires `pyarrow`
        (a hard dependency of this package)."""
        import pyarrow.parquet as pq

        path = self._path("data.parquet")
        if not path.exists():
            raise LinalError(
                f"{path} does not exist — has dataset '{self.name}' been "
                f"saved yet (SAVE DATASET {self.name})?"
            )
        return pq.read_table(path)

    def to_pandas(self):
        """`.to_arrow().to_pandas()`. Requires the `pandas` extra."""
        try:
            import pandas  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Dataset.to_pandas() requires the `pandas` extra: "
                'pip install "linaldb[pandas]"'
            ) from e
        return self.to_arrow().to_pandas()


class Db:
    """An embedded LINALDB instance — an in-process engine, no server.

    `data_dir`, when given, overrides where persistence reads/writes
    (defaults to `./data` relative to the current working directory,
    exactly like the CLI/REPL).
    """

    def __init__(self, data_dir: str | None = None):
        self._native = _NativeDb(data_dir)

    def execute(self, dsl: str):
        """Run one DSL statement and return its unwrapped result: `None`
        (no output), a `str` (`Message`), or an `ExecuteResult`
        (`Table`/`TensorTable`). Raises `LinalError` on a DSL error.
        """
        raw = self._native.execute_raw(dsl)
        if isinstance(raw, dict):
            if "columns" in raw:
                return ExecuteResult(raw["columns"], raw["rows"])
            return TensorResult(raw["shape"], raw["data"], raw["strides"], raw["offset"])
        return raw

    def query(self, dsl: str):
        """Run a DSL command expected to return a table and return a
        `pandas.DataFrame` directly. Requires the `pandas` extra.
        """
        result = self.execute(dsl)
        if not isinstance(result, ExecuteResult):
            raise LinalError(
                f"query() expects a table-shaped result, got {type(result).__name__} "
                "(use execute() for non-table results)"
            )
        return result.to_pandas()

    def dataset(self, name: str) -> Dataset:
        """A handle to a saved dataset's on-disk package —
        `.schema()`/`.stats()`/`.manifest()`/`.to_arrow()`/`.to_pandas()`.
        """
        return Dataset(self, name)

    def active_db(self) -> str:
        """The database currently active on this instance (`USE <db>`
        changes it)."""
        return self._native.active_db()

    def data_dir(self) -> str:
        """The resolved data directory this instance persists to/recovers
        from."""
        return self._native.data_dir()

    def dataset_dir(self, name: str) -> str:
        """The on-disk package directory for a saved dataset —
        `{data_dir}/{active_db}/datasets/{name}/`."""
        return self._native.dataset_dir(name)
