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

__version__ = "0.1.16"

__all__ = [
    "Db",
    "Dataset",
    "ExecuteResult",
    "TensorResult",
    "LinalError",
    "bitvector_array",
    "peaks_array",
    "sparse_array",
]


def bitvector_array(bits):
    """A `pyarrow.FixedSizeBinaryArray` of packed bit vectors from a 2-D
    boolean (or 0/1) NumPy array of shape `(n, nbits)`, in the layout a
    `BitVector(nbits)` column loads from (`numpy.packbits`, MSB first). Give
    the field `{"linal.logical_value_type": f"BitVector:{nbits}"}` metadata
    when `nbits` isn't a multiple of 8, so the exact length survives.
    """
    import numpy as np
    import pyarrow as pa

    bits = np.asarray(bits)
    packed = np.packbits(bits.astype(bool), axis=1)
    width = packed.shape[1]
    return pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(width), len(packed), [None, pa.py_buffer(packed.tobytes())]
    )


def sparse_array(rows, dim):
    """`(array, field_metadata)` for a `SparseVector(dim)` column, from a
    list of `(indices, values)` pairs (or `None` for a NULL row). Indices
    must be strictly increasing and below `dim`; the engine checks. Use the
    metadata on the table's field so the dimension is known::

        arr, meta = linaldb.sparse_array(rows, 10000)
        table = pa.Table.from_arrays([arr], schema=pa.schema([pa.field("s", arr.type, metadata=meta)]))
    """
    import numpy as np
    import pyarrow as pa

    indices, values, mask = [], [], []
    for row in rows:
        if row is None:
            indices.append([])
            values.append([])
            mask.append(True)
            continue
        idx, vals = row
        idx = np.asarray(idx, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float32)
        if idx.shape != vals.shape or idx.ndim != 1:
            raise LinalError("sparse_array: each row needs 1-D indices and values of equal length")
        if (idx < 0).any() or (idx > np.iinfo(np.uint32).max).any():
            raise LinalError("sparse_array: indices must be non-negative and fit in 32 bits")
        indices.append(idx.astype(np.uint32))
        values.append(vals)
        mask.append(False)
    struct = pa.StructArray.from_arrays(
        [
            pa.array(indices, type=pa.list_(pa.field("item", pa.uint32(), nullable=False))),
            pa.array(values, type=pa.list_(pa.field("item", pa.float32(), nullable=False))),
        ],
        names=["indices", "values"],
        mask=pa.array(mask),
    )
    return struct, {"linal.logical_value_type": f"SparseVector:{dim}"}


def peaks_array(spectra):
    """A `pyarrow` array for a `Matrix(2, *)` peak-list column (one spectrum
    per row: row 0 the m/z values, ascending; row 1 the intensities), from
    a list of `(mz, intensities)` pairs of equal-length 1-D arrays. Values
    are stored as float32. Pass it to `Db.load_arrow()` in a table.
    """
    import numpy as np
    import pyarrow as pa

    rows = []
    for i, (mz, intensities) in enumerate(spectra):
        mz = np.asarray(mz, dtype=np.float32)
        intensities = np.asarray(intensities, dtype=np.float32)
        if mz.ndim != 1 or mz.shape != intensities.shape:
            raise LinalError(
                f"peaks_array: spectrum {i} needs two 1-D arrays of the same length, "
                f"got {mz.shape} and {intensities.shape}"
            )
        rows.extend([mz, intensities])
    lists = pa.array(rows, type=pa.list_(pa.field("item", pa.float32(), nullable=False)))
    return pa.FixedSizeListArray.from_arrays(lists, 2)


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

    def load_arrow(self, name: str, data, *, origin: str = "arrow") -> int:
        """Create dataset `name` from in-memory Arrow data -- a
        `pyarrow.Table`, a `pyarrow.RecordBatch`, or anything
        `pyarrow.table()` accepts (e.g. an object exposing
        `__arrow_c_stream__`). No file and no DSL parsing in between.

        Column types map like the engine's own Parquet packages:
        int64/int32 -> Int, float32 -> Float, float64 -> Float64,
        string/large_string -> String, bool -> Bool,
        fixed_size_list<float32> -> Vector(d), and
        fixed_size_list<fixed_size_list<float32>> -> Matrix(r, c).
        Any other type, NaN or infinite values, or an existing dataset
        named `name` raise `LinalError`. The load is recorded in the
        lineage log under `origin`. Returns the number of rows loaded.
        """
        import pyarrow as pa

        if isinstance(data, pa.RecordBatch):
            table = pa.Table.from_batches([data])
        elif isinstance(data, pa.Table):
            table = data
        else:
            table = pa.table(data)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return self._native.load_arrow_ipc(name, sink.getvalue().to_pybytes(), origin)

    def load_numpy(
        self,
        name: str,
        vectors,
        *,
        column: str = "embedding",
        columns: dict | None = None,
        bit_columns: dict | None = None,
        quantize: str | None = None,
        origin: str = "numpy",
    ) -> int:
        """Create dataset `name` from a 2-D `float32` NumPy array of shape
        `(n, d)`: one `Vector(d)` column named `column`, plus optional
        scalar columns from `columns` (a dict of name -> 1-D array of length
        `n`, placed before the vector column, in dict order). Values are
        loaded bit-exact; a `float64` array is rejected rather than rounded
        silently -- cast it with `.astype(numpy.float32)` first.
        `quantize="F16"` or `"I8"` stores the vector column as `Vector(d, F16)`
        / `Vector(d, I8)` (quantized once, on load; see the DSL reference).
        `bit_columns` adds `BitVector` columns (e.g. fingerprints): a dict of
        name -> 2-D boolean (or 0/1) array of shape `(n, nbits)`; bit `j` of
        row `i` is `array[i, j]`. Returns the number of rows loaded. See
        `load_arrow()` for the type mapping and errors.
        """
        import numpy as np
        import pyarrow as pa

        arr = np.asarray(vectors)
        if arr.ndim != 2:
            raise LinalError(
                f"load_numpy expects a 2-D array of shape (n, d), got shape {arr.shape}"
            )
        if arr.dtype != np.float32:
            raise LinalError(
                f"load_numpy expects float32 vectors, got {arr.dtype} -- "
                "cast explicitly with .astype(numpy.float32)"
            )
        n, d = arr.shape
        if d == 0:
            raise LinalError("load_numpy: vectors have dimension 0")
        flat = pa.array(np.ascontiguousarray(arr).reshape(-1), type=pa.float32())
        names, arrays = [], []
        for col_name, values in (columns or {}).items():
            values = np.asarray(values)
            if values.ndim != 1 or len(values) != n:
                raise LinalError(
                    f"load_numpy: column '{col_name}' has shape {values.shape}, "
                    f"expected ({n},) to match the {n} vectors"
                )
            names.append(col_name)
            arrays.append(pa.array(values))
        for col_name, bits in (bit_columns or {}).items():
            bits = np.asarray(bits)
            if bits.ndim != 2 or bits.shape[0] != n:
                raise LinalError(
                    f"load_numpy: bit column '{col_name}' has shape {bits.shape}, "
                    f"expected ({n}, nbits)"
                )
            if not np.isin(bits, (0, 1)).all():
                raise LinalError(f"load_numpy: bit column '{col_name}' may only hold 0/1 or booleans")
            names.append(col_name)
            arrays.append(bitvector_array(bits))
        if column in names:
            raise LinalError(
                f"load_numpy: column '{column}' is both the vector column and a scalar column"
            )
        names.append(column)
        arrays.append(pa.FixedSizeListArray.from_arrays(flat, d))
        fields = []
        for col_name, arr in zip(names, arrays):
            field = pa.field(col_name, arr.type)
            if col_name == column and quantize:
                enc = quantize.upper()
                if enc not in ("F16", "I8"):
                    raise LinalError(f"load_numpy: quantize must be 'F16' or 'I8', got {quantize!r}")
                field = field.with_metadata({"linal.logical_value_type": f"QVector:{d},{enc}"})
            if col_name in (bit_columns or {}):
                nbits = np.asarray(bit_columns[col_name]).shape[1]
                field = field.with_metadata({"linal.logical_value_type": f"BitVector:{nbits}"})
            fields.append(field)
        table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
        return self.load_arrow(name, table, origin=origin)

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
