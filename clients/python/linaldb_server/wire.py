"""Unwraps the JSON shapes documented in ``clients/CONTRACT.md`` into
plain Python values. Kept separate from ``client.py`` so it can be unit
tested against fixture JSON with no HTTP involved.
"""

from .errors import LinalError

_SCALAR_UNWRAPPERS = {
    "Float": float,
    "Int": int,
    "String": str,
    "Bool": bool,
}


def unwrap_value(value):
    """Unwrap one tagged ``Value`` cell (contract §3) into a plain Python
    value: float/int/str/bool/None, ``list[float]`` for a Vector, or
    ``list[list[float]]`` for a Matrix.
    """
    if value == "Null":
        return None
    if isinstance(value, dict) and len(value) == 1:
        (key, inner), = value.items()
        if key in _SCALAR_UNWRAPPERS:
            return _SCALAR_UNWRAPPERS[key](inner)
        if key == "Vector":
            return [float(x) for x in inner]
        if key == "Matrix":
            return [[float(x) for x in row] for row in inner]
    raise LinalError(f"Unrecognized Value wire shape: {value!r}")


class TableResult:
    """A `Table` result: `.columns` (list of names, in order) and
    `.rows` (list of tuples of already-unwrapped Python values, one
    tuple per row, in column order).
    """

    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows

    def __repr__(self):
        return f"TableResult(columns={self.columns!r}, rows={len(self.rows)})"

    @classmethod
    def from_wire(cls, payload):
        # See CONTRACT.md's verified `Table` shape: top-level `schema`
        # gives column order/names once; each row's cells live under a
        # `values` key (not the row object itself).
        columns = [f["name"] for f in payload["schema"]["fields"]]
        rows = [
            tuple(unwrap_value(v) for v in row["values"])
            for row in payload["rows"]
        ]
        return cls(columns, rows)


def _default_strides(shape):
    """Row-major (C-order) contiguous strides for `shape`, in element
    counts -- matching the engine's `Tensor::compute_default_strides`
    (src/core/tensor.rs): the last dimension has stride 1, accumulating
    right-to-left. Used when the wire payload omits `strides` (the
    assume-contiguous case).
    """
    strides = [0] * len(shape)
    stride = 1
    for i in reversed(range(len(shape))):
        strides[i] = stride
        stride *= shape[i]
    return strides


class TensorResult:
    """A standalone `Tensor`/`LazyTensor` result. Structural shape
    verified against a live server for `shape`/`strides`/`offset`/`data`
    (see `test_transpose_over_http_to_numpy_end_to_end` in
    `test_client_integration.py`) -- CONTRACT.md's original caveat about
    this not being independently verified applied to `TableResult` only
    by the time this was checked.

    `.to_numpy()` honors `strides`/`offset` exactly as the engine defines
    them (element counts, row-major -- see `src/core/tensor.rs`), so a
    zero-copy result like `TRANSPOSE` (which swaps strides and reuses its
    input's buffer as-is, never copying data) reconstructs correctly.
    """

    def __init__(self, shape, data, strides=None, offset=0):
        self.shape = shape
        self.data = data
        self.strides = strides
        self.offset = offset

    def __repr__(self):
        return f"TensorResult(shape={self.shape!r}, len(data)={len(self.data)})"

    @classmethod
    def from_wire(cls, payload):
        return cls(
            shape=payload["shape"]["dims"],
            data=payload["data"],
            strides=payload.get("strides"),
            offset=payload.get("offset", 0),
        )

    def to_numpy(self):
        """Reconstruct this tensor's logical values as an owned
        `numpy.ndarray`. Needed because a zero-copy engine op like
        `TRANSPOSE` (src/engine/kernels.rs) swaps `strides` and reuses
        the pre-transpose buffer as-is -- a plain `reshape` of `data`
        would silently reinterpret the raw buffer in the wrong order
        instead of raising, since it never looks at `strides`/`offset`
        at all.
        """
        import numpy as np

        flat = np.asarray(self.data, dtype="float32")
        shape = tuple(self.shape)
        strides = self.strides if self.strides is not None else _default_strides(shape)

        if len(strides) != len(shape):
            raise LinalError(
                f"Tensor wire payload shape/strides rank mismatch: "
                f"shape={shape!r} strides={strides!r}"
            )

        offset = self.offset or 0
        # Fail loudly instead of letting as_strided silently read past
        # the buffer on a malformed/mismatched payload.
        max_index = offset
        for dim, stride in zip(shape, strides):
            if dim > 0:
                max_index += (dim - 1) * stride
        if flat.size and max_index >= flat.size:
            raise LinalError(
                f"Tensor wire payload out of bounds: shape={shape!r} "
                f"strides={strides!r} offset={offset!r} but data has "
                f"only {flat.size} element(s)"
            )

        itemsize = flat.itemsize
        view = np.lib.stride_tricks.as_strided(
            flat[offset:],
            shape=shape,
            strides=tuple(s * itemsize for s in strides),
            writeable=False,
        )
        # Detach from the raw/possibly-oversized/shared buffer -- callers
        # get a normal, fully-owned array.
        return view.copy()


def unwrap_result(result):
    """Unwrap one `DslOutput` JSON value (contract §1) into `None` (no
    output), a plain `str` (`Message`), a `TableResult`, or a
    `TensorResult`.
    """
    if result is None:
        return None
    if not isinstance(result, dict) or len(result) != 1:
        raise LinalError(f"Unrecognized DslOutput wire shape: {result!r}")
    (kind, payload), = result.items()
    if kind == "Message":
        return payload
    if kind == "Table":
        return TableResult.from_wire(payload)
    if kind == "TensorTable":
        # Not yet exercised against a real response — no confirmed
        # example command produces one in this checkpoint's test pass.
        # Fail loudly rather than guess at the wire shape.
        raise LinalError(
            "TensorTable unwrapping is not yet implemented (no verified "
            "wire example) — see PYTHON_R_INTEROP_PLAN.md checkpoint 1 findings"
        )
    if kind in ("Tensor", "LazyTensor"):
        return TensorResult.from_wire(payload)
    raise LinalError(f"Unknown DslOutput variant: {kind!r}")
