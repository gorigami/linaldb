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


class TensorResult:
    """A standalone `Tensor`/`LazyTensor` result. Structural shape only
    — see CONTRACT.md's caveat that this wasn't independently verified
    against a live response the way `TableResult`'s shape was.

    `.to_numpy()` assumes a contiguous, zero-offset tensor (the common
    case for a freshly computed `LET`/`SHOW` result); if the source is a
    non-trivial stride/offset view, the reshape may not reflect the
    logical data correctly — verify against a live server before relying
    on this for sliced/transposed tensor results.
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
        import numpy as np

        arr = np.asarray(self.data, dtype="float32")
        if self.shape:
            arr = arr.reshape(self.shape)
        return arr


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
