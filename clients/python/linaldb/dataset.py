"""`/delivery` dataset export — see clients/CONTRACT.md §2. This is the
checkpoint-2 priority workload (saved-dataset Parquet export), chosen
over ad-hoc /execute query-result ergonomics in the design discussion
that started this plan.
"""

from __future__ import annotations

import io
import json

from .errors import LinalError


def _vector_or_matrix_columns(schema: dict) -> dict:
    """Map column name -> declared value_type for every Vector/Matrix
    column in a `schema.json` document (contract §2's "authoritative
    column typing" — used to detect the legacy JSON-string fallback
    encoding, since that can't be told apart from a real string column
    by looking at the Parquet file alone).
    """
    result = {}
    for col in schema["columns"]:
        vt = col["value_type"]
        if isinstance(vt, dict) and ("Vector" in vt or "Matrix" in vt):
            result[col["name"]] = vt
    return result


def _unwrap_json_fallback_cell(raw: str | None):
    if raw is None:
        return None
    parsed = json.loads(raw)
    # Same tagged shape as a live /execute Value cell (contract §3) --
    # {"Vector": [...]} / {"Matrix": [[...]]} -- reuse the same unwrap.
    from .wire import unwrap_value

    return unwrap_value(parsed)


class Dataset:
    def __init__(self, client, name: str):
        self._client = client
        self.name = name

    def _delivery_url(self, path: str) -> str:
        return f"{self._client.url}/delivery/datasets/{self.name}/{path}"

    def _get_json(self, path: str) -> dict:
        resp = self._client._session.get(self._delivery_url(path), timeout=30)
        if resp.status_code != 200:
            raise LinalError(
                f"GET /delivery/datasets/{self.name}/{path} failed "
                f"(HTTP {resp.status_code}): {resp.text}"
            )
        return resp.json()

    def manifest(self) -> dict:
        return self._get_json("manifest.json")

    def schema(self) -> dict:
        return self._get_json("schema.json")

    def stats(self) -> dict:
        return self._get_json("stats.json")

    def to_arrow(self):
        """Read `data.parquet` as a `pyarrow.Table`, transparently
        unwrapping any column that landed in the legacy JSON-string
        fallback encoding (contract §2) back into a real nested-list
        column — the caller never sees the raw `{"Vector": [...]}` text.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        resp = self._client._session.get(self._delivery_url("data.parquet"), timeout=60)
        if resp.status_code != 200:
            raise LinalError(
                f"GET /delivery/datasets/{self.name}/data.parquet failed "
                f"(HTTP {resp.status_code}): {resp.text}"
            )
        table = pq.read_table(io.BytesIO(resp.content))

        vector_or_matrix = _vector_or_matrix_columns(self.schema())
        for col_name in vector_or_matrix:
            if col_name not in table.column_names:
                continue
            idx = table.column_names.index(col_name)
            field_type = table.schema.field(idx).type
            if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
                # Legacy fallback encoding -- unwrap each cell's JSON text.
                # Note: pa.array() infers float64 for a plain Python list of
                # floats, so this path widens f32 -> f64 (the values
                # themselves are unaffected -- the JSON text already fixed
                # their precision at serde_json::to_string time on the
                # server). The native FixedSizeList path is genuinely
                # float32, matching the engine's f32-only Value::Float type.
                raw_values = table.column(idx).to_pylist()
                unwrapped = [_unwrap_json_fallback_cell(v) for v in raw_values]
                table = table.set_column(idx, col_name, pa.array(unwrapped))
            # Otherwise it's already a native FixedSizeList column
            # (pyarrow/pandas read it correctly with no help needed).

        return table

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
