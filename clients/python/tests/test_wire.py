import pytest

from linaldb_server.errors import LinalError
from linaldb_server.wire import TableResult, TensorResult, unwrap_result, unwrap_value


def test_unwrap_value_scalars():
    assert unwrap_value({"Float": 1.5}) == 1.5
    assert unwrap_value({"Int": 5}) == 5
    assert unwrap_value({"String": "x"}) == "x"
    assert unwrap_value({"Bool": True}) is True


def test_unwrap_value_null_is_bare_string_not_object():
    # Confirmed against a live v0.1.72 server (see clients/CONTRACT.md
    # §3) -- Value::Null is a unit variant, serializes as "Null", not
    # {"Null": ...}. This is the exact case an assumption-only wire
    # contract would get wrong.
    assert unwrap_value("Null") is None


def test_unwrap_value_vector_and_matrix():
    assert unwrap_value({"Vector": [1.0, 2.0, 3.0]}) == [1.0, 2.0, 3.0]
    assert unwrap_value({"Matrix": [[1.0, 0.0], [0.0, 1.0]]}) == [[1.0, 0.0], [0.0, 1.0]]


def test_unwrap_value_rejects_unknown_shape():
    with pytest.raises(LinalError):
        unwrap_value({"Unknown": 1})
    with pytest.raises(LinalError):
        unwrap_value(42)


def test_unwrap_result_none_and_message():
    assert unwrap_result(None) is None
    assert unwrap_result({"Message": "Switched to database 'default'"}) == "Switched to database 'default'"


# Real payload captured from a live v0.1.72 server (SELECT * FROM probe
# where probe is (id: Int, emb: Vector(3)?) with rows (1, [1,2,3]) and
# (2, NULL)) -- see clients/CONTRACT.md's verified Table shape.
REAL_TABLE_PAYLOAD = {
    "id": 0,
    "schema": {
        "fields": [
            {"name": "id", "value_type": "Int", "nullable": False, "is_lazy": False},
            {"name": "emb", "value_type": {"Vector": 3}, "nullable": True, "is_lazy": False},
        ],
        "field_indices": {"id": 0, "emb": 1},
    },
    "rows": [
        {
            "schema": {"fields": [], "field_indices": {}},  # real payload repeats full schema; irrelevant to unwrapping
            "values": [{"Int": 1}, {"Vector": [1.0, 2.0, 3.0]}],
        },
        {
            "schema": {"fields": [], "field_indices": {}},
            "values": [{"Int": 2}, "Null"],
        },
    ],
    "metadata": {"name": "Query Result", "row_count": 2},
}


def test_unwrap_result_table_matches_verified_wire_shape():
    result = unwrap_result({"Table": REAL_TABLE_PAYLOAD})
    assert isinstance(result, TableResult)
    assert result.columns == ["id", "emb"]
    assert result.rows == [(1, [1.0, 2.0, 3.0]), (2, None)]


def test_unwrap_result_tensor_table_not_yet_implemented():
    # Deliberately unimplemented -- no confirmed real wire example yet.
    # See checkpoint 1 findings in PYTHON_R_INTEROP_PLAN.md.
    with pytest.raises(LinalError):
        unwrap_result({"TensorTable": [{}, []]})


def test_unwrap_result_tensor():
    payload = {
        "Tensor": {
            "id": "t1",
            "shape": {"dims": [2, 3]},
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "strides": [3, 1],
            "offset": 0,
        }
    }
    result = unwrap_result(payload)
    assert isinstance(result, TensorResult)
    assert result.shape == [2, 3]
    assert result.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_unwrap_result_rejects_unknown_variant():
    with pytest.raises(LinalError):
        unwrap_result({"NotARealVariant": {}})
    with pytest.raises(LinalError):
        unwrap_result({"Message": "a", "extra": "b"})


# --- TensorResult.to_numpy(): strides/offset-aware reconstruction ---------
#
# Regression coverage for a real, shipped bug (linaldb-server 0.1.0):
# to_numpy() used to do a plain `reshape(self.shape)` on the raw `data`
# buffer, ignoring `strides`/`offset` entirely -- silently wrong for any
# non-contiguous tensor (e.g. a zero-copy TRANSPOSE). See CHANGELOG.md's
# 0.1.1 entry and clients/CONTRACT.md's Tensor/LazyTensor wire shape note.


def test_to_numpy_contiguous_unchanged():
    pytest.importorskip("numpy")
    import numpy as np

    result = TensorResult(shape=[2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], strides=[3, 1], offset=0)
    np.testing.assert_array_equal(result.to_numpy(), np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"))


def test_to_numpy_respects_strides_and_offset_transposed():
    pytest.importorskip("numpy")

    # Real payload confirmed live against a v0.1.82 server:
    # MATRIX m = [[1, 2, 3], [4, 5, 6]]; LET mt = TRANSPOSE m; SHOW mt.
    # The buggy code returned [[1, 2], [3, 4], [5, 6]] (a plain reshape
    # of the untransposed buffer) instead of the correct transpose below.
    result = TensorResult(shape=[3, 2], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], strides=[1, 3], offset=0)
    assert result.to_numpy().tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_to_numpy_missing_strides_defaults_to_contiguous():
    pytest.importorskip("numpy")

    result = TensorResult(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])  # strides=None, offset=0 (defaults)
    assert result.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_to_numpy_scalar_rank_zero():
    pytest.importorskip("numpy")

    result = TensorResult(shape=[], data=[7.0])
    arr = result.to_numpy()
    assert arr.shape == ()
    assert float(arr) == 7.0


def test_to_numpy_offset_nonzero():
    pytest.importorskip("numpy")

    # Simulates a sliced view: two leading elements skipped via offset.
    result = TensorResult(shape=[2, 2], data=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0], strides=[2, 1], offset=2)
    assert result.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_to_numpy_returns_owned_copy():
    pytest.importorskip("numpy")

    result = TensorResult(shape=[2], data=[1.0, 2.0])
    arr = result.to_numpy()
    arr[0] = 999.0
    assert result.data == [1.0, 2.0]


def test_to_numpy_rejects_out_of_bounds():
    pytest.importorskip("numpy")

    # shape/strides/offset claim 3 elements; data only has 1.
    result = TensorResult(shape=[3], data=[1.0], strides=[1], offset=0)
    with pytest.raises(LinalError):
        result.to_numpy()
