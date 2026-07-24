import pytest

from linaldb.errors import LinalError
from linaldb.wire import TableResult, TensorResult, unwrap_result, unwrap_value


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
