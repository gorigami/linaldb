"""In-process tests for the embedded native bindings -- no server, no
subprocess, unlike `clients/python/tests/test_client_integration.py`
(which needs a real `linal serve`). Each test gets its own `Db` pointed
at a fresh `tmp_path`-backed data dir so tests can't see each other's
datasets/databases.
"""

import pytest

import linaldb_embedded as linaldb
from linaldb_embedded import ExecuteResult, LinalError


@pytest.fixture
def db(tmp_path):
    return linaldb.Db(data_dir=str(tmp_path / "data"))


def test_execute_message_result(db):
    result = db.execute("DATASET t COLUMNS (id: Int, score: Float)")
    assert isinstance(result, str)
    assert "t" in result


def test_execute_table_result_with_vector_and_null(db):
    db.execute("DATASET t COLUMNS (id: Int, emb: Vector(3)?)")
    db.execute("INSERT INTO t VALUES (1, [1.0, 2.0, 3.0])")
    db.execute("INSERT INTO t VALUES (2, null)")

    result = db.execute("SELECT * FROM t ORDER BY id")

    assert isinstance(result, ExecuteResult)
    assert result.columns == ["id", "emb"]
    assert result.rows == [[1, [1.0, 2.0, 3.0]], [2, None]]


def test_execute_raises_linal_error_for_unknown_dataset(db):
    with pytest.raises(LinalError, match="not found"):
        db.execute("SELECT * FROM this_dataset_does_not_exist")


def test_execute_raises_linal_error_for_parse_error(db):
    with pytest.raises(LinalError, match="Parse error"):
        db.execute("THIS IS NOT VALID DSL")


def test_execute_none_result_for_use(db):
    db.execute("CREATE DATABASE other")
    result = db.execute("USE other")
    assert result == "Switched to database 'other'"
    assert db.active_db() == "other"


def test_query_returns_dataframe(db):
    pd = pytest.importorskip("pandas")

    db.execute("DATASET t COLUMNS (id: Int, name: String)")
    db.execute('INSERT INTO t VALUES (1, "alice")')
    db.execute('INSERT INTO t VALUES (2, "bob")')

    df = db.query("SELECT * FROM t ORDER BY id")

    assert isinstance(df, pd.DataFrame)
    assert list(df["name"]) == ["alice", "bob"]


def test_execute_result_to_pandas(db):
    pytest.importorskip("pandas")

    db.execute("DATASET t COLUMNS (id: Int)")
    db.execute("INSERT INTO t VALUES (1)")
    result = db.execute("SELECT * FROM t")

    df = result.to_pandas()
    assert list(df.columns) == ["id"]
    assert list(df["id"]) == [1]


def test_dataset_round_trip_to_pandas(db):
    pytest.importorskip("pandas")

    db.execute("DATASET t COLUMNS (id: Int, score: Float)")
    db.execute("INSERT INTO t VALUES (1, 0.9)")
    db.execute("INSERT INTO t VALUES (2, 0.4)")
    db.execute("SAVE DATASET t")

    dataset = db.dataset("t")
    assert dataset.schema()["columns"][0]["name"] == "id"
    assert "stats" in dataset.manifest() or dataset.manifest()

    df = dataset.to_pandas()
    assert sorted(df["id"].tolist()) == [1, 2]


def test_dataset_dir_matches_data_dir_and_active_db(db, tmp_path):
    db.execute("DATASET t COLUMNS (id: Int)")
    db.execute("SAVE DATASET t")

    expected = str(tmp_path / "data" / "default" / "datasets" / "t")
    assert db.dataset_dir("t") == expected


def test_dataset_missing_raises_linal_error(db):
    with pytest.raises(LinalError):
        db.dataset("never_saved").schema()
