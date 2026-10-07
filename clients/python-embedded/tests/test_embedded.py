"""In-process tests for the embedded native bindings -- no server, no
subprocess, unlike `clients/python/tests/test_client_integration.py`
(which needs a real `linal serve`). Each test gets its own `Db` pointed
at a fresh `tmp_path`-backed data dir so tests can't see each other's
datasets/databases.
"""

import pytest

import linaldb
from linaldb import ExecuteResult, LinalError


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


# --- In-memory loading: Db.load_numpy / Db.load_arrow (CASMI P1) ---------


def test_load_numpy_round_trip_is_bit_exact(db):
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((50, 16)).astype(np.float32)
    vecs[0, 0] = np.float32(1e-40)  # subnormal
    vecs[1, 1] = -0.0
    ids = np.arange(50, dtype=np.int64)
    mass = rng.uniform(100, 900, 50)  # float64 -> Float64 column

    n = db.load_numpy("spec", vecs, column="e", columns={"id": ids, "mass": mass})
    assert n == 50

    result = db.execute("SELECT * FROM spec ORDER BY id")
    assert result.columns == ["id", "mass", "e"]
    got = np.array([row[2] for row in result.rows], dtype=np.float32)
    assert got.tobytes() == vecs.tobytes()
    assert [row[1] for row in result.rows] == mass.tolist()


def test_load_numpy_rejects_bad_input(db):
    np = pytest.importorskip("numpy")

    with pytest.raises(LinalError, match="float32"):
        db.load_numpy("a", np.zeros((2, 3)))  # float64
    with pytest.raises(LinalError, match="2-D"):
        db.load_numpy("a", np.zeros(3, dtype=np.float32))
    with pytest.raises(LinalError, match="expected \\(2,\\)"):
        db.load_numpy("a", np.zeros((2, 3), dtype=np.float32), columns={"id": [1, 2, 3]})
    bad = np.ones((2, 3), dtype=np.float32)
    bad[1, 2] = np.nan
    with pytest.raises(LinalError, match="row 1 is NaN"):
        db.load_numpy("a", bad)
    db.load_numpy("a", np.ones((2, 3), dtype=np.float32))
    with pytest.raises(LinalError, match="already exists"):
        db.load_numpy("a", np.ones((2, 3), dtype=np.float32))


def test_load_arrow_table_and_search(db):
    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")

    vecs = np.eye(4, dtype=np.float32)
    table = pa.table(
        {
            "name": ["a", "b", "c", "d"],
            "e": pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1)), 4),
        }
    )
    assert db.load_arrow("lib", table) == 4
    db.execute("CREATE VECTOR INDEX ON lib(e)")
    result = db.execute("SEARCH lib ON e QUERY [0.0, 0.0, 1.0, 0.0] LIMIT 1")
    assert result.rows[0][0] == "c"


def test_load_arrow_rejects_float64_vectors(db):
    pa = pytest.importorskip("pyarrow")

    table = pa.table(
        {"e": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 2.0], type=pa.float64()), 2)}
    )
    with pytest.raises(LinalError, match="float32"):
        db.load_arrow("x", table)


def test_load_is_recorded_in_lineage(db):
    np = pytest.importorskip("numpy")

    db.load_numpy("spec", np.ones((3, 2), dtype=np.float32))
    text = db.execute("EXPLAIN LINEAGE spec AS JSON")
    assert "LOAD FROM MEMORY" in text
    assert "numpy" in text
