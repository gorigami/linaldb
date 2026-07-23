"""Integration tests against a real `linal serve` subprocess (see
conftest.py's `linal_server` fixture) -- these are the tests that would
actually catch a wire-shape drift between this client and the real
engine, which fixture-based unit tests (test_wire.py) can't.
"""

import pytest

import linaldb
from linaldb import LinalError


def test_execute_message_result(linal_server, unique_name):
    client = linaldb.connect(linal_server)
    result = client.execute(f"DATASET {unique_name} COLUMNS (id: Int, score: Float)")
    assert isinstance(result, str)
    assert unique_name in result


def test_execute_table_result_with_vector_and_null(linal_server, unique_name):
    client = linaldb.connect(linal_server)
    client.execute(f"DATASET {unique_name} COLUMNS (id: Int, emb: Vector(3)?)")
    client.execute(f"INSERT INTO {unique_name} VALUES (1, [1.0, 2.0, 3.0])")
    client.execute(f"INSERT INTO {unique_name} VALUES (2, null)")

    result = client.execute(f"SELECT * FROM {unique_name} ORDER BY id")

    assert result.columns == ["id", "emb"]
    assert result.rows == [(1, [1.0, 2.0, 3.0]), (2, None)]


def test_execute_raises_linal_error_with_real_server_message(linal_server):
    client = linaldb.connect(linal_server)
    with pytest.raises(LinalError, match="not found"):
        client.execute("SELECT * FROM this_dataset_does_not_exist")


def test_execute_none_result_for_use(linal_server):
    client = linaldb.connect(linal_server)
    client.execute("CREATE DATABASE pytest_use_target")
    result = client.execute("USE pytest_use_target")
    assert result == "Switched to database 'pytest_use_target'"


def test_query_returns_dataframe(linal_server, unique_name):
    pd = pytest.importorskip("pandas")

    client = linaldb.connect(linal_server)
    client.execute(f"DATASET {unique_name} COLUMNS (id: Int, name: String)")
    client.execute(f'INSERT INTO {unique_name} VALUES (1, "alice")')
    client.execute(f'INSERT INTO {unique_name} VALUES (2, "bob")')

    df = client.query(f"SELECT * FROM {unique_name} ORDER BY id")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "name"]
    assert df["name"].tolist() == ["alice", "bob"]


def test_query_rejects_non_table_result(linal_server, unique_name):
    pytest.importorskip("pandas")
    client = linaldb.connect(linal_server)
    with pytest.raises(LinalError, match="table-shaped"):
        client.query(f"DATASET {unique_name} COLUMNS (id: Int)")


def test_x_linal_database_header_targets_correct_database(linal_server, unique_name):
    client = linaldb.connect(linal_server)
    client.execute("CREATE DATABASE pytest_header_target")

    header_client = linaldb.connect(linal_server, database="pytest_header_target")
    header_client.execute(f"DATASET {unique_name} COLUMNS (id: Int)")
    header_client.execute(f"INSERT INTO {unique_name} VALUES (1)")

    # Not visible from the default database's client.
    with pytest.raises(LinalError, match="not found"):
        client.execute(f"SELECT * FROM {unique_name}")

    # Visible via the header-scoped client.
    result = header_client.execute(f"SELECT * FROM {unique_name}")
    assert result.rows == [(1,)]
