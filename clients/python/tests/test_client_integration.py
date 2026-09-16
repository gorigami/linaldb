"""Integration tests against a real `linal serve` subprocess (see
conftest.py's `linal_server` fixture) -- these are the tests that would
actually catch a wire-shape drift between this client and the real
engine, which fixture-based unit tests (test_wire.py) can't.
"""

import warnings

import pytest

import linaldb_server as linaldb
from linaldb_server import LinalError, TensorResult
from linaldb_server.client import _BARE_USE_DATABASE_RE


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
    try:
        result = client.execute("USE pytest_use_target")
        assert result == "Switched to database 'pytest_use_target'"
    finally:
        # A headerless USE now genuinely persists across requests on the
        # shared session-scoped server (see engine v0.1.74's fix) -- every
        # other test in this session sends headerless requests assuming
        # "default", so this must switch back rather than leak state.
        client.execute("USE default")


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


# --- bare-USE concurrency warning ------------------------------------------
#
# Regression coverage for a real, confirmed footgun: two Clients with no
# database= set, driven concurrently, reliably race on the server's shared,
# process-wide active database (100% collision rate across 80 interleaved
# iterations in the original reproduction). Not a contract violation --
# clients/CONTRACT.md documents this as intentional, ambient-session
# behavior -- but silent at the call site until this warning was added.
# See CHANGELOG.md's 0.1.1 entry.


def test_bare_use_database_regex_matches():
    assert _BARE_USE_DATABASE_RE.match("USE research")
    assert _BARE_USE_DATABASE_RE.match("  USE research")
    # Lowercase *identifier* literally named "dataset" -- a real database
    # switch, not the USE DATASET FROM ... import form (keywords are
    # uppercase-only/case-sensitive in this DSL, src/dsl/lexer.rs).
    assert _BARE_USE_DATABASE_RE.match("USE dataset")


def test_bare_use_database_regex_excludes_use_dataset_from():
    assert _BARE_USE_DATABASE_RE.match('USE DATASET FROM "vectors.h5" AS d') is None


def test_bare_use_database_regex_ignores_lowercase_use_keyword():
    # Not a valid statement in the real grammar either (USE is
    # uppercase-only) -- correctly not flagged.
    assert _BARE_USE_DATABASE_RE.match("use research") is None


def test_bare_use_warns_when_database_unset(linal_server):
    client = linaldb.connect(linal_server)
    client.execute("CREATE DATABASE pytest_warn_target")
    try:
        with pytest.warns(UserWarning, match="active database"):
            client.execute("USE pytest_warn_target")
    finally:
        client.execute("USE default")


def test_bare_use_no_warning_when_database_set(linal_server, unique_name):
    admin = linaldb.connect(linal_server)
    try:
        admin.execute(f"CREATE DATABASE {unique_name}")
    except LinalError:
        pass

    client = linaldb.connect(linal_server, database=unique_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.execute(f"USE {unique_name}")
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_use_dataset_from_does_not_warn(linal_server, tmp_path):
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("id,val\n1,2.0\n")

    client = linaldb.connect(linal_server)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            client.execute(f'USE DATASET FROM "{csv_path}" AS ds1')
        except LinalError:
            pass  # only the absence of a warning matters here
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_transpose_over_http_to_numpy_end_to_end(linal_server, unique_name):
    """The real-world regression for the to_numpy() strides/offset bug,
    live over HTTP against a real running server -- not just a hand-built
    fixture. Closes the exact coverage gap (zero to_numpy() tests of any
    kind existed before this) that let the bug ship silently in 0.1.0.
    """
    pytest.importorskip("numpy")

    client = linaldb.connect(linal_server)
    client.execute(f"MATRIX {unique_name} = [[1, 2, 3], [4, 5, 6]]")
    client.execute(f"LET {unique_name}_t = TRANSPOSE {unique_name}")
    result = client.execute(f"SHOW {unique_name}_t")

    assert isinstance(result, TensorResult)
    assert result.to_numpy().tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
