"""Integration tests for Dataset (/delivery Parquet export) against a
real `linal serve` subprocess -- deliberately covers BOTH Parquet
encodings from clients/CONTRACT.md §2: the native FixedSizeList path
(no NULLs) and the legacy JSON-string fallback path (a real NULL
present), since a client that only handles one would silently mis-read
the other.
"""

import pytest

import linaldb


def test_to_arrow_native_vector_column_no_nulls(linal_server, unique_name):
    client = linaldb.connect(linal_server)
    client.execute(f"DATASET {unique_name} COLUMNS (id: Int, emb: Vector(3))")
    client.execute(f"INSERT INTO {unique_name} VALUES (1, [1.0, 2.0, 3.0])")
    client.execute(f"INSERT INTO {unique_name} VALUES (2, [4.0, 5.0, 6.0])")
    client.execute(f"SAVE DATASET {unique_name}")

    table = client.dataset(unique_name).to_arrow()

    assert table.column_names == ["id", "emb"]
    assert table.column("emb").to_pylist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_to_arrow_legacy_fallback_vector_column_with_null(linal_server, unique_name):
    client = linaldb.connect(linal_server)
    client.execute(f"DATASET {unique_name} COLUMNS (id: Int, emb: Vector(3)?)")
    client.execute(f"INSERT INTO {unique_name} VALUES (1, [1.0, 2.0, 3.0])")
    client.execute(f"INSERT INTO {unique_name} VALUES (2, null)")
    client.execute(f"SAVE DATASET {unique_name}")

    table = client.dataset(unique_name).to_arrow()

    # The server fell back to JSON-string encoding for this column (a
    # real NULL is present -- see CHANGELOG v0.1.72) -- the client must
    # transparently unwrap it, never leak the raw `{"Vector": [...]}`
    # text or a bare "null" string to the caller.
    assert table.column_names == ["id", "emb"]
    assert table.column("emb").to_pylist() == [[1.0, 2.0, 3.0], None]


def test_to_arrow_native_matrix_column(linal_server, unique_name):
    client = linaldb.connect(linal_server)
    client.execute(f"DATASET {unique_name} COLUMNS (id: Int, m: Matrix(2, 2))")
    client.execute(f"INSERT INTO {unique_name} VALUES (1, [[1.0, 2.0], [3.0, 4.0]])")
    client.execute(f"SAVE DATASET {unique_name}")

    table = client.dataset(unique_name).to_arrow()

    assert table.column("m").to_pylist() == [[[1.0, 2.0], [3.0, 4.0]]]


def test_to_pandas(linal_server, unique_name):
    pd = pytest.importorskip("pandas")

    client = linaldb.connect(linal_server)
    client.execute(f"DATASET {unique_name} COLUMNS (id: Int, emb: Vector(2))")
    client.execute(f"INSERT INTO {unique_name} VALUES (1, [0.5, 0.5])")
    client.execute(f"SAVE DATASET {unique_name}")

    df = client.dataset(unique_name).to_pandas()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "emb"]
    assert list(df["emb"].iloc[0]) == [0.5, 0.5]


def test_schema_manifest_stats(linal_server, unique_name):
    client = linaldb.connect(linal_server)
    client.execute(f"DATASET {unique_name} COLUMNS (id: Int, emb: Vector(3))")
    client.execute(f"INSERT INTO {unique_name} VALUES (1, [1.0, 2.0, 3.0])")
    client.execute(f"SAVE DATASET {unique_name}")

    ds = client.dataset(unique_name)
    schema = ds.schema()
    manifest = ds.manifest()
    stats = ds.stats()

    col_by_name = {c["name"]: c for c in schema["columns"]}
    assert col_by_name["emb"]["value_type"] == {"Vector": 3}
    assert manifest["formats"]["parquet"] == "data.parquet"
    assert stats["row_count"] == 1


def test_dataset_export_honors_non_default_database(linal_server, unique_name):
    # Regression test: Dataset's HTTP calls didn't send X-Linal-Database
    # at all, so a Client configured for a non-default database silently
    # fell back to the server's default database for /delivery -- found
    # while building a real example against a non-default database
    # (checkpoint 5), not by checkpoint 2's tests, which only ever used
    # the default database. A same-named dataset in each database, with
    # different data, proves the header is actually being honored (not
    # just "it didn't 404").
    default_client = linaldb.connect(linal_server)
    db_name = f"db_{unique_name}"
    default_client.execute(f"CREATE DATABASE {db_name}")

    default_client.execute(f"DATASET {unique_name} COLUMNS (id: Int, emb: Vector(2))")
    default_client.execute(f"INSERT INTO {unique_name} VALUES (1, [1.0, 1.0])")
    default_client.execute(f"SAVE DATASET {unique_name}")

    scoped_client = linaldb.connect(linal_server, database=db_name)
    scoped_client.execute(f"DATASET {unique_name} COLUMNS (id: Int, emb: Vector(2))")
    scoped_client.execute(f"INSERT INTO {unique_name} VALUES (1, [9.0, 9.0])")
    scoped_client.execute(f"SAVE DATASET {unique_name}")

    default_table = default_client.dataset(unique_name).to_arrow()
    scoped_table = scoped_client.dataset(unique_name).to_arrow()

    assert default_table.column("emb").to_pylist() == [[1.0, 1.0]]
    assert scoped_table.column("emb").to_pylist() == [[9.0, 9.0]]
