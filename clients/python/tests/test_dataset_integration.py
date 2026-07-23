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
