#!/usr/bin/env python3
"""Checks an *installed* `linaldb` wheel, the way a user (or a Kaggle kernel)
gets it: run from any directory after `pip install linaldb==X`, with no repo
checkout and no network.

    python scripts/verify_wheel.py [--version X]

It loads a fixed table through the Arrow and DSL paths (scalars, a vector
column, MS/MS peak lists), runs a query and a similarity search, and
compares the loaded datasets' content hashes with golden values. A different hash means the
engine inside the wheel computes provenance differently from the release
the goldens were taken from -- lineage files and `ASSERT LINEAGE` checks
written against older releases would stop matching.

Release-pipeline step (`IMPROVEMENTS_PLAN.md`, PR 3); asserts, exits 1 on any
mismatch. Recapture the goldens only for a deliberate hash change, and say so
in the CHANGELOG.
"""

from __future__ import annotations

import argparse
import sys
import tempfile

# Captured from linaldb 0.1.19 (engine v0.1.94). Only datasets built from
# input values, not computed floats: a search score can differ in the last
# bit between SIMD paths (NEON vs AVX), so `hits` is checked by its ids.
GOLDEN = {
    "spectra": "b840a674719bc9378593b6aa3835c4a9ae093cb7303a18d40e9b2b5ccf79b3ef",
    "items": "580aa9123187c75b1bbd5aca25edd9a3f4d81ec7ce87a307f0115e3e04d23915",
}


def content_hash(db, name: str) -> str:
    return db.lineage(name)["entity"]["Dataset"]["content_hash"]


def build(db) -> None:
    import pyarrow as pa

    vec = pa.list_(pa.float32(), 4)
    spectra = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "name": pa.array(["caffeine", "aspirin", "glucose"]),
            "mass": pa.array([194.0804, 180.0423, 180.0634], pa.float64()),
            "e": pa.array(
                [[1.0, 0.0, 0.5, 0.25], [0.0, 1.0, 0.5, 0.75], [0.5, 0.5, 1.0, 0.0]],
                vec,
            ),
            "mz": pa.array(
                [[69.07, 110.07, 138.07, 195.09], [93.03, 121.03, 163.04], [85.03, 163.06]],
                pa.list_(pa.float32()),
            ),
            "intensity": pa.array(
                [[0.1, 0.4, 1.0, 0.6], [0.3, 1.0, 0.2], [1.0, 0.5]],
                pa.list_(pa.float32()),
            ),
        }
    )
    assert db.load_arrow("spectra", spectra, peaks={"peaks": ("mz", "intensity")}) == 3

    for line in [
        "DATASET items COLUMNS (id: Int, tag: String, score: Float64, v: Vector(3))",
        'INSERT INTO items VALUES (1, "a", 0.5, [1.0, 2.0, 3.0])',
        'INSERT INTO items VALUES (2, "b", 1.25, [0.0, 1.0, 0.0])',
        'INSERT INTO items VALUES (3, "a", -2.0, [3.0, 2.0, 1.0])',
        "CREATE VECTOR INDEX ON spectra(e)",
        "SEARCH spectra ON e QUERY [1.0, 0.0, 0.5, 0.25] LIMIT 2 INTO hits",
    ]:
        db.execute(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", help="fail unless linaldb.__version__ is this")
    parser.add_argument("--print", action="store_true", help="print hashes (to capture goldens)")
    args = parser.parse_args()

    import linaldb

    print(f"linaldb {linaldb.__version__} from {linaldb.__file__}")
    failed = False
    if args.version and linaldb.__version__ != args.version:
        print(f"FAIL version: expected {args.version}, got {linaldb.__version__}")
        failed = True

    with tempfile.TemporaryDirectory() as data_dir:
        db = linaldb.Db(data_dir)
        build(db)
        if db.execute("SELECT id FROM items WHERE tag <> \"a\"").rows != [[2]]:
            print("FAIL query: SELECT ... WHERE tag <> 'a'")
            failed = True
        hits = db.execute("SELECT id FROM hits").rows
        if sorted(hits) != [[1], [3]]:
            print(f"FAIL search: expected ids 1 and 3, got {hits}")
            failed = True
        for name, expected in GOLDEN.items():
            got = content_hash(db, name)
            if args.print:
                print(f'    "{name}": "{got}",')
            elif got != expected:
                print(f"FAIL {name}: content hash {got}, golden {expected}")
                failed = True

    if not args.print:
        print("FAILED" if failed else "OK: version, query and content hashes match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
