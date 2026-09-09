#!/usr/bin/env python3
"""Real end-to-end example, embedded-mode counterpart of
`clients/python/examples/digit_classification.py`: replays the real UCI
handwritten-digits classification workflow from
`../../../examples/hdf5_digit_classification.lnl` through an in-process
`linaldb.Db()` -- no `linal serve` subprocess, no HTTP at all --
then independently recomputes the classification in plain Python/numpy
from the same dataset, read directly off disk via `Db.dataset(...)`, and
confirms the numbers match exactly. Same rigor as the HTTP client's
example, minus the server.

Reuses the real data already checked into the repo (real UCI Optical
Recognition of Handwritten Digits samples, see the .lnl file's own header
comment) by replaying that file's real DSL statements through the
embedded engine, rather than duplicating ~40 real 64-dimension vectors
into this script as literal data.

Usage:
    python digit_classification_embedded.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import linaldb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
LNL_SCRIPT = REPO_ROOT / "examples" / "hdf5_digit_classification.lnl"
DATABASE = "hdf5_digit_classification"


def replay_lnl_file(db: "linaldb.Db", path: Path) -> None:
    """Execute each real DSL statement in a `.lnl` file in-process.
    Mirrors `linal run`'s own multi-line joiner (`src/main.rs`) and the
    HTTP example's `replay_lnl_file`: accumulate lines, track paren
    balance, execute once balance returns to zero.
    """
    current = ""
    paren_balance = 0
    start_lineno = None
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not current:
            if not line or line.startswith("--"):
                continue
            start_lineno = lineno
        current = f"{current} {line}".strip() if current else line
        paren_balance += line.count("(") - line.count(")")
        if paren_balance == 0:
            print(f"  [{start_lineno}] {current[:80]}{'...' if len(current) > 80 else ''}")
            db.execute(current)
            current = ""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    print(f"Opening an embedded LINALDB instance (cwd for relative paths = {REPO_ROOT})...")
    os.chdir(REPO_ROOT)
    db = linaldb.Db()

    print(f"\nReplaying real DSL from {LNL_SCRIPT.relative_to(REPO_ROOT)} (in-process, no server):")
    replay_lnl_file(db, LNL_SCRIPT)

    # The .lnl file's own last line does `USE default` -- switch back to
    # the database it populated before querying/exporting from it,
    # embedded-mode's equivalent of the HTTP example's per-request
    # `X-Linal-Database` header scoping.
    db.execute(f"USE {DATABASE}")

    print(f"\nRunning the real classification query in-process (database={DATABASE!r})...")
    classify_sql = (
        "WITH classified AS ("
        "SELECT query_digits.digit_id AS digit_id, query_digits.true_label AS true_label, "
        "reference_centroids.digit_class AS predicted_label, "
        "COSINE_SIM(query_digits.pixels, reference_centroids.centroid) AS similarity, "
        "ROW_NUMBER() OVER (PARTITION BY digit_id ORDER BY similarity DESC) AS rn "
        "FROM query_digits JOIN reference_centroids "
        "ON COSINE_SIM(query_digits.pixels, reference_centroids.centroid) > 0.5"
        ") SELECT digit_id, true_label, predicted_label, similarity "
        "FROM classified WHERE rn = 1 ORDER BY digit_id"
    )
    sql_result = db.execute(classify_sql)
    sql_rows = {row[0]: row for row in sql_result.rows}  # digit_id -> (id, true, pred, sim)
    print(f"  in-process query returned {len(sql_rows)} classified rows")

    print("\nExporting query_digits and reference_centroids directly from disk (no /delivery, no HTTP)...")
    query_df = db.dataset("query_digits").to_pandas()
    centroids_df = db.dataset("reference_centroids").to_pandas()
    print(f"  query_digits: {len(query_df)} rows, reference_centroids: {len(centroids_df)} rows")

    print("\nIndependently recomputing classification in pure numpy from the exported raw vectors...")
    centroid_vecs = {
        row["digit_class"]: np.array(row["centroid"], dtype=np.float64)
        for _, row in centroids_df.iterrows()
    }

    mismatches = []
    correct = 0
    for _, row in query_df.iterrows():
        digit_id = row["digit_id"]
        if digit_id not in sql_rows:
            continue  # SQL's similarity > 0.5 threshold excluded this one entirely
        query_vec = np.array(row["pixels"], dtype=np.float64)

        sims = {label: cosine_similarity(query_vec, c) for label, c in centroid_vecs.items()}
        best_label = max(sims, key=sims.get)
        best_sim = sims[best_label]

        _, true_label, sql_predicted, sql_similarity = sql_rows[digit_id]
        if abs(best_sim - sql_similarity) > 1e-4:
            mismatches.append(
                f"{digit_id}: numpy similarity {best_sim:.6f} vs in-process query's {sql_similarity:.6f}"
            )
        if best_label != sql_predicted:
            mismatches.append(
                f"{digit_id}: numpy predicted {best_label} vs in-process query's {sql_predicted}"
            )
        if best_label == true_label:
            correct += 1

    total = len(sql_rows)
    print(f"\nIndependently-recomputed accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

    sql_correct = sum(1 for row in sql_rows.values() if row[1] == row[2])
    print(f"In-process-query-reported accuracy: {sql_correct}/{total} ({100 * sql_correct / total:.1f}%)")

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} mismatch(es) between the in-process query and the dataset export:")
        for m in mismatches:
            print(f"  - {m}")
        sys.exit(1)
    elif correct != sql_correct:
        print("\nFAIL: aggregate accuracy differs between the two independently-computed paths.")
        sys.exit(1)
    else:
        print(
            "\nPASS: every per-row similarity, every predicted label, and the aggregate "
            "accuracy computed from the raw exported vectors exactly match what the "
            "in-process DSL query reported -- entirely within one Python process, no server."
        )


if __name__ == "__main__":
    main()
