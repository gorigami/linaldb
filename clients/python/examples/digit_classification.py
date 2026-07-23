#!/usr/bin/env python3
"""Real end-to-end example (PYTHON_R_INTEROP_PLAN.md checkpoint 5):
starts a real `linal serve`, replays the real UCI handwritten-digits
classification workflow from ../../../examples/hdf5_digit_classification.lnl
through this client's `/execute`, exports the resulting datasets through
`/delivery`, and independently recomputes the classification in plain
Python/numpy from the exported raw vectors -- confirming the numbers
`/execute`'s SQL engine reports match the numbers you get from the raw
data `/delivery` serves, not just "did it run".

Reuses the real data already checked into this repo (real UCI Optical
Recognition of Handwritten Digits samples, see the .lnl file's own header
comment) by replaying that file's real DSL statements through the
client, rather than duplicating ~40 real 64-dimension vectors into this
script as literal data.

Usage:
    python digit_classification.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import linaldb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
LNL_SCRIPT = REPO_ROOT / "examples" / "hdf5_digit_classification.lnl"
DATABASE = "hdf5_digit_classification"


def find_linal_binary() -> Path:
    for profile in ("release", "debug"):
        candidate = REPO_ROOT / "target" / profile / "linal"
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"No `linal` binary found under {REPO_ROOT}/target/{{debug,release}}/. "
        "Run `cargo build --bin linal` in the repo root first."
    )


def start_server(port: int) -> subprocess.Popen:
    binary = find_linal_binary()
    # cwd = repo root so the .lnl script's `examples/data/...` HDF5 path
    # and the DROP/CREATE DATABASE statements resolve exactly like they
    # would running `linal run` directly from the repo root.
    proc = subprocess.Popen(
        [str(binary), "serve", "--port", str(port)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"linal serve exited early: {proc.stderr.read()}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.1)
    proc.kill()
    raise SystemExit("linal serve never became healthy")


def replay_lnl_file(client: linaldb.Client, path: Path) -> None:
    """Execute each real DSL statement in a `.lnl` file through the
    client. Most statements in this repo's example scripts are one
    physical line (examples/README.md item 2's convention), but not
    all -- e.g. this file's `DATASET ... COLUMNS (...)` blocks genuinely
    span several lines. Mirrors `linal run`'s own multi-line joiner
    (src/main.rs): accumulate lines, track paren balance, execute once
    balance returns to zero.
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
            client.execute(current)
            current = ""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    port = 18400
    print(f"Starting linal serve on port {port} (cwd={REPO_ROOT})...")
    proc = start_server(port)
    try:
        url = f"http://127.0.0.1:{port}"
        replay_client = linaldb.connect(url)

        print(f"\nReplaying real DSL from {LNL_SCRIPT.relative_to(REPO_ROOT)}:")
        replay_lnl_file(replay_client, LNL_SCRIPT)

        # Independent of the replay client's active-database state (the
        # .lnl file's own last line does `USE default`) -- scoped
        # explicitly via the database= parameter fixed for exactly this
        # checkpoint, see the "Dataset export didn't honor a non-default
        # database" fix.
        export_client = linaldb.connect(url, database=DATABASE)

        print(f"\nQuerying the real classification result via /execute (database={DATABASE!r})...")
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
        sql_result = export_client.execute(classify_sql)
        sql_rows = {row[0]: row for row in sql_result.rows}  # digit_id -> (id, true, pred, sim)
        print(f"  /execute returned {len(sql_rows)} classified rows")

        print("\nExporting query_digits and reference_centroids via /delivery...")
        query_df = export_client.dataset("query_digits").to_pandas()
        centroids_df = export_client.dataset("reference_centroids").to_pandas()
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
                    f"{digit_id}: numpy similarity {best_sim:.6f} vs /execute's {sql_similarity:.6f}"
                )
            if best_label != sql_predicted:
                mismatches.append(
                    f"{digit_id}: numpy predicted {best_label} vs /execute's {sql_predicted}"
                )
            if best_label == true_label:
                correct += 1

        total = len(sql_rows)
        print(f"\nIndependently-recomputed accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

        sql_correct = sum(1 for row in sql_rows.values() if row[1] == row[2])
        print(f"/execute-reported accuracy:        {sql_correct}/{total} ({100 * sql_correct / total:.1f}%)")

        if mismatches:
            print(f"\nFAIL: {len(mismatches)} mismatch(es) between /execute and /delivery-derived numbers:")
            for m in mismatches:
                print(f"  - {m}")
            sys.exit(1)
        elif correct != sql_correct:
            print("\nFAIL: aggregate accuracy differs between the two independently-computed paths.")
            sys.exit(1)
        else:
            print(
                "\nPASS: every per-row similarity, every predicted label, and the aggregate "
                "accuracy computed from /delivery's raw exported vectors exactly match what "
                "/execute's SQL engine reported."
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
