#!/usr/bin/env python3
"""Peak-RSS and wall-time measurement for `linaldb.Db.load_arrow` on a
synthetic MS/MS peak table.

Part of the CASMI 2026 load-memory work (P16: stream the content hash, P17:
move values into rows instead of cloning them). It builds an Arrow table of
`--spectra` rows, each with `--peaks-per` m/z / intensity pairs, then loads it
through the Python binding's `Db.load_arrow(..., peaks=...)` path -- the exact
entry point that OOM-killed the 32 GB Kaggle machine (`IMPROVEMENTS_PLAN.md`,
PR 2 and PR 4). It reports the process peak RSS
(`resource.getrusage(RUSAGE_SELF).ru_maxrss`) and the wall time of the load.

Run it twice, in fresh processes, to separate the Arrow table's own footprint
from the load's transient memory, as the original measurement did:

    python scripts/measure_load_memory.py --spectra 2500000 --peaks-per 30
    python scripts/measure_load_memory.py --spectra 2500000 --peaks-per 30 --no-load

The load's extra peak memory is the difference between the two peak-RSS
numbers. `--no-load` builds the same table and stops.

The table is deterministic (`--seed`), so the same command on the same machine
is comparable before and after a code change. This is a performance harness,
not a unit test: it asserts nothing and lives under `scripts/`.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import tempfile
import time


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    `ru_maxrss` is bytes on macOS and KiB on Linux.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def current_rss_bytes() -> int:
    """Current resident set size, in bytes, read from the OS if possible."""
    try:
        pages = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm") as f:  # Linux
            return int(f.read().split()[1]) * pages
    except OSError:
        pass
    # macOS: no cheap current-RSS read without psutil; ru_maxrss is the peak.
    return peak_rss_bytes()


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def build_table(spectra: int, peaks_per: int, seed: int):
    """An Arrow table of `spectra` rows, two `list<float32>` columns of
    `peaks_per` values each (`ms2_mzs` ascending, `ms2_intensities` in
    [0, 1]), and an `int64` id. Built from flat buffers + offsets, so the
    2.5M-row case does not go through 2.5M Python lists."""
    import numpy as np
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    mz = np.sort(
        rng.uniform(50.0, 2000.0, size=(spectra, peaks_per)), axis=1
    ).astype(np.float32)
    intensity = rng.uniform(0.05, 1.0, size=(spectra, peaks_per)).astype(np.float32)
    # Row-contiguous pairs, so each row is one slice of the flat buffer.
    flat_mz = np.ascontiguousarray(mz).reshape(-1)
    flat_intensity = np.ascontiguousarray(intensity).reshape(-1)
    offsets = np.arange(0, spectra * peaks_per + 1, peaks_per, dtype=np.int32)
    list_type = pa.list_(pa.float32())
    mz_col = pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat_mz), type=list_type)
    int_col = pa.ListArray.from_arrays(
        pa.array(offsets), pa.array(flat_intensity), type=list_type
    )
    ids = pa.array(np.arange(spectra, dtype=np.int64))
    return pa.table({"spectrum_id": ids, "ms2_mzs": mz_col, "ms2_intensities": int_col})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra", type=int, default=2_500_000)
    parser.add_argument("--peaks-per", type=int, default=30)
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="build the Arrow table, then stop (baseline footprint)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="run SHOW MEMORY after the load",
    )
    args = parser.parse_args()

    import pyarrow as pa  # noqa: F401  (fail early if pyarrow is missing)

    print(
        f"linaldb {_linaldb_version()} | pyarrow {pa.__version__} | "
        f"spectra={args.spectra:,} peaks/row={args.peaks_per} "
        f"({args.spectra * args.peaks_per:,} peaks)",
        flush=True,
    )

    t_build0 = time.perf_counter()
    table = build_table(args.spectra, args.peaks_per, args.seed)
    t_build1 = time.perf_counter()
    input_bytes = table.nbytes
    rss_after_build = peak_rss_bytes()
    print(
        f"built Arrow table in {t_build1 - t_build0:.2f} s "
        f"({human(input_bytes)} of buffers), peak RSS {human(rss_after_build)}",
        flush=True,
    )

    if args.no_load:
        print(f"RESULT mode=no-load peak_rss={peak_rss_bytes()}", flush=True)
        return 0

    import linaldb

    data_dir = tempfile.mkdtemp(prefix="linaldb-measure-")
    db = linaldb.Db(data_dir)

    t0 = time.perf_counter()
    n = db.load_arrow(
        "spectra",
        table,
        peaks={"peaks": ("ms2_mzs", "ms2_intensities")},
    )
    t1 = time.perf_counter()
    peak = peak_rss_bytes()
    print(
        f"Db.load_arrow loaded {n:,} rows in {t1 - t0:.2f} s; "
        f"peak RSS {human(peak)}",
        flush=True,
    )
    if args.show_memory:
        print(db.query("SHOW MEMORY"), flush=True)
    print(
        f"RESULT mode=load spectra={args.spectra} peaks_per={args.peaks_per} "
        f"rows={n} wall_s={t1 - t0:.3f} peak_rss_bytes={peak} "
        f"rss_after_build_bytes={rss_after_build} "
        f"extra_peak_bytes={peak - rss_after_build} input_buffers_bytes={input_bytes}",
        flush=True,
    )
    return 0


def _linaldb_version() -> str:
    try:
        import linaldb

        return getattr(linaldb, "__version__", "?")
    except Exception:  # noqa: BLE001  (--no-load never imports linaldb)
        return "not-imported"


if __name__ == "__main__":
    raise SystemExit(main())
