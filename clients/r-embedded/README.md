# linaldb.embedded (R embedded/native binding)

Embedded/native R binding for [LINALDB](../../README.md) — a SQL-inspired
analytical engine treating vectors, matrices, and tensors as first-class
citizens. Wraps the engine's Rust `TensorDb` directly via
[`extendr`](https://extendr.github.io/) — **no HTTP, no `linal serve`
process**. Contrast with [`clients/r`](../r/README.md) (package
`linaldb`), a thin HTTP client for a running server — that's still the
right choice for talking to a remote/shared instance; this package is for
the "embedded like SQLite" use case, in-process, same address space as
your R session.

See [`../EMBEDDED_CONTRACT.md`](../EMBEDDED_CONTRACT.md) for the exact
Rust/R result-shape contract this package implements against, and
[`../python-embedded/README.md`](../python-embedded/README.md) for the
Python counterpart (same design, `pyo3` instead of `extendr`).

Not yet published to CRAN — the non-standard license string in
`DESCRIPTION` needs resolving first (same caveat as `clients/r`), and
this hasn't gone through a real `R CMD check` pass yet either.

## Usage

```r
library(linaldb.embedded)

db <- linal_embedded_db()  # in-memory engine; data_dir defaults to "./data"
linal_embedded_execute(db, "DATASET docs COLUMNS (id: Int, embedding: Vector(3))")
linal_embedded_execute(db, "INSERT INTO docs VALUES (1, [0.9, 0.1, 0.0])")

df <- linal_embedded_query(db, "SELECT id, embedding FROM docs")

linal_embedded_execute(db, "SAVE DATASET docs")
ds <- linal_embedded_dataset(db, "docs")
df <- linal_embedded_dataset_read(ds)        # data.frame, read straight off disk
tbl <- linal_embedded_dataset_to_arrow(ds)   # arrow::Table
```

`linal_embedded_active_db(db)` / `linal_embedded_data_dir(db)` expose the
engine's active database name and configured data directory — useful for
locating a saved dataset's package directory yourself
(`{data_dir}/{active_db}/datasets/{name}/`) without going through the
`linal_embedded_dataset*()` helpers.

See
[`examples/digit_classification_embedded.R`](examples/digit_classification_embedded.R)
for a complete real-data walkthrough: it opens an embedded engine (no
server at all), replays a real UCI handwritten-digits classification
workflow through it, queries the result via `linal_embedded_execute()`,
reads the same data straight off disk via `linal_embedded_dataset_read()`,
and independently recomputes the classification from the raw persisted
vectors to confirm both paths agree exactly.

## Development

This package embeds a real Rust crate (`src/rust/`, extendr) that depends
on the repo-root `linal` crate via a `path` dependency — **not** part of
the repo-root Cargo workspace (there is none; this crate and
`clients/python-embedded`'s are both standalone Cargo projects, by
design, so building either never touches the root crate's protected CI
gates). Building it compiles the entire `linal` dependency graph,
including a from-source vendored HDF5 build (`hdf5-metno`'s `static`
feature) — expect the **first** install to take several minutes; `cargo
build`/`clippy`/`fmt` directly inside `src/rust/` are much faster once
`linal`'s own build is cached in that crate's `target/` dir. Note that
the package's own `Makevars` deletes `src/rust/target/` after a
successful `R CMD INSTALL` (standard CRAN-safe extendr packaging), so
each *package* install redoes the full build from scratch even though a
plain `cargo build` in `src/rust/` in between installs does not.

```r
install.packages(c("rextendr", "testthat", "arrow", "jsonlite"))

# After changing the Rust side (src/rust/src/lib.rs), regenerate the R
# wrapper functions this package's `#[extendr]` annotations produce:
setwd("clients/r-embedded/src")
system("cargo run --bin document --manifest-path=./rust/Cargo.toml")
setwd("../../..")

# Then regenerate NAMESPACE/man/ from roxygen comments and (re)install:
roxygen2::roxygenise("clients/r-embedded")
```

(`rextendr::document()` itself now shells out to `devtools::document()`,
which pulls in `usethis`/`pkgdown` and their heavier system dependencies
such as `libgit2`/`harfbuzz`/`freetype` — the two-step
`cargo run --bin document` + `roxygen2::roxygenise()` above does the same
work without that dependency chain, and is what was actually used to
build and verify this package.)

Run tests against the installed package:

```r
library(linaldb.embedded)
testthat::test_dir("clients/r-embedded/tests/testthat")
```
