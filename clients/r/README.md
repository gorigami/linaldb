# linaldb (R client)

R client for [LINALDB](../../README.md) — a SQL-inspired analytical
engine treating vectors, matrices, and tensors as first-class citizens.

**Status**: functional through checkpoint 4 of
[`PYTHON_R_INTEROP_PLAN.md`](../../PYTHON_R_INTEROP_PLAN.md) — `/execute`
and `/delivery` are both implemented and tested against a real server
(53 passing `testthat` tests, including a full `R CMD check` pass). Not
yet published to CRAN — the non-standard license string in `DESCRIPTION`
needs resolving first, see checkpoint 0's notes there.

Talks to a running `linal serve` instance over its HTTP API
(`POST /execute` for ad-hoc DSL, `/delivery/*` for real Parquet dataset
export) — see [`../CONTRACT.md`](../CONTRACT.md) for the exact wire
contract this client implements against. Requires `linal serve` running
an engine version `>= 0.1.73` (the version that fixed `/delivery` to
actually resolve a dataset saved through the real multi-database
`SAVE DATASET` path, and made `schema.json` correctly report a
fallback-encoded Vector/Matrix column's logical type).

## Usage

```r
library(linaldb)

conn <- linal_connect("http://localhost:8080")
df <- linal_query(conn, "SELECT id, embedding FROM docs WHERE score > 0.8")

ds <- linal_dataset(conn, "my_dataset")
df <- linal_dataset_read(ds)        # data.frame, via the arrow package
tbl <- linal_dataset_to_arrow(ds)   # arrow::Table
```

## Development

This environment doesn't have `devtools` installed (only its lighter
dependencies `pkgload`/`roxygen2`/`testthat`), so local development here
uses:

```r
pkgload::load_all(".")
testthat::test_dir("tests/testthat")
```

If `devtools` is available, the usual `devtools::load_all()` /
`devtools::test()` work the same way. Before committing a change to any
`@export`ed function or its roxygen docs, regenerate `NAMESPACE`/`man/`:

```r
roxygen2::roxygenise(".")
```

To validate the way CRAN would (catches real issues `load_all()`-only
testing can miss — see checkpoint 3's findings in
`PYTHON_R_INTEROP_PLAN.md`):

```sh
R CMD build --no-build-vignettes .
R CMD check --no-manual --no-vignettes --no-build-vignettes linaldb_0.1.0.tar.gz
```
