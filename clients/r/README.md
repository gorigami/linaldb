# linaldb (R client)

R client for [LINALDB](../../README.md) — a SQL-inspired analytical
engine treating vectors, matrices, and tensors as first-class citizens.

**Status**: functional through checkpoint 5 of
[`PYTHON_R_INTEROP_PLAN.md`](../../PYTHON_R_INTEROP_PLAN.md) — `/execute`
and `/delivery` are both implemented and tested against a real server
(55 passing `testthat` tests, including a full `R CMD check` pass, plus a
real end-to-end example, see below). Not yet published to CRAN — the
non-standard license string in `DESCRIPTION` needs resolving first, see
checkpoint 0's notes there.

Talks to a running `linal serve` instance over its HTTP API
(`POST /execute` for ad-hoc DSL, `/delivery/*` for real Parquet dataset
export) — see [`../CONTRACT.md`](../CONTRACT.md) for the exact wire
contract this client implements against. Requires `linal serve` running
an engine version `>= 0.1.74` (the version that fixed `USE <database>`
sent to `/execute` to actually persist, and fixed `/delivery` to honor a
non-default database).

## Usage

```r
library(linaldb)

conn <- linal_connect("http://localhost:8080")
df <- linal_query(conn, "SELECT id, embedding FROM docs WHERE score > 0.8")

ds <- linal_dataset(conn, "my_dataset")
df <- linal_dataset_read(ds)        # data.frame, via the arrow package
tbl <- linal_dataset_to_arrow(ds)   # arrow::Table
```

See [`examples/digit_classification.R`](examples/digit_classification.R)
for a complete real-data walkthrough: it starts a real `linal serve`,
replays a real UCI handwritten-digits classification workflow through
this client, queries the result via `/execute`, exports the same data via
`/delivery`, and independently recomputes the classification from the raw
exported vectors to confirm both paths agree exactly.

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
