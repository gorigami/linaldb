# linaldb (R client)

R client for [LINALDB](../../README.md) — a SQL-inspired analytical
engine treating vectors, matrices, and tensors as first-class citizens.

**Status**: scaffolding only (checkpoint 0 of
[`PYTHON_R_INTEROP_PLAN.md`](../../PYTHON_R_INTEROP_PLAN.md)) — not
functional yet, not published to CRAN.

Talks to a running `linal serve` instance over its HTTP API
(`POST /execute` for ad-hoc DSL, `/delivery/*` for real Parquet dataset
export) — see [`../CONTRACT.md`](../CONTRACT.md) for the exact wire
contract this client implements against. Requires `linal serve` running
an engine version `>= 0.1.72` (the version that fixed Vector/Matrix
columns to encode as native Arrow types in delivered Parquet rather than
JSON strings).

## Planned usage (not yet implemented)

```r
library(linaldb)

conn <- linal_connect("http://localhost:8080")
df <- linal_query(conn, "SELECT id, embedding FROM docs WHERE score > 0.8")

ds <- linal_dataset(conn, "my_dataset")
df <- linal_dataset_read(ds)   # returns a data.frame / tibble via the arrow package
```

## Development

```r
devtools::load_all()
devtools::test()
```
