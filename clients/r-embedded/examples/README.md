# Examples

Real end-to-end example scripts, the embedded/native counterpart of
[`../../r/examples/`](../../r/examples/README.md) — a real dataset (the
UCI handwritten-digits fixture already checked into `examples/data/`, no
synthetic data), replayed in-process through `linaldb.embedded` (no
`linal serve`, no HTTP), with results cross-checked against an
independent recomputation from the raw persisted vectors.

- [`digit_classification_embedded.R`](digit_classification_embedded.R) —
  opens an embedded engine via `linal_embedded_db()`, replays
  `examples/hdf5_digit_classification.lnl` directly against it, queries
  the classification result via `linal_embedded_execute()`, reads the
  saved `query_digits`/`reference_centroids` datasets straight off disk
  via `linal_embedded_dataset_read()`, and independently recomputes the
  classification in base R to confirm both paths agree exactly.

Run from the repo root's `clients/r-embedded/` after installing the
package (see [`../README.md`](../README.md)):

```sh
Rscript examples/digit_classification_embedded.R
```
