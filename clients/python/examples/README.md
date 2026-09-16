# Examples

Real end-to-end example scripts against a real, running `linal serve`.

- **`digit_classification.py`** — starts a real `linal serve` subprocess,
  replays the real UCI handwritten-digits classification workflow (the
  same one `../../../examples/hdf5_digit_classification.lnl` defines)
  through this client's `/execute`, exports the resulting datasets
  through `/delivery`, and independently recomputes the classification in
  plain Python/numpy from the raw exported vectors to confirm both paths
  agree exactly. Requires `cargo build --bin linal` to have been run in
  the repo root first (skipped, not failed, if the binary is missing).

Run it with `python examples/digit_classification.py` from
`clients/python/` (after `pip install -e ".[dev]"`).
