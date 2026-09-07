# Examples

Real end-to-end examples, embedded-mode counterparts of
[`clients/python/examples`](../../python/examples): no `linal serve`
subprocess, no HTTP — an in-process `linaldb_embedded.Db()` replaying real
DSL, running a real query, and cross-checking the result against an
independent numpy recomputation from the same data read directly off
disk.

- [`digit_classification_embedded.py`](digit_classification_embedded.py)
  — script form.
- [`digit_classification_embedded.ipynb`](digit_classification_embedded.ipynb)
  — the same workflow as a Jupyter notebook, with narration.

Both replay the real UCI handwritten-digits data from
[`../../../examples/hdf5_digit_classification.lnl`](../../../examples/hdf5_digit_classification.lnl)
rather than duplicating the data — see that file's own header comment for
provenance.

Run either from the repo root or from here; both resolve the repo root
themselves (the script via a fixed relative path, the notebook by
searching upward for the `.lnl` file it replays, since a notebook has no
`__file__`).
