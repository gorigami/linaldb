# LINAL DSL Reference

**LINAL Script** is a high-performance, SQL-inspired language for tensor algebra and relational analytics. This document serves as the complete technical specification for all keywords, operators, and built-in functions.

Line comments start with `--`, `#`, or `//` (all three are equivalent) and run to the end of the line. Blank lines are ignored.

---

## 1. Data Types & Literals

LINAL supports both standard relational types and multi-dimensional numeric structures.

### Relational Types

- `Int`: 64-bit signed integer.
- `Float` (aliases: `FLOAT32`): 32-bit floating point (standard for tensor values — `Vector`/`Matrix`/`Tensor` elements are always this precision).
- `Double` (aliases: `FLOAT64`): 64-bit floating point. Use this for real-world large-magnitude scalar values (GPS/Unix timestamps, etc.) that exceed `Float`'s ~7 significant digits — a plain `Float` column silently rounds these. Arithmetic mixing a `Double` with a `Float`/`Int` always promotes the result to `Double`. Not available for `Vector`/`Matrix`/`Tensor` elements, which remain `Float`-only.
- `String`: UTF-8 character sequence.
- `Bool`: `true` or `false`.
- `Complex`: Scalar complex number (`f64` real + imaginary parts, wrapping `num_complex::Complex64`). No dedicated literal syntax — construct one with `COMPLEX(re, im)` (§3). Fully usable as a column type, `SELECT`/`WHERE` expression, and `SUM`/`AVG` aggregate (both well-defined for complex numbers), but **has no ordering**: `MIN`/`MAX`, `ORDER BY`, and `<`/`>`/`<=`/`>=` all error loudly on a `Complex` operand rather than guessing one (`=`/`!=` work — equality is well-defined even without an order). Scalar-only — there is no `Vector`/`Matrix` of `Complex` (a genuine `Tensor<Complex>` type is a separate, larger initiative); a *collection* of complex numbers (e.g. `EIGENVALUES_GENERAL`'s output) is instead a `Matrix(2, N)` with real parts in row 0 and imaginary parts in row 1, the same convention `FFT` already uses for its spectrum.
- `Null`: Represents a missing value. Use the `?` suffix in `DATASET` definitions for nullable columns (e.g., `score: Float?`).

### Tensor Types

Defined with specific dimensionality:

- `Vector(N)`: A 1D tensor with `N` elements.
- `Matrix(R, C)`: A 2D tensor with `R` rows and `C` columns.
- `Tensor(d1, d2, ...)`: An N-dimensional tensor.

---

## 2. Resource Definition

Create and initialize numeric resources and structured schemas.

### VECTOR / MATRIX

Quick shorthand for defining tensors.

```sql
VECTOR v = [1.0, 2.0, 3.0]
MATRIX m = [[1, 2], [3, 4]]
```

### DEFINE

Explicit tensor definition for higher dimensions. Shape is specified as a bracket-delimited list.

```sql
DEFINE t AS TENSOR [2, 2, 2] VALUES [1, 2, 3, 4, 5, 6, 7, 8]
```

The optional `STRICT` modifier enforces shape-strictness: any binary operation involving a strict tensor propagates the strict flag to its output, preventing accidental shape relaxation.

```sql
DEFINE w AS STRICT TENSOR [3] VALUES [1, 0, 0]
```

### Tensor-First Dataset Constructor

Create a zero-copy tensor-first dataset using the `dataset()` constructor inside a `LET` assignment:

```sql
LET ds = dataset("my_dataset")
```

This registers an empty named dataset in the `DatasetRegistry`. Columns are added later via `.add_column()`.

### Adding Columns to a Tensor-First Dataset

After creating a tensor-first dataset you can attach any in-memory tensor as a column using dot-method syntax:

```sql
VECTOR v_temp = [36.6, 37.1, 36.9]
LET raw = dataset("raw")
raw.add_column(temp, v_temp)
```

Syntax: `<dataset_var>.add_column(<column_name>, <tensor_var>)`

This is an O(1) metadata operation — no data is copied.

### DATASET

Define a persistent relational structure.

```sql
DATASET diagnostics COLUMNS (
    id: Int,
    region: String,
    score: Float?,           -- Nullable column
    features: Vector(128)    -- Embedded tensor
)
```

### DATASET ... FROM (Materialized View)

A second `DATASET` form builds a new dataset from an existing one by running a
query and materializing the result under a new name — equivalent to `SELECT
... FROM <source> ... ` but persisted as a real dataset instead of returned
inline:

```sql
DATASET seniors FROM employees FILTER age >= 60

DATASET top_scores FROM diagnostics
    FILTER region = "west"
    SELECT region, AVG(score)
    GROUP BY region
    HAVING AVG(score) > 0.5
    ORDER BY region
    LIMIT 10
```

`DATASET <name> FROM <source> [FILTER|WHERE <expr>] [SELECT <cols>] [GROUP BY <cols>] [HAVING <expr>] [ORDER BY <cols>] [LIMIT <n>] [OFFSET <n>]` — all clauses after `FROM <source>` are optional and behave the same as their `SELECT` statement equivalents (§4).

---

## 3. Numerical DSL (Core Algebra)

LINAL provides two ways to perform math: Functional keywords and Infix operators.

### Functional Keywords

- `ADD a b`: Element-wise addition.
- `SUBTRACT a b`: Element-wise subtraction.
- `MULTIPLY a b`: Element-wise multiplication (Hadamard product).
- `DIVIDE a b`: Element-wise division.
- `MATMUL a b`: Standard matrix multiplication.
- `TRANSPOSE a`: Swap dimensions of a matrix/tensor.
- `RESHAPE a TO [dims]`: Change shape without copying data.
- `FLATTEN a`: Convert multidimensional tensor to a 1D vector.
- `NORMALIZE a`: Scales vector to unit length (L2 norm).
- `SCALE a BY n`: Multiplies all elements by a scalar `n`.
- `STACK t1 t2 ...`: Combines tensors along Axis 0.
- `SUM a`: Sum of all elements in the tensor. Result is a true scalar (rank-0 tensor, shape
  `[]`) — the same convention `CORRELATE`/`SIMILARITY`/`DISTANCE` (below) use — so it correctly
  broadcasts against a longer vector in a subsequent `ADD`/`SUBTRACT`/`MULTIPLY`/`DIVIDE`
  (e.g. `v - SUM v`) instead of being treated as a same-rank vector of differing length.
- `MEAN a`: Arithmetic mean of all elements. Result is a true scalar (rank-0), same as `SUM`.
- `STDEV a`: Standard deviation of all elements. Result is a true scalar (rank-0), same as `SUM`.
- `VARIANCE a`: Population variance of all elements (`STDEV a` squared — the two share one code
  path, so they're always numerically consistent). Result is a true scalar (rank-0), same as `SUM`.
  Supports `LAZY`, like `SUM`/`MEAN`/`STDEV`.
- `MEDIAN a`: Median of all elements, flattened and sorted (any rank). Averages the two middle
  values on an even element count. Result is a true scalar (rank-0).
- `QUANTILE a AT p`: The `p`-th quantile (`0.0..=1.0`) of all elements, flattened and sorted, via
  linear interpolation between the two nearest ranks (numpy's default `linear` method) — so
  `QUANTILE a AT 0.5` matches `MEDIAN a` exactly. Result is a true scalar (rank-0).
- `COVARIANCE a WITH b`: Population covariance between two same-shape tensors, treating
  corresponding (flattened, row-major) elements as paired samples. Result is a true scalar
  (rank-0).
- `COVARIANCE MATRIX a`: Feature covariance matrix of `a` (rows = samples, columns = features):
  mean-centers each column, then `(centeredᵗ * centered) / (rows - 1)` — the standard *sample*
  covariance matrix (Bessel's correction), matching numpy's `cov` default. **Note the different
  normalization from `COVARIANCE a WITH b` above** (`n` vs. `n - 1`) — two different statistics
  serving different purposes, not an inconsistency. `a` must have at least 2 rows. Result is a
  `Matrix(cols, cols)`.

```sql
VECTOR v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LET var = VARIANCE v          -- 8.25 (STDEV v squared)
LET med = MEDIAN v            -- 5.5
LET p90 = QUANTILE v AT 0.9   -- 9.1
VECTOR w = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
LET cov = COVARIANCE v WITH w -- -8.25 (perfectly anti-correlated)
MATRIX samples = [[1,2],[2,1],[3,4],[4,3],[5,6]]
LET cm = COVARIANCE MATRIX samples   -- Matrix(2, 2), sample covariance
```

### Complex Numbers

`Complex` (§1) is a scalar SQL/relational type, not a tensor-DSL keyword — these are SQL-callable functions (usable in `SELECT`/`WHERE`/computed columns), not standalone `LET`-bound operators.

- `COMPLEX(re, im)`: constructs a `Complex` scalar. The only way to write a complex value directly — there is no `3+4i`-style literal syntax.
- `REAL(z)` / `IMAG(z)`: real / imaginary part. Result: `Double`.
- `ABS(z)`: magnitude (`sqrt(re² + im²)`). Named `ABS`, not `MAGNITUDE`, to avoid any confusion with the unrelated `MAGNITUDE a` tensor-DSL keyword (§3, FFT spectrum magnitude — a different operator on a different type). Result: `Double`.
- `PHASE(z)`: phase angle (`atan2(im, re)`, radians). Result: `Double`.
- `CONJ(z)`: complex conjugate (`re - im·i`). Result: `Complex`.

Arithmetic (`+`, `-`, `*`, `/`) works between two `Complex` values, or a `Complex` and any real numeric type (`Int`/`Float`/`Double`), promoting the real operand to a zero-imaginary-part complex number first — the same "mixed arithmetic always promotes" convention `Double` itself uses. `SUM`/`AVG` aggregates (plain and windowed, `OVER (...)`) are well-defined and fully supported for a `Complex` column. `=`/`!=` compare by real equality. **`Complex` has no ordering** — `MIN`/`MAX`, `ORDER BY`, and `<`/`>`/`<=`/`>=` all error loudly rather than silently picking an arbitrary "winner" or leaving rows unsorted.

```sql
DATASET nums COLUMNS (id: Int, re: Double, im: Double)
INSERT INTO nums (id = 1, re = 3.0, im = 4.0)
INSERT INTO nums (id = 2, re = 1.0, im = -1.0)

SELECT id, COMPLEX(re, im) AS z FROM nums          -- z: 3+4i, 1-1i
SELECT REAL(COMPLEX(re, im)), ABS(COMPLEX(re, im)) FROM nums WHERE id = 1  -- 3, 5

TRANSFORM nums SELECT id, COMPLEX(re, im) AS val INTO cnums
SELECT SUM(val) AS s FROM cnums                    -- s: 4+3i
SELECT MIN(val) FROM cnums                         -- error: no ordering
```

### Lazy Evaluation

Prefix a `LET` with `LAZY` (either word order is accepted) to defer computation. The expression is stored as a computation graph and materialized only when `SHOW` is called.

```sql
LAZY LET trend = STDEV sensor_3d   -- deferred
LET LAZY trend = STDEV sensor_3d   -- identical alias
SHOW trend                         -- triggers materialization
```

### Infix Operators

Standard math notation for scalar and tensor variables:

```sql
LET result = (v_a + v_b) / 2.0
LET scaled = m_a * 10
```

### Advanced Operators

- `CORRELATE a WITH b`: Pearson correlation between two vectors.
- `SIMILARITY a WITH b`: Cosine similarity score [-1.0, 1.0].
- `DISTANCE a TO b`: Euclidean distance between points.

### Frequency-Domain Operators

- `FFT a [WINDOW HANN|HAMMING]`: Real-to-complex forward Fast Fourier Transform. `a` must be a rank-1 `Vector(N)`. Result is a `Matrix(2, N/2+1)` — **row 0 is the real part, row 1 is the imaginary part** of each frequency bin (only non-negative frequencies are computed, since a real input signal's spectrum is symmetric — this is the standard real-input FFT optimization, not a data loss). The optional `WINDOW` clause applies a Hann or Hamming window to `a` before transforming, to reduce spectral leakage — omitting it is the original unwindowed (rectangular) behavior. `HANN`/`HAMMING` are plain identifiers, not reserved keywords.
- `IFFT a`: Complex-to-real inverse FFT. `a` must be a `Matrix(2, M)` spectrum (as `FFT` produces). Result is a real `Vector`. **Assumes the original signal length was even** (reconstructs length `2*(M-1)`) — the spectrum alone can't distinguish an even- from an odd-length source signal (both produce the same `M`), and there is currently no side-channel carrying the true length through the DSL layer. If you need an odd-length round trip, keep the original vector around rather than relying on `IFFT` to recover its exact length.
- `MAGNITUDE a`: Power/magnitude spectrum. `a` must be a `Matrix(2, M)` spectrum (as `FFT` produces). Result is a real `Vector(M)`, `sqrt(re² + im²)` per bin. The convenience most whitening/PSD-estimation work actually needs without touching phase.
- `PSD a WINDOW n [HANN|HAMMING]`: Power spectral density (noise-floor) estimate via averaged periodograms. `a` must be a rank-1 `Vector` at least `n` samples long; splits it into non-overlapping `n`-sample chunks (any remainder that doesn't fill a full chunk is dropped), computes each chunk's power spectrum, and averages them elementwise. Result is a real `Vector(n/2+1)`. The optional trailing `HANN`/`HAMMING` applies that window function to each chunk before its FFT (omitting it is the original unwindowed behavior). **Still simplified vs. textbook Welch's method**: no 50% chunk overlap, even with a window function applied — good enough for noise-floor estimation, not a research-grade PSD estimator.
- `WHITEN a WITH b`: Flattens `a`'s noise spectrum against a PSD estimate `b` (as `PSD` produces): divides each bin of `FFT(a)` by `sqrt(b[bin])`, then inverse-transforms back to the time domain. `b` must have exactly `a.len()/2+1` entries — the same spectrum length `FFT(a)` itself would produce (in practice, estimate it with `PSD a WINDOW <a's own length>`, a single-chunk periodogram; resampling a PSD estimated at a different window size onto a longer signal is not implemented). Result is a real `Vector` the same length as `a`. The standard first step before matched filtering — pulling a real signal out of instrument noise needs the noise spectrum flattened first, or a loud broadband frequency band silently dominates over the signal you're looking for.
- `BANDPASS a FROM low_hz TO high_hz WITH RATE sample_rate`: Brick-wall bandpass filter — zeros every FFT bin whose frequency (`bin_index * sample_rate / a.len()`) falls outside `[low_hz, high_hz]`, then inverse-transforms back to the time domain. `a` must be a rank-1 `Vector`. Result is a real `Vector` the same length as `a`. **Simplified vs. a real filter design** (IIR/FIR with a proper transition band, e.g. Butterworth/Chebyshev): a hard bin cutoff introduces ringing (Gibbs phenomenon) at sharp edges, unlike a designed filter's smooth rolloff.
- `MATCHED_FILTER a WITH b`: FFT-based cross-correlation — `IFFT(FFT(a) * conj(FFT(b)))` — the standard real-world signal-detection statistic: the peak of the result (by absolute value) marks the best-matching lag between `a` (the data being searched) and `b` (the template being searched for). `a`/`b` must be rank-1 `Vector`s of the same length; result is a real `Vector` that length. **The peak lag is relative to `b`'s own reference point, not an absolute location in `a`** — if the feature `b` is modeling sits at index `c` within `b`'s own buffer, and `a`'s copy of that feature is truly at index `s`, the correlation peaks at `s - c`, not at `s` directly; recover the true location as `peak_lag + c`. Also computes **circular correlation, not linear correlation** (the FFT-multiply trick wraps around at the buffer edges) — fine for a peak safely inside the buffer, not for a match expected right at the boundary.

```sql
VECTOR signal = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
LET spectrum = FFT signal        -- Matrix(2, 5): real row, imaginary row
LET recovered = IFFT spectrum    -- back to the original 8-sample Vector
LET mag = MAGNITUDE spectrum     -- Vector(5): power spectrum
LET noise_floor = PSD signal WINDOW 8   -- Vector(5): matches signal's own FFT length
LET whitened = WHITEN signal WITH noise_floor   -- Vector(8): flattened spectrum
LET filtered = BANDPASS signal FROM 35.0 TO 350.0 WITH RATE 4096.0  -- keep only 35-350 Hz
LET correlation = MATCHED_FILTER whitened WITH template  -- Vector(8): correlation-vs-lag
```

A complex *spectrum* (many complex numbers) stays an ordinary `Matrix(2,
N)` value by convention, even though the scalar `Complex` type (§1) now
exists — so every existing `Matrix`-handling feature (`SHOW`, persistence,
`TRANSPOSE`, row indexing) already works on it unmodified, and `Complex`
stays genuinely scalar-only (a locked design decision, not an oversight —
see `SCIENTIFIC_ENGINE_EXPANSION_PLAN.md` Phase 3). `EIGENVALUES_GENERAL`
below uses the identical `Matrix(2, N)` convention for the same reason.
See `SIGNAL_PROCESSING_PLAN.md` at the repo root for `FFT`'s own full
design history.

### Classical Linear Algebra

Built on `nalgebra` (pure Rust, matching the `realfft` precedent above over
binding to a system LAPACK/BLAS). Every fallible operator here errors
loudly on a singular/non-square/non-symmetric input — never a silent `NaN`
in the result. See `LINEAGE_AND_LINALG_PLAN.md` Phase 8 for the full design
history.

**Single-output** (bind with a plain `LET <name> = ...`):

- `TRACE a`: Sum of the diagonal. `a` must be a square `Matrix`. Result is a true scalar (rank-0), same convention as `SUM`/`MEAN`/`STDEV` above.
- `DETERMINANT a`: `a` must be a square `Matrix`. Result is a true scalar. `0.0` for a singular matrix is a legitimate result, not an error.
- `RANK a`: Numerical rank via singular value decomposition. Result is a true scalar, integer-valued (e.g. `2.0`) — no new scalar `Value` variant for an integer result.
- `INVERSE a`: `a` must be a square `Matrix`. Result is a `Matrix` the same shape. **Errors if `a` is singular** — never returns a matrix full of `NaN`/`Inf`.
- `SOLVE a b`: Solves `a x = b` for `x` via LU decomposition with partial pivoting. `a` must be square; `b` a `Vector` with length matching `a`'s row count. Result is a `Vector`. **Errors if `a` is singular.**
- `LSTSQ a b`: Least-squares (minimum-norm) solve of `a x = b` for **any** shape of `a` — over-determined, under-determined, or square-but-singular — via the Moore-Penrose pseudo-inverse (SVD-based). `b` a `Vector` with length matching `a`'s row count. Result is a `Vector`. **Never errors on a non-square or singular `a`**, unlike `SOLVE` — the two are deliberately distinct keywords, not one polymorphic operator, so `SOLVE` keeps erroring loudly on non-square input rather than silently falling back to least-squares.
- `EIGENVALUES a`: Real eigenvalues of a **symmetric** matrix only (guarantees real results — see `EIGENVALUES_GENERAL` below for a non-symmetric matrix, whose eigenvalues can be genuinely complex). `a` must be square and symmetric (checked within a relative numerical tolerance; a non-symmetric input errors rather than silently producing a wrong answer). Result is a `Vector` of eigenvalues in no particular guaranteed order.
- `EIGENVALUES_GENERAL a`: Eigenvalues of a **general** (not necessarily symmetric) square matrix, via Schur decomposition — no symmetry requirement, and the result can be genuinely complex. Result is a `Matrix(2, N)` (row 0 = real parts, row 1 = imaginary parts), the same convention `FFT` uses. `EIGENVALUES` itself is unchanged, still symmetric-only.
- `CHOLESKY a`: Cholesky decomposition (`a = L * Lᵗ`) of a symmetric **positive-definite** matrix. Result is the lower-triangular `Matrix` `L`. **Errors if `a` isn't positive-definite.**
- `PCA a COMPONENTS k`: Projects `a`'s rows (samples) onto their top-`k` principal components — mean-centers each column, then keeps the top-`k` components of the centered data's SVD. `k` must be between 1 and `a`'s column count. Result is a `Matrix` with the same row count as `a` and `k` columns. Built directly on `SVD` below.

**Multi-output** (bind with `LET a, b[, c] = ...` — see below):

- `QR a`: QR decomposition (`a = Q * R`) of any rectangular `Matrix`. Two outputs, bind order `Q`, `R`.
- `LU a`: LU decomposition with partial pivoting (`P * a = L * U`). `a` must be square. Three outputs, bind order `P`, `L`, `U` — `P` is included specifically so `P @ a == L @ U` actually holds; a caller that only kept `L`/`U` and dropped `P` would find that equality silently false for any input that needs row pivoting.
- `EIGEN a`: Full eigendecomposition (eigenvalues + eigenvectors) of a **symmetric** matrix only. Two outputs, bind order eigenvalues (`Vector`), eigenvectors (`Matrix`, as columns). Unchanged — see `EIGEN_GENERAL` below for the non-symmetric case.
- `EIGEN_GENERAL a`: Full eigendecomposition of a **general** square matrix. Two outputs, bind order eigenvalues (`Vector`), eigenvectors (`Matrix`, as columns) — same shape as `EIGEN`. **Real eigenvalues only**: there is no general complex-eigenvector solver here, so this **errors loudly** if any eigenvalue is genuinely complex (use `EIGENVALUES_GENERAL` for the eigenvalues alone in that case). Not exact for a *defective* matrix (a repeated eigenvalue with no full eigenvector basis) — documented, not silently claimed exact.
- `SVD a`: Singular value decomposition (`a = U * diag(s) * Vᵗ`) of any rectangular `Matrix`. Three outputs, bind order `U`, `s` (`Vector` of singular values), `Vᵗ`.

```sql
MATRIX m = [[4, 7], [2, 6]]
LET tr = TRACE m           -- 10.0
LET det = DETERMINANT m    -- 10.0
LET inv = INVERSE m        -- Matrix(2, 2)
VECTOR b = [4, 6]
LET x = SOLVE m b          -- Vector(2): solves m @ x = b

MATRIX overdetermined = [[1, 1], [2, 1], [3, 1]]
VECTOR y = [2.1, 3.9, 6.05]
LET fit = LSTSQ overdetermined y   -- Vector(2): least-squares [slope, intercept]

LET q, r = QR m            -- multi-output LET: two names, one expression
LET p, l, u = LU m
MATRIX sym = [[2, 1], [1, 2]]
LET vals, vecs = EIGEN sym
LET u, s, vt = SVD m

MATRIX rot = [[0, -1], [1, 0]]       -- eigenvalues are +-i, genuinely complex
LET spec = EIGENVALUES_GENERAL rot   -- Matrix(2, 2): real row [0, 0], imag row [1, -1]
MATRIX tri = [[2, 1], [0, 3]]        -- non-symmetric, but real eigenvalues (2, 3)
LET gvals, gvecs = EIGEN_GENERAL tri
```

#### Multi-output `LET`

`LET a, b[, c] = <expr>` binds more than one name from a single expression
in one statement — the only expressions that support this are the
decompositions above with more than one natural output (`QR`/`LU`/`EIGEN`/
`EIGEN_GENERAL`/`SVD`). The number of names must match that expression's real output count
exactly; a mismatch (either direction) is a clear error, not a silent
truncation or `NULL`-fill. Using a single-output operator (e.g. `CHOLESKY`)
with multi-output `LET`, or a multi-output operator with a plain single-name
`LET`, is also a clear error pointing at the correct form to use instead.

---

## 4. Query & Engineering (SQL)

### Inline Vector Literals

Any SQL expression can contain an inline vector literal. The syntax mirrors Python list notation:

```sql
SELECT id, COSINE_SIM(embedding, [0.1, 0.2, 0.3]) AS score FROM docs
SELECT id, VEC_ADD(v, [0.0, 0.0, 1.0]) AS shifted FROM vecs
SELECT L2_NORM([3.0, 4.0]) AS five  -- FROM is optional for a literal/computed-only SELECT
```

### Vector Scalar Functions

Use inside SELECT columns, WHERE predicates, or ORDER BY:

| Function | Signature | Returns | Description |
|---|---|---|---|
| `NORMALIZE(v)` | `Vector → Vector` | Unit vector | Scales `v` to L2 norm = 1 |
| `L2_NORM(v)` | `Vector → Float` | Euclidean length | `√(∑ vᵢ²)` |
| `COSINE_SIM(a, b)` | `Vector, Vector → Float` | [-1, 1] | `dot(a,b) / (‖a‖ · ‖b‖)` |
| `DOT(a, b)` | `Vector, Vector → Float` | Scalar | Dot product `∑ aᵢbᵢ` |
| `DISTANCE(a, b)` | `Vector, Vector → Float` | Euclidean distance | `√(∑ (aᵢ-bᵢ)²)` — magnitude-sensitive, unlike `COSINE_SIM` (also usable inside `SELECT`, in addition to the standalone `DISTANCE a TO b` keyword form in §3) |
| `VEC_ADD(a, b)` | `Vector, Vector → Vector` | Same dim | Element-wise addition |
| `VEC_SCALE(v, s)` | `Vector, Float → Vector` | Same dim | Multiply all elements by `s` |
| `MAT_SHAPE(m)` | `Matrix → String` | e.g. `"2x2"` | Shape of a matrix value as `"rows x cols"` |
| `MATMUL(a, b)` | `Matrix, Matrix/Vector → Matrix/Vector` | Product | Standard matrix multiplication (also usable inside `SELECT`, unlike the standalone `MATMUL a b` keyword form in §3) |
| `TRANSPOSE(m)` | `Matrix → Matrix` | Swapped dims | Transpose a matrix value (also usable inside `SELECT`) |

**Typical similarity search**:

```sql
SELECT id, title, COSINE_SIM(embedding, [0.9, 0.1, 0.0]) AS score
FROM docs
WHERE COSINE_SIM(embedding, [0.9, 0.1, 0.0]) > 0.7
ORDER BY score DESC
LIMIT 10
```

**`COSINE_SIM` is angle-only, not magnitude-aware** — `[1, 1, 1]` and
`[1000, 1000, 1000]` score a perfect `1.0`. This is the right tool for
pre-normalized semantic embeddings (text/image models, where direction
*is* the meaning), but the wrong one for vectors whose components share a
physical scale — masses, prices, counts, distances — where two very
different-magnitude points can end up scoring as near-identical. Use
`DISTANCE` (Euclidean) instead when magnitude itself carries the signal;
see `examples/gw_transient_analysis.lnl` §2 for a real, verified case
where `COSINE_SIM` fails to separate a ~24x mass difference and
`DISTANCE` on the same data correctly does.

### Vector Aggregate Functions

Compute element-wise statistics across all rows in a group:

| Function | Description |
|---|---|
| `AVG_VEC(col)` | Element-wise average — produces the centroid of all vectors in the group |
| `SUM_VEC(col)` | Element-wise sum across all vectors in the group |

```sql
-- Compute per-category centroids
SELECT category, AVG_VEC(embedding) AS centroid
FROM docs
GROUP BY category

-- Compute total embedding mass per user
SELECT user_id, SUM_VEC(event_vector) AS total
FROM events
GROUP BY user_id
```

### SELECT

Query datasets with familiar syntax.

```sql
SELECT region, AVG(score) 
FROM diagnostics 
WHERE id > 100 
GROUP BY region 
HAVING AVG(score) > 0.5 
LIMIT 10
```

- **`GROUP BY` output order**: without `ORDER BY`, groups come out in the order each group's
  first row appears in the input. That order is deterministic, so the same query over the same
  data always returns the same row order and the same content hash. Before v0.1.91 the order was
  random per run. Use `ORDER BY` for any other order. Rows that tie under `ORDER BY` keep their
  group order, so `ORDER BY ... LIMIT n` with ties is also reproducible.
- **Aggregate Functions**: `SUM`, `AVG`, `COUNT`, `MIN`, `MAX`, `AVG_VEC`, `SUM_VEC`, `VARIANCE`, `MEDIAN`. A `SELECT` with an aggregate and no `GROUP BY` computes a single "global" aggregate row over the whole result set (e.g. `SELECT COUNT(*) FROM t`).
- **`VARIANCE(col)`/`MEDIAN(col)`**: population variance and median of a scalar (`Int`/`Float`/`Float64`) column, computed per group (or globally, with no `GROUP BY`) exactly like `SUM`/`AVG`. Both always produce a `DOUBLE` result regardless of the input column's own numeric type. Neither is supported as a window function (`OVER`) — `SELECT VARIANCE(x) OVER (...)` is a parse error, not a silently wrong result.
- **`SUM`/`AVG` on a `Complex` column**: well-defined and fully supported, plain or windowed (`OVER (...)`). **`MIN`/`MAX` on a `Complex` column are a hard error** — `Complex` has no ordering (§3), so there is no "smallest"/"largest" value to silently guess at.
- **`HAVING` on an aliased aggregate**: `HAVING` resolves an aggregate by alias too, not just by its bare call — `SELECT region, AVG(score) AS avg_score FROM diagnostics GROUP BY region HAVING avg_score > 0.5` matches rows the same as `HAVING AVG(score) > 0.5` would.
- **Filtering**: `WHERE` or `FILTER` can be used interchangeably.
- **`DISTINCT`**: `SELECT DISTINCT <cols> FROM ...` removes duplicate rows from the result.
- **`LIMIT`/`OFFSET`**: `LIMIT <n>` caps the row count; `OFFSET <n>` skips the first `n` rows before applying `LIMIT` (both may be used together or independently).
- **`FROM` is optional** for a `SELECT` list of only literal/computed expressions — no column, aggregate, or window reference, since there'd be no dataset to resolve one against: `SELECT L2_NORM([3.0, 4.0]) AS five` evaluates the list once and returns a single row. Any real column reference still requires `FROM`.

**Predicate vocabulary** (usable in `WHERE`/`FILTER`/`HAVING`):

```sql
SELECT * FROM items WHERE category IN ('a', 'b', 'c')
SELECT * FROM items WHERE price BETWEEN 5 AND 25
SELECT * FROM items WHERE tag IS NULL
SELECT * FROM items WHERE tag IS NOT NULL
SELECT * FROM items LIMIT 10 OFFSET 20
```

- `<expr> IN (<v1>, <v2>, ...)`: true if `<expr>` equals any of the listed values.
- `<expr> BETWEEN <low> AND <high>`: inclusive range check, equivalent to `<expr> >= <low> AND <expr> <= <high>`.
- `<expr> IS NULL` / `<expr> IS NOT NULL`: null checks.

**Automatic partition pruning**: a `WHERE`/`FILTER` predicate of the form `col <op> literal` (`<`, `<=`, `>`, `>=`, either operand order) or `col BETWEEN low AND high`, against a column with no index at all, is still optimized automatically once the dataset is large enough to span more than one internal 1024-row partition. Each partition tracks its own column min/max; a partition whose range can't satisfy the predicate is skipped without reading its rows. This needs no `CREATE INDEX` and no special syntax — it's purely a scan optimization, so the query's result is identical with or without it. Datasets under ~1024 rows never engage it (nothing to prune yet).

### Subqueries in FROM

A `SELECT`'s `FROM` clause can be another `SELECT`, wrapped in parentheses
and given an alias:

```sql
SELECT * FROM (SELECT id, price FROM items WHERE price > 5) AS cheap
```

`FROM (<SELECT>) AS <alias>` executes the inner query first and treats its
result as the outer query's source, referenced by `<alias>`. The result lives only for that
query: `<alias>` isn't created as a dataset, so the same query can run any number of times.
Until the fix, the alias leaked as a permanent dataset and a second run failed with `Dataset
name already exists`.

### INSERT / UPDATE / DELETE

```sql
-- Positional values, in column-declaration order
INSERT INTO users VALUES (1, "alice", 30, true)

-- Named values, any order, only the columns you specify
INSERT INTO users (id = 1, name = "alice", active = true)

-- Vector / Matrix literals work in either form
INSERT INTO docs VALUES (1, [0.1, 0.2, 0.3])
INSERT INTO grids (id = 1, m = [[1, 0], [0, 1]])

UPDATE users SET active = false, name = "bob" WHERE id = 1

DELETE FROM users WHERE active = false
```

- `INSERT` values may be `NULL`, a string, a number, `true`/`false`, a bracketed vector `[..]`/matrix `[[..], ..]` literal, or a bare identifier referencing an existing tensor.
- `UPDATE ... SET` accepts one or more `col = expr` assignments (comma-separated) and an optional `WHERE`/`FILTER` predicate; omitting the predicate updates every row.
- `DELETE FROM` accepts an optional `WHERE`/`FILTER` predicate; omitting it deletes every row.

### JOIN

```sql
SELECT o.id, u.name FROM orders o JOIN users u ON o.user_id = u.uid

SELECT id, name FROM orders JOIN users ON orders.user_id = users.uid

SELECT * FROM a LEFT JOIN b ON a.key = b.key
SELECT * FROM a RIGHT JOIN b ON a.key = b.key
SELECT * FROM a FULL JOIN b ON a.key = b.key

-- Index-accelerated similarity join: joins on cosine similarity instead
-- of equality, using a Vector index on the right dataset's column when
-- one exists (falls back to a brute-force comparison otherwise)
SELECT aid, bid FROM a JOIN b ON COSINE_SIM(a.v, b.v) > 0.8
```

- Kinds: `[INNER] JOIN`, `LEFT [OUTER] JOIN`, `RIGHT [OUTER] JOIN`, `FULL [OUTER] JOIN`. Multiple `JOIN` clauses may be chained on one `SELECT`.
- `ON` supports scalar equality (`<left> = <right>`) or, for two `Vector` columns, similarity (`COSINE_SIM(<left>, <right>) > <threshold>`) — no other comparison operator (`>=`, `<`, etc.) is supported for the similarity form yet.
- A dataset in `FROM`/`JOIN` may be given an alias: `FROM orders o` or `FROM orders AS o` (`AS` is optional). Table qualifiers (`col`, `table.col`, or `alias.col`) work the same way anywhere a column is referenced — `ON`, `WHERE`, and the `SELECT` list — but **the qualifier itself is not used to disambiguate**: only the bare column name is resolved, so column names must still be unique across the joined datasets (a self-join's two sides are distinguished only by the built-in `r_`-prefix collision renaming on `SELECT *`, not by which alias you write).
- An unaliased expression in the `SELECT` list (including a qualified column like `a.id`, or any computed expression like `price * 2`) gets an auto-generated column name (`__cmp_0`, `__cmp_1`, ...) — give it an explicit `AS name` if you need a predictable name.

### Common Table Expressions (CTEs) & UNION

```sql
WITH recent AS (SELECT * FROM events WHERE ts > 100) SELECT * FROM recent WHERE user_id = 1

-- Multiple CTEs
WITH cte_a AS (SELECT * FROM t1), cte_b AS (SELECT * FROM t2) SELECT * FROM cte_a

SELECT id FROM users_a UNION SELECT id FROM users_b

SELECT id FROM users_a UNION ALL SELECT id FROM users_b
```

- `WITH <name> AS (<SELECT>), ...` computes each CTE before the main query runs, and keeps its rows in the query's own scope. A CTE is never created as a dataset, and the name isn't available in later statements.
  - **Visibility:** a CTE is visible to later CTEs in the same `WITH`, to the main query (including its `JOIN`s), to subqueries in `FROM`, and to the right side of a `UNION`.
  - **Shadowing:** a CTE with the same name as a real dataset shadows it for that query only, as in SQL. The dataset itself is untouched. This used to fail, because the CTE was created as a temporary dataset under that name.
- `UNION` deduplicates matching rows; `UNION ALL` keeps duplicates. `UNION`/`UNION ALL` clauses can be chained (`A UNION B UNION C`, three-way and beyond) — each right-hand side is itself a full `SELECT`, so chaining just recurses.
- **A `WITH` clause's trailing `SELECT` must be part of the same statement** — in `.lnl` files and the CLI/REPL, keep the whole `WITH ... SELECT ...` on one line. Splitting it across lines the way this section's examples are formatted above for readability (`WITH recent AS (\n ...\n)\nSELECT ...`) does *not* work when actually pasted into a `.lnl` file: `linal run`'s line joiner only tracks paren balance, and the `WITH` clause's own parens close before the file reaches the trailing `SELECT`, so the joiner treats the `WITH ... AS (...)` part as a complete (and invalid) statement on its own. This is a real gap in the file-runner's statement-joining heuristic, not a DSL limitation — the parser itself accepts the full multi-line form fine when given as one string (e.g. over `/execute`).

### Window Functions

```sql
SELECT id, price, ROW_NUMBER() OVER (ORDER BY price DESC) AS rn FROM items

SELECT id, price, category,
       RANK() OVER (PARTITION BY category ORDER BY price DESC) AS rk
FROM items

SELECT id, price, category,
       DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) AS drk
FROM items

SELECT id, price, LAG(price) OVER (ORDER BY id) AS prev_price FROM items
SELECT id, price, LEAD(price, 2) OVER (ORDER BY id) AS next2_price FROM items

-- Multiple window functions with different PARTITION BY / ORDER BY specs
-- can be freely combined in one SELECT:
SELECT id, price,
       ROW_NUMBER() OVER (ORDER BY price DESC) AS rn,
       RANK() OVER (PARTITION BY category ORDER BY price DESC) AS rk,
       DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) AS drk,
       LAG(price) OVER (ORDER BY id) AS prev_price,
       LEAD(price, 2) OVER (ORDER BY id) AS next2_price,
       SUM(price) OVER (PARTITION BY category ORDER BY id) AS running_total
FROM items
```

- Ranking functions: `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()` — no arguments.
- Offset functions: `LAG(col [, offset])`, `LEAD(col [, offset])` — `offset` defaults to `1`.
- Aggregate-as-window: any of `SUM`, `AVG`, `COUNT`, `MIN`, `MAX` (or `SUM_VEC`/`AVG_VEC`) followed by `OVER (...)` computes a running aggregate within the window instead of collapsing to one row.
- `OVER (...)` accepts an optional `PARTITION BY col [, col ...]` and an optional `ORDER BY col [ASC|DESC] [, ...]` — at least one of the two should be present for a meaningful window; `ORDER BY` on a Vector/Matrix column inside `OVER (...)` is rejected (see §1 — these types have no defined ordering).
- Unaliased default column names: ranking/offset functions use the lowercase function name (e.g. `row_number`, `rank`, `lag`); aggregate-as-window functions instead default to `<func>(expr)_over` (e.g. `sum(expr)_over`) — always give an explicit `AS alias` rather than relying on either default.

### CASE, COALESCE, NULLIF, CAST

```sql
-- Searched CASE
SELECT id, CASE WHEN score > 90 THEN "A" WHEN score > 80 THEN "B" ELSE "C" END AS grade FROM students

-- Simple CASE (operand form)
SELECT id, CASE status WHEN 1 THEN "active" WHEN 0 THEN "inactive" ELSE "unknown" END AS label FROM accounts

SELECT id, COALESCE(nickname, name, "anonymous") AS display_name FROM users
SELECT id, NULLIF(score, 0) AS score_or_null FROM results   -- NULL if score = 0
SELECT id, CAST(price AS INT) AS price_int FROM items

-- Reshape a Vector/Matrix column inline in a query
SELECT id, CAST(flat_embedding AS MATRIX(2, 2)) AS as_matrix FROM t
SELECT id, CAST(grid AS VECTOR(6)) AS flattened FROM t
SELECT id, FLATTEN(grid) AS flattened FROM t   -- equivalent, no shape needed
```

- `CASE [operand] WHEN <cond> THEN <expr> [WHEN ... THEN ...] [ELSE <expr>] END` — with an operand, each `WHEN` value is compared for equality against it; without one, each `WHEN` is a standalone boolean condition.
- `COALESCE(a, b, ...)` returns the first non-`NULL` argument (2+ args). `NULLIF(a, b)` (alias `IFNULL`) returns `NULL` if `a = b`, else `a`.
- `CAST(expr AS <type>)` — scalar target types: `INT`/`INTEGER`, `FLOAT`/`FLOAT32` (32-bit), `DOUBLE`/`FLOAT64` (64-bit, full precision), `TEXT`/`STRING`/`VARCHAR`, `BOOL`/`BOOLEAN`.
- `CAST(expr AS VECTOR(n))` / `CAST(expr AS MATRIX(r, c))` — reshape/flatten a `Vector`/`Matrix` value to the given shape, row-major. The source and target must have the same total element count (`r * c == n` when converting between the two, or an exact length/shape match for same-kind casts); a mismatch returns `NULL` rather than resizing or erroring, consistent with other invalid `CAST` combinations. This is the way to reshape *to an arbitrary shape* inside a query — the standalone `RESHAPE` keyword (§3) only operates on tensor variables outside of `SELECT` (`RESHAPE(...)` inside a query does not parse).
- `FLATTEN(expr)` also works inside `SELECT` (in addition to its standalone tensor-DSL form, §3) — flattens a `Matrix` row-major into a `Vector`, or is a no-op on an already-flat `Vector`. Equivalent to `CAST(expr AS VECTOR(total_element_count))` but without needing to know the count up front.

### String Functions

| Function | Signature | Description |
|---|---|---|
| `UPPER(s)` | `String → String` | Uppercase |
| `LOWER(s)` | `String → String` | Lowercase |
| `LENGTH(s)` | `String → Int` | Character count |
| `TRIM(s)` | `String → String` | Strip leading/trailing whitespace |
| `CONCAT(a, b, ...)` | `String... → String` | Concatenate 2+ strings |
| `SUBSTR(s, start [, len])` | `String, Int, Int? → String` | 1-based substring; omit `len` to take the rest of the string |

```sql
SELECT SUBSTR(name, 1, 3) AS prefix, UPPER(TRIM(email)) AS clean_email FROM users
```

### Semantic Transforms (Zero-Copy)

- `BIND alias TO resource`: Create a semantic link (alias) to a tensor or dataset.
- `ATTACH tensor TO ds.col`: Link an independent tensor into a dataset column.
- `DERIVE target FROM expr`: Create a new resource with full automated lineage tracking. `expr` must be a real computed expression — `DERIVE b FROM a` (a bare identifier, nothing to derive) is a clear error pointing at `BIND`/`LET` instead, since aliasing creates no new lineage node to track.
- `LET name = <bare identifier>` is also a zero-copy alias, equivalent to `BIND name TO <bare identifier>` — both accept a plain tensor name, or a `dataset()`-constructed tensor-first dataset variable. `LAZY LET name = <bare identifier>` is a clear error (there is nothing to defer in a plain alias).

### Schema Evolution

- `ALTER DATASET ds ADD COLUMN col: type [DEFAULT val]`
- `ALTER DATASET ds ADD COLUMN col = expression [LAZY]`
- `MATERIALIZE ds`: Physicalize all `LAZY` columns in a dataset.
- `SET DATASET ds [METADATA] key = "value"`: Attach an arbitrary string metadata key to a dataset (the `METADATA` keyword is optional). Retrieve it with `SHOW DATASET METADATA <name>` (§9).

---

## 5. Persistence & Ingestion

Load and save data across different formats.

- `USE DATASET FROM "path" [AS name] [FIELDS (name1, name2, ...)]`: Load external data (CSV, HDF5, NetCDF, NumPy, Parquet, Zarr) into the current session as ephemeral tensors and a dataset view.
  - Automatically detects format from file extension (`.csv`, `.h5`/`.hdf5`/`.h5ad`, `.nc`, `.npy`, `.npz`, `.parquet`, `.zarr`).
  - `FIELDS (...)`: explicitly pick which named columns/datasets/arrays to ingest, by exact name. Without it, a source that bundles fields of different shapes (e.g. an HDF5 file with a `(10, 64)` array and a `(7,)` array) keeps whichever fields share the first-encountered one's shape and silently-but-loudly skips the rest (reported as a warning). With `FIELDS`, only the named fields are read — a name that doesn't exist in the source, or a set of named fields that can't share one row count, is a hard error instead of a skip, since you've said exactly what you want.
- `IMPORT DATASET FROM "path" [AS name] [FIELDS (name1, name2, ...)]`: Load and normalize external data into a persistent LINAL Dataset Package.
  - Supports CSV, HDF5, NetCDF, NumPy, Parquet, and Zarr. `FIELDS (...)` works the same way as for `USE DATASET FROM` above.
- `IMPORT CSV FROM "path" AS name`: (Legacy) Auto-infer schema and load CSV into a relational dataset.
- `EXPORT [CSV] name TO "path"`: Save dataset to CSV. The `CSV` keyword is optional — `EXPORT name TO "path"` behaves identically. A `Vector`/`Matrix` column is written as a JSON string per cell (e.g. `{"Vector":[1.0,2.0,3.0]}`), the same encoding `SAVE DATASET`'s legacy fallback uses — CSV has no native representation for nested/list data. Use `SAVE DATASET` instead for a native binary (Parquet `FixedSizeList`) encoding of vector/matrix data.
- `SAVE DATASET name [TO "path"]`: Persist to Parquet (includes metadata/lineage).
- `LOAD DATASET name [FROM "path"]`: Restore a persisted dataset.
- `SAVE TENSOR name [TO "path"]`: Persist a tensor to JSON.
- `LOAD TENSOR name [FROM "path"]`: Restore a persisted tensor (preserves lineage).
- `SAVE PIPELINE name [TO "path"]`: Serialize a named pipeline to JSON. Defaults to `<data_dir>/<db>/pipelines/<name>.json`.
- `LOAD PIPELINE name [FROM "path"]`: Restore a pipeline from its JSON file by re-parsing the stored DSL source. Overwrites any in-memory definition with the same name.
- `LIST DATASETS [FROM "path"]`: Show available datasets in the current database context.
- `LIST DATASET PACKAGES`: Equivalent to `LIST DATASETS` — lists the same persisted dataset packages.
- `LIST TENSORS [FROM "path"]`: Show available tensors in the current storage path.
- `LIST DATASET VERSIONS <name>`: Show version history and schema evolution log for a persisted dataset.

### Scientific Data Ingestion

LINAL supports direct ingestion of multi-dimensional data:

- **HDF5**: Ingests datasets from groups; flattens them into columns. `.h5`/`.hdf5`/`.h5ad`
  files are read as opaque generic containers (no attribute interpretation) — see **NetCDF**
  below for the CF-aware alternative.
- **NetCDF** (`.nc`): Real CF-convention semantics, not just opaque HDF5 bytes (NetCDF4 files
  are HDF5 under the hood). Per variable: `scale_factor`/`add_offset` unpacking
  (`unpacked = raw * scale_factor + add_offset`), `_FillValue`/`missing_value` mapped to `NaN`,
  and `units`/`standard_name`/`long_name` surfaced as column metadata (visible via `SHOW SCHEMA`).
  Scoped to float-stored variables — integer-packed variables (the classic raw-satellite-data
  case) aren't supported, since this engine's connectors are float-only throughout.
- **Numpy**: Supports `.npy` (single vector/matrix) and `.npz` (named collections).
- **Parquet** (`.parquet`): Generic external Parquet ingestion (e.g. a Pandas/Arrow/Spark
  export) — distinct from `SAVE`/`LOAD DATASET`'s own internal Parquet dataset-package format,
  which never goes through `USE`/`IMPORT DATASET FROM` at all.
- **Zarr**: Supports V3 Zarr stores with recursive group traversal.

A source file can bundle several fields of different shapes (e.g. an HDF5
file with both a `(10, 64)` data matrix and a `(10,)` label vector). Since
one LINAL dataset from `USE`/`IMPORT DATASET FROM` is one row-count-aligned
table, only fields that share a common row count can end up in the same
result. Use `FIELDS (...)` to pick exactly which ones:

```sql
-- Only the "labels" field, even though the file also has a differently-
-- shaped "embeddings" field.
USE DATASET FROM "vectors.h5" AS d FIELDS (labels)
```

---

## 6. Pipelines

Named, reusable transformation chains that can be saved to disk and restored across sessions.

### Pipeline Lifecycle

```sql
-- Define
DEFINE PIPELINE clean AS WHERE active = 1 THEN ORDER BY score DESC THEN LIMIT 10

-- Inspect
SHOW PIPELINES
DESCRIBE PIPELINE clean

-- Apply
APPLY PIPELINE clean ON products INTO top_products
APPLY PIPELINE clean ON products          -- in-place (overwrites source)

-- Persist
SAVE PIPELINE clean                        -- saves to <data_dir>/<db>/pipelines/clean.json
SAVE PIPELINE clean TO '/backups/clean.json'

-- Restore
LOAD PIPELINE clean
LOAD PIPELINE clean FROM '/backups/clean.json'

-- Remove
DROP PIPELINE clean
```

### Pipeline Steps

Steps are chained with `THEN`:

| Step | Syntax | Description |
|---|---|---|
| Projection | `SELECT col [AS alias], ...` | Keep/rename columns |
| Filter | `WHERE expr` / `FILTER expr` | Row predicate |
| Sort | `ORDER BY col [ASC\|DESC] [, ...]` | Row ordering |
| Limit | `LIMIT n` | Cap row count |
| Normalize | `NORMALIZE col` | L2-normalize a vector column |

### Pipeline Persistence Details

Pipelines are stored as human-readable JSON containing the original DSL source:

```json
{ "name": "clean", "source": "DEFINE PIPELINE clean AS WHERE active = 1 THEN LIMIT 10", "version": "0.1.46" }
```

The `version` field records the LINAL version that saved the pipeline (`env!("CARGO_PKG_VERSION")` at save time) — it's informational only, not a compatibility gate. On load, the source is re-parsed to reconstruct the pipeline exactly. The file is editable — any valid `DEFINE PIPELINE` DSL can replace the source field.

---

## 7. Vector Search & Indexing

### CREATE INDEX

```sql
CREATE INDEX ON docs(category)
CREATE INDEX my_idx ON docs(category)      -- name is optional and currently unused
CREATE VECTOR INDEX ON docs(embedding)
CREATE VECTOR INDEX ON docs(embedding) USING HNSW   -- opt into an HNSW graph index instead
```

- `CREATE INDEX [<name>] ON <dataset>(<column>)`: Build a standard lookup index on a scalar column.
- `CREATE VECTOR INDEX [<name>] ON <dataset>(<column>) [USING HNSW]`: Build an index-accelerated structure over a `Vector` column, enabling `SEARCH` and index-aware `COSINE_SIM` filtering in `WHERE` clauses. Without `USING HNSW` (the default), this builds the IVF-clustered index described below. With `USING HNSW`, it instead builds an HNSW (Hierarchical Navigable Small World) graph index — see "HNSW vector index" below for how the two differ.
  - **Hybrid filters**: `WHERE COSINE_SIM(embedding, [...]) > threshold AND category = 'electronics'` is index-accelerated too, not just the bare `COSINE_SIM` comparison alone — the query planner finds the `COSINE_SIM(...) > threshold` conjunct anywhere in a top-level `AND` chain (any position, any number of other conjuncts), routes it through the vector index, and applies the remaining conjuncts as a post-filter on those results. `EXPLAIN` always shows whether this actually fired (look for `CosineFilterExec` in the physical plan) — an `AND` predicate whose `COSINE_SIM` conjunct doesn't match this shape, or whose column has no *IVF* vector index (an HNSW-only index never accelerates this path — see below), falls back to a full scan+filter exactly as before, never silently wrong, just unaccelerated.
- List existing indexes with `SHOW INDEXES [<dataset>]` (§9) — an HNSW index is reported as type `VECTOR (HNSW)`, distinct from a plain IVF `VECTOR` index.
- **Persistence**: `SAVE DATASET` writes which columns are indexed (and with what index type) alongside the data; `LOAD DATASET` rebuilds each one from the reloaded rows automatically. Before this, a `CREATE INDEX` only lived for the current process — reloading a saved dataset silently lost every index with no warning. `LOAD DATASET`'s output message now reports which indexes were restored (e.g. `"... indices restored on: category, embedding"`).
  - **Vector index clustering is persisted too**, not just the (column, type) definition: `SAVE DATASET` also writes a `vector_index_clusters.json` snapshot of a `CREATE VECTOR INDEX` column's k-means clustering (once it's large enough to have actually clustered — see below), and `LOAD DATASET` restores that clustering directly instead of recomputing it, avoiding a potentially expensive k-means rebuild on every load. A content hash of the column's data travels with the snapshot; if it no longer matches the freshly loaded column (e.g. `data.parquet` edited independently of the snapshot), `LOAD DATASET` silently falls back to a full rebuild rather than trusting a stale clustering — the restored-indexes message distinguishes the two (`"... (from snapshot: embedding)"` vs. plain `"... indices restored on: embedding"`).
  - **An HNSW index's built graph is persisted the same way**, in a sibling `hnsw_index_graphs.json`, with the same content-hash staleness check and the same `"... (from snapshot: embedding)"` restored-indexes message.
- **Vector index clustering**: `CREATE VECTOR INDEX` automatically clusters the column's vectors (IVF-style, k-means with a cosine-similarity metric) once the column has at least ~64 rows — no extra syntax, this is transparent. Below that size, or before enough rows exist, it falls back to the original brute-force scan. `SEARCH`/`SELECT ... ORDER BY COSINE_SIM(...)` (approximate top-k) only probe the nearest few clusters; `WHERE COSINE_SIM(...) > threshold` (an exact predicate, not a ranking) instead uses a provable per-cluster similarity bound to skip clusters that provably can't contain a match, so it never drops a qualifying row.
- **HNSW vector index**: `CREATE VECTOR INDEX ... USING HNSW` builds an HNSW graph once the column has at least ~16 rows (below that, brute-force scan, same idea as IVF's ~64-row threshold). It only accelerates top-k `SEARCH`/`ORDER BY COSINE_SIM(...)` — unlike IVF's clusters, an HNSW graph traversal has no cheap provable bound on what it might have skipped, so it can't safely accelerate an *exact* `WHERE COSINE_SIM(...) > threshold` predicate the way IVF's per-cluster bound does. A column with only an HNSW index still answers such a `WHERE` correctly (falls back to a full scan+filter, exact, just not index-accelerated) — it never errors and never drops a qualifying row. Rows inserted after the index was created (or after the last `LOAD DATASET`) are always additionally brute-force scanned regardless of index type, so correctness never depends on how recently the graph/clustering was (re-)built, only performance does. `EXPLAIN` reports which index type (`Vector`, `Hnsw`, or none) a `SEARCH` will actually use.

### SEARCH (Vector Similarity)

The modern form:

```sql
SEARCH docs ON embedding QUERY [0.9, 0.1, 0.0] LIMIT 10
SEARCH docs ON embedding QUERY my_query_tensor LIMIT 10 INTO results
SEARCH docs ON embedding QUERY [0.9, 0.1, 0.0] LIMIT 10 FILTER category = "electronics"
```

- `SEARCH <dataset> ON <column> QUERY <[vector literal]|tensor_name> LIMIT <k> [FILTER <predicate>] [INTO <target>]`
- Returns the top-`k` nearest rows by cosine similarity; `INTO <target>` materializes the results as a new dataset instead of returning them inline.
- **`FILTER <predicate>`** (modern syntax only — the two alternate forms below don't have it): applies `<predicate>` to the `k` nearest-neighbor results as a post-filter, using the same predicate vocabulary as `WHERE`/`FILTER` in `SELECT` (§4). This is a **post**-filter, not a pre-filtered/expanded search — a highly selective predicate can return fewer than `k` rows, since filtering happens after the top-`k` candidates are already chosen, not before. A separate keyword from `WHERE`: `WHERE` is already claimed by this statement's alternate query-vector syntax below, so `FILTER` avoids silently breaking those scripts.

Two alternate forms exist and parse to the exact same statement:

```sql
-- WHERE-style shorthand (approx-equals operator ~=)
SEARCH docs WHERE embedding ~= [0.9, 0.1, 0.0] LIMIT 10

-- Legacy explicit-target form
SEARCH results FROM docs QUERY [0.9, 0.1, 0.0] ON embedding K=10
```

All three forms **require a `CREATE VECTOR INDEX` on `<column>` first** — `SEARCH` always executes as an index-accelerated lookup and errors if no vector index exists on the target column. For ad hoc similarity scoring without a prebuilt index, use `COSINE_SIM` directly in `SELECT`/`WHERE`/`ORDER BY` (§4) instead — that's the more common pattern for one-off queries; `SEARCH` is specifically for index-accelerated top-k retrieval.

### TRANSFORM

```sql
TRANSFORM docs SELECT id, UPPER(name) AS name_upper WHERE active = 1 INTO clean_docs
TRANSFORM docs SELECT id, UPPER(name) AS name_upper WHERE active = 1   -- overwrites docs in place
```

- `TRANSFORM <source> SELECT <columns> [WHERE <expr>] [INTO <target>]`: A single-shot projection/filter over a dataset, equivalent to `SELECT ... FROM <source> [WHERE ...]` under the hood. With `INTO <target>`, writes the result to `<target>` (creating it if it doesn't exist). **Without `INTO`, it overwrites `<source>` in place** — it does not return results inline like a plain `SELECT`.

---

## 8. Instance & Session Management

### Database Management

LINAL supports multi-platform isolated instances.

```sql
CREATE DATABASE research
USE research
DROP DATABASE obsolete_db
SHOW DATABASES          -- also: SHOW ALL DATABASES
```

### RESET SESSION

Clears all in-memory registers (Tensors and Datasets) for the current session.

### Write-ahead log & CHECKPOINT

With the write-ahead log enabled in `linal.toml`, in-memory state survives a restart or crash
without an explicit `SAVE`:

```toml
[wal]
enabled = true          # default: false
sync = "always"         # "always" (fsync per write, survives power loss) or "never"
                        # (survives a process crash / kill -9, not an OS crash)
checkpoint_bytes = 67108864   # auto-checkpoint once wal.log passes this size (default 64 MiB)
```

Every successful mutating statement (`VECTOR`, `LET`, `INSERT`, `UPDATE`, `CREATE INDEX`,
`LOAD`, `IMPORT`, `DEFINE PIPELINE`, ...) is appended to `{data_dir}/{db}/wal.log`. On startup,
each database restores its last checkpoint and replays the log after it. Read-only statements,
database-catalog statements (`CREATE`/`DROP`/`USE DATABASE`) and statements already durable on
disk (`SAVE`, `EXPORT`, `PRUNE LINEAGE`) aren't logged.

```sql
CHECKPOINT      -- snapshot the active database and truncate its log
```

`CHECKPOINT` writes a private snapshot of the active database to `{data_dir}/{db}/checkpoint/`.
It doesn't touch your own `SAVE`d packages or their versions. A checkpoint also happens
automatically:
- after every successful `SAVE`;
- when `wal.log` outgrows `checkpoint_bytes`.

`CHECKPOINT` is an error when the WAL is disabled.

**Limitations:**
- A `LOAD` or `IMPORT` is replayed by re-reading its file. If that file changed since it was
  logged, recovery refuses to guess: the database reports a recovery error on every statement
  until the file is restored or `wal.log` is moved aside. That means losing the changes since the
  last checkpoint.
- Lazy *columns* on record datasets don't survive a checkpoint, the same as `SAVE`/`LOAD`.

---

## 9. Diagnostics

### Resource Display

- `SHOW <name>`: Display contents of any resource — tensor, legacy dataset, or tensor-first dataset. Automatically materializes lazy tensors before displaying.
- `SHOW ALL` / `SHOW ALL TENSORS`: List all in-memory tensors with shapes and data.
- `SHOW ALL DATASETS`: List all legacy datasets with row/column counts.
- `SHOW DATABASES` / `SHOW ALL DATABASES`: List all database instances.
- `SHOW SCHEMA <dataset>`: Display column names and types for a legacy dataset.
- `SHOW SHAPE <name>`: Display only the shape dimensions of a tensor.
- `SHOW LINEAGE <name>`: Display the recursive derivation graph that produced a tensor *or* a dataset — superseded by `EXPLAIN LINEAGE` (§9, Query Planning) below, kept working as an alias for backward compatibility.
- `SHOW INDEXES [<dataset>]`: List all indexes; optionally filter to a specific dataset.
- `SHOW BACKEND`: The active database's compute backend: CPU (SIMD/Rayon), or GPU when enabled
  (see below). `BACKEND` is a contextual keyword, so a tensor or dataset literally named `BACKEND`
  can't be shown with `SHOW BACKEND`.

### Compute backend (experimental GPU)

```toml
# linal.toml
[compute]
backend = "gpu"   # default: "cpu"
```

With a `linal` built with `--features gpu-wgpu` (not part of the default build or the release
binaries), `backend = "gpu"` sends large dense `MATMUL`s to the GPU through `wgpu`: Metal on
macOS, Vulkan/DX12 elsewhere. Everything else stays on the CPU, and so does any matmul under
about 2M multiply-adds (roughly 128×128×128).

If the build lacks the feature or no GPU adapter is found, it warns once and uses the CPU backend.
Results match the CPU within f32 rounding, but not bit-for-bit. This is a measured spike (see
`docs/SCALING_AND_GPU_ROADMAP.md`), not a stable feature.

### Dataset Metadata & Versioning

- `SHOW DATASET METADATA <name>`: Display version, hash, origin, author, tags, and timestamps for a dataset (checks in-memory first, falls back to disk).
- `SHOW DATASET VERSIONS <name>`: Display the full schema evolution history for a persisted dataset.
- `LIST DATASET VERSIONS <name>`: Equivalent to `SHOW DATASET VERSIONS` — returns the same schema history output.

### Utility

- `SHOW "<string>"`: Print a string literal directly. Useful for annotating script output.

```sql
SHOW "--- Begin training phase ---"
```

### Query Planning

- `EXPLAIN [PLAN] <target>`: Show the logical and physical execution plan. The optional `PLAN` keyword is accepted but doesn't change behavior. `<target>` is one of:
  - `EXPLAIN [PLAN] SELECT ...`: plan for a `SELECT` query.
  - `EXPLAIN [PLAN] DATASET <name>`: plan for a plain dataset scan, or, if followed by `FROM <source> ...`, for a `DATASET ... FROM` materialized-view query (§2).
  - `EXPLAIN [PLAN] SEARCH ...`: plan for any of the three `SEARCH` forms (§7).
  - `EXPLAIN <name>`: shorthand for `EXPLAIN DATASET <name>`.

### Lineage & Provenance

- `EXPLAIN LINEAGE <name>`: Show the real, persisted derivation ancestry for a tensor or dataset — a genuinely different thing from `EXPLAIN <target>` above (that shows a *query plan*; this shows *how the data actually got here*: every `IMPORT`, `DATASET ... FROM`, `ADD COMPUTED COLUMN`, tensor op, and `SAVE`, in order). Resolves `<name>` against tensor names first, then dataset names. Survives a restart: ancestry is read from a persisted, content-hash-addressed provenance log (`{data_dir}/{db}/provenance.jsonl`), not just the current session's in-memory state, so it still works on a dataset you just `LOAD`ed fresh. A name with no recorded history (e.g. one that predates this feature) resolves as a single `ROOT` node rather than erroring.
  - `EXPLAIN LINEAGE <name> AS JSON`: same ancestry, as JSON, for programmatic or compliance consumption.
  - `SHOW LINEAGE <name>` is a working, documented-as-superseded alias for the text-tree form.
- `PRUNE LINEAGE BEFORE <RFC3339 timestamp string>`: Compact the provenance log by removing records older than the given cutoff (e.g. `PRUNE LINEAGE BEFORE "2026-01-01T00:00:00Z"`) — a maintenance operation for long-running/edge deployments where `provenance.jsonl` would otherwise grow unbounded. **Never removes a record still needed to resolve `EXPLAIN LINEAGE` for a currently-live tensor or dataset**, no matter how old it is — this is not a blind time-window truncation. Records that are old enough to prune but are still reachable from something live are kept and counted separately in the output message (`"... N retained because a live tensor/dataset's lineage still needs them ..."`), so the message always reflects exactly what happened rather than overclaiming a full prune. A malformed timestamp is a loud parse error, not a silent no-op.
- `AUDIT DATASET <name>`: Perform a deep **referential-integrity** health check — detects dangling tensor references in a dataset's columns. This is unrelated to derivation history despite the naming similarity: `AUDIT DATASET` answers "do this dataset's column references still resolve?"; `EXPLAIN LINEAGE` answers "how was this data derived?". Use `EXPLAIN LINEAGE`, not `AUDIT DATASET`, to inspect provenance. **Only works on tensor-first datasets** (built via the `dataset()` constructor, §2) — it errors `Tensor dataset '<name>' not found` against an ordinary `DATASET <name> COLUMNS (...)` (legacy relational) dataset, even one that exists and works fine with `SHOW`/`SELECT`/etc. Almost every other example in this reference uses the legacy form, so this is easy to hit by surprise.
- `DELIVER <dataset> [TO '<path>']`: Check whether a dataset is deliverable over the `/delivery` HTTP routes (§10). Errors if the dataset doesn't exist. If it exists but hasn't been persisted yet, reports that and points to `SAVE DATASET`; if a delivery manifest is found (default path `<data_dir>/<db>/datasets/<name>/manifest.json`, or the directory given by `TO`), confirms it's ready to serve. **"Doesn't exist" means "not in the current in-memory session"**, not "not on disk" — a dataset saved in an earlier process (e.g. a previous `linal run`) needs an explicit `LOAD DATASET <name>` first, even though it's already persisted; `DELIVER` itself doesn't check disk for a dataset that hasn't been loaded.

---

## 10. Server & Job Management

For remote execution and production workloads.

### Request format

`/execute`, `/execute/batch`, and `/jobs` (`POST`) all take the raw DSL command as the
request body with `Content-Type: text/plain` — not JSON. `/execute` additionally accepts
a legacy `{"command": "..."}` JSON body, but it's deprecated (the server logs a
deprecation warning on every use); prefer `text/plain`. Append `?format=json` for a JSON
response, or `?format=arrow` for a binary Arrow IPC stream (a successful `Table` result
only — anything else under `?format=arrow` falls back to the JSON body; see
`clients/CONTRACT.md` §1 for the wire-level contract), on any of the three — the default
response format is a plain-text "toon" encoding, not JSON. **`/schedule` (`POST`) is the
exception**: it takes a real JSON body
(`{"name": ..., "command": ..., "interval_secs": ..., "target_db": ...}`), since it's
registering a task definition, not executing a command directly.

`/execute/batch`'s body is a whole multi-statement script, in the exact same format as a
`.lnl` file run via `linal run`: one statement per line, or spanning multiple lines — a
statement ends once its parentheses balance out, not at the next line break — with
`#`/`--`/`//` comment lines skipped between statements. Statements run in order; the
batch stops at the first error. The JSON response is `{"status": "ok"|"error",
"statements": [{"statement": ..., "status": ..., "result": ..., "error": ...}, ...]}` —
one entry per statement that actually ran.

### Background Jobs

| Endpoint | Method | Description |
|---|---|---|
| `/jobs` | `POST` | Submit a DSL command for background execution (`text/plain` body — see above). Returns `job_id`. |
| `/jobs` | `GET` | List all jobs and their statuses. |
| `/jobs/:id` | `GET` | Poll a specific job — returns `Pending`, `Running`, `Completed`, or `Failed`. |
| `/jobs/:id/result` | `GET` | Retrieve structured `DslOutput` for a completed job. |
| `/jobs/:id` | `DELETE` | Cancel a **Pending** job. Running or finished jobs cannot be cancelled. |

### Scheduler

Submit recurring DSL commands that execute on a fixed interval:

| Endpoint | Method | Description |
|---|---|---|
| `/schedule` | `POST` | Register a named scheduled command (`name`, `command`, `interval_secs`, optional `target_db`). |
| `/schedule` | `GET` | List all active scheduled tasks. |
| `/schedule/:id` | `DELETE` | Remove a scheduled task by ID. |

### Other Server Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Server health check. |
| `/execute` | `POST` | Execute one DSL statement synchronously (`text/plain` body, one statement per request — see "Request format" above). |
| `/execute/batch` | `POST` | Execute a whole multi-statement script synchronously, in one request (see "Request format" above). |
| `/databases` | `GET` | List database instances. |
| `/databases/:name` | `POST` | Create a database instance. |
| `/databases/:name` | `DELETE` | Drop a database instance. |
| `/delivery/...` | `GET` | Read-only dataset delivery endpoints. |

Multi-tenant isolation is provided via the `X-Linal-Database: <db_name>` request header.
A request that supplies this header is pinned to that database for its whole duration and
never changes the server's active database, so concurrent requests targeting different
databases via the header can't affect each other's context. This applies to
`/execute/batch` too, scoped to the whole batch: a `USE` inside a header-bearing batch
switches databases for the rest of that batch only. **The target database must already exist before you address it with
this header** — `CREATE DATABASE <name>` itself has to run *without* the header (or with
it pointed at an existing database), since the header resolves its target before the
statement runs, and the database you're trying to create doesn't exist yet.

A headerless request's own active-database changes persist across future headerless
requests, exactly like the embedded CLI/REPL — including an explicit `USE <db>`. **A
`USE <db>` statement combined *with* the header on a single `/execute` or `/jobs`
request is rejected as an error**, rather than silently reporting success and reverting:
a single statement has no "rest of the request" for `USE` to usefully persist across, so
the combination is ambiguous and was previously misleading (the response claimed
`"Switched to database 'x'"`, but the switch never outlived that one request). If a
script needs `USE`/`CREATE DATABASE` to control multiple subsequent statements, send it
to `/execute/batch` instead. There, `USE` persists for the rest of that one batch. In a
header-bearing batch it never leaves the batch; in a headerless batch it also becomes the
server's active database, like a headerless `USE` on `/execute`.

### Concurrency and scaling

Each database has its own lock. Requests against different databases run in parallel.
Within one database, reads share the lock and run in parallel: `SELECT` (with CTEs,
subqueries, joins and `UNION`), `SEARCH` without `INTO`, `EXPLAIN`, `AUDIT`, `LIST`, `DELIVER`,
`SHOW` (except of a lazy tensor) and `DESCRIBE PIPELINE`. Everything that writes (`INSERT`,
`UPDATE`, `DELETE`, `SEARCH ... INTO`, `DATASET ... FROM`, tensor statements, ...) is exclusive.
For more write throughput, split data across databases.

Instances don't communicate with each other. To spread databases over several machines, run one
instance per group of databases and route by `X-Linal-Database` with a reverse proxy. See
`ARCHITECTURE.md` → "Scaling & Deployment" for the full model, a proxy example, and what isn't
supported yet: cross-instance queries and replication.

A scheduled task's `target_db` (`/schedule`, above) is a different, simpler case: the
switch it makes is **permanent**, not restored — a recurring task is operator-configured,
not per-visitor request traffic, so there's no "previous" context to protect. Keep this
in mind if you mix scheduled tasks with headerless `/execute` traffic on the same server:
a scheduled task's `target_db` becomes the new session-wide active database for every
following headerless request, until something else changes it again.

- **Graceful Shutdown**: Server handles `SIGINT`/`SIGTERM` to safely close connections.

---

**LINALDB**: *Where SQL meets Linear Algebra.*
Copyright (c) 2025 gorigami (gorigami.xyz)
Licensed under the LinalDB Community License v1.0
