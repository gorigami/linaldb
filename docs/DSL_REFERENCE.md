# LINAL DSL Reference

**LINAL Script** is a high-performance, SQL-inspired language for tensor algebra and relational analytics. This document serves as the complete technical specification for all keywords, operators, and built-in functions.

Line comments start with `--`, `#`, or `//` (all three are equivalent) and run to the end of the line. Blank lines are ignored.

---

## 1. Data Types & Literals

LINAL supports both standard relational types and multi-dimensional numeric structures.

### Relational Types

- `Int`: 64-bit signed integer.
- `Float`: 32-bit floating point (standard for tensor values).
- `String`: UTF-8 character sequence.
- `Bool`: `true` or `false`.
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
- `SUM a`: Sum of all elements in the tensor.
- `MEAN a`: Arithmetic mean of all elements.
- `STDEV a`: Standard deviation of all elements.

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

- `FFT a`: Real-to-complex forward Fast Fourier Transform. `a` must be a rank-1 `Vector(N)`. Result is a `Matrix(2, N/2+1)` — **row 0 is the real part, row 1 is the imaginary part** of each frequency bin (only non-negative frequencies are computed, since a real input signal's spectrum is symmetric — this is the standard real-input FFT optimization, not a data loss).
- `IFFT a`: Complex-to-real inverse FFT. `a` must be a `Matrix(2, M)` spectrum (as `FFT` produces). Result is a real `Vector`. **Assumes the original signal length was even** (reconstructs length `2*(M-1)`) — the spectrum alone can't distinguish an even- from an odd-length source signal (both produce the same `M`), and there is currently no side-channel carrying the true length through the DSL layer. If you need an odd-length round trip, keep the original vector around rather than relying on `IFFT` to recover its exact length.
- `MAGNITUDE a`: Power/magnitude spectrum. `a` must be a `Matrix(2, M)` spectrum (as `FFT` produces). Result is a real `Vector(M)`, `sqrt(re² + im²)` per bin. The convenience most whitening/PSD-estimation work actually needs without touching phase.
- `PSD a WINDOW n`: Power spectral density (noise-floor) estimate via averaged periodograms. `a` must be a rank-1 `Vector` at least `n` samples long; splits it into non-overlapping `n`-sample chunks (any remainder that doesn't fill a full chunk is dropped), computes each chunk's power spectrum, and averages them elementwise. Result is a real `Vector(n/2+1)`. **Simplified vs. textbook Welch's method**: no 50% chunk overlap, and no window function (Hann/Hamming/etc.) applied before each chunk's FFT — good enough for noise-floor estimation, not a research-grade PSD estimator.
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

No new `Value`/`ValueType::Complex` variant exists to represent a complex
spectrum — it is an ordinary `Matrix(2, N)` value by convention, so every
existing `Matrix`-handling feature (`SHOW`, persistence, `TRANSPOSE`, row
indexing) already works on it unmodified. See `SIGNAL_PROCESSING_PLAN.md`
at the repo root for the full design history — this is the last operator
that plan calls for, though the plan may grow in future rounds.

---

## 4. Query & Engineering (SQL)

### Inline Vector Literals

Any SQL expression can contain an inline vector literal. The syntax mirrors Python list notation:

```sql
SELECT id, COSINE_SIM(embedding, [0.1, 0.2, 0.3]) AS score FROM docs
SELECT id, VEC_ADD(v, [0.0, 0.0, 1.0]) AS shifted FROM vecs
SELECT L2_NORM([3.0, 4.0]) AS five FROM dual
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

- **Aggregate Functions**: `SUM`, `AVG`, `COUNT`, `MIN`, `MAX`, `AVG_VEC`, `SUM_VEC`. A `SELECT` with an aggregate and no `GROUP BY` computes a single "global" aggregate row over the whole result set (e.g. `SELECT COUNT(*) FROM t`).
- **Filtering**: `WHERE` or `FILTER` can be used interchangeably.
- **`DISTINCT`**: `SELECT DISTINCT <cols> FROM ...` removes duplicate rows from the result.
- **`LIMIT`/`OFFSET`**: `LIMIT <n>` caps the row count; `OFFSET <n>` skips the first `n` rows before applying `LIMIT` (both may be used together or independently).

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

### Subqueries in FROM

A `SELECT`'s `FROM` clause can be another `SELECT`, wrapped in parentheses
and given an alias:

```sql
SELECT * FROM (SELECT id, price FROM items WHERE price > 5) AS cheap
```

`FROM (<SELECT>) AS <alias>` executes the inner query first and treats its
result as the outer query's source dataset, referenced by `<alias>`.

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
WITH recent AS (
    SELECT * FROM events WHERE ts > 100
)
SELECT * FROM recent WHERE user_id = 1

-- Multiple CTEs
WITH cte_a AS (SELECT * FROM t1), cte_b AS (SELECT * FROM t2)
SELECT * FROM cte_a

SELECT id FROM users_a
UNION
SELECT id FROM users_b

SELECT id FROM users_a
UNION ALL
SELECT id FROM users_b
```

- `WITH <name> AS (<SELECT>), ...` materializes each CTE as a temporary dataset (by that name) before the main query runs, then removes it once the statement completes — the name is not available in later statements. Avoid reusing the name of an existing real dataset for a CTE, since the CTE temporarily creates a dataset under that name for the duration of the statement.
- `UNION` deduplicates matching rows; `UNION ALL` keeps duplicates. `UNION`/`UNION ALL` clauses can be chained (`A UNION B UNION C`, three-way and beyond) — each right-hand side is itself a full `SELECT`, so chaining just recurses.

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
- `CAST(expr AS <type>)` — scalar target types: `INT`/`INTEGER`, `FLOAT`/`DOUBLE`, `TEXT`/`STRING`/`VARCHAR`, `BOOL`/`BOOLEAN`.
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
- `DERIVE target FROM expr`: Create a new resource with full automated lineage tracking.

### Schema Evolution

- `ALTER DATASET ds ADD COLUMN col: type [DEFAULT val]`
- `ALTER DATASET ds ADD COLUMN col = expression [LAZY]`
- `MATERIALIZE ds`: Physicalize all `LAZY` columns in a dataset.
- `SET DATASET ds [METADATA] key = "value"`: Attach an arbitrary string metadata key to a dataset (the `METADATA` keyword is optional). Retrieve it with `SHOW DATASET METADATA <name>` (§9).

---

## 5. Persistence & Ingestion

Load and save data across different formats.

- `USE DATASET FROM "path" [AS name] [FIELDS (name1, name2, ...)]`: Load external data (CSV, HDF5, Numpy, Zarr) into the current session as ephemeral tensors and a dataset view.
  - Automatically detects format from file extension (`.csv`, `.h5`, `.npy`, `.npz`, `.zarr`).
  - `FIELDS (...)`: explicitly pick which named columns/datasets/arrays to ingest, by exact name. Without it, a source that bundles fields of different shapes (e.g. an HDF5 file with a `(10, 64)` array and a `(7,)` array) keeps whichever fields share the first-encountered one's shape and silently-but-loudly skips the rest (reported as a warning). With `FIELDS`, only the named fields are read — a name that doesn't exist in the source, or a set of named fields that can't share one row count, is a hard error instead of a skip, since you've said exactly what you want.
- `IMPORT DATASET FROM "path" [AS name] [FIELDS (name1, name2, ...)]`: Load and normalize external data into a persistent LINAL Dataset Package.
  - Supports CSV, HDF5, Numpy, and Zarr. `FIELDS (...)` works the same way as for `USE DATASET FROM` above.
- `IMPORT CSV FROM "path" AS name`: (Legacy) Auto-infer schema and load CSV into a relational dataset.
- `EXPORT [CSV] name TO "path"`: Save dataset to CSV. The `CSV` keyword is optional — `EXPORT name TO "path"` behaves identically.
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

- **HDF5**: Ingests datasets from groups; flattens them into columns.
- **Numpy**: Supports `.npy` (single vector/matrix) and `.npz` (named collections).
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
```

- `CREATE INDEX [<name>] ON <dataset>(<column>)`: Build a standard lookup index on a scalar column.
- `CREATE VECTOR INDEX [<name>] ON <dataset>(<column>)`: Build an index-accelerated structure over a `Vector` column, enabling `SEARCH` and index-aware `COSINE_SIM` filtering in `WHERE` clauses.
- List existing indexes with `SHOW INDEXES [<dataset>]` (§9).

### SEARCH (Vector Similarity)

The modern form:

```sql
SEARCH docs ON embedding QUERY [0.9, 0.1, 0.0] LIMIT 10
SEARCH docs ON embedding QUERY my_query_tensor LIMIT 10 INTO results
```

- `SEARCH <dataset> ON <column> QUERY <[vector literal]|tensor_name> LIMIT <k> [INTO <target>]`
- Returns the top-`k` nearest rows by cosine similarity; `INTO <target>` materializes the results as a new dataset instead of returning them inline.

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

---

## 9. Diagnostics

### Resource Display

- `SHOW <name>`: Display contents of any resource — tensor, legacy dataset, or tensor-first dataset. Automatically materializes lazy tensors before displaying.
- `SHOW ALL` / `SHOW ALL TENSORS`: List all in-memory tensors with shapes and data.
- `SHOW ALL DATASETS`: List all legacy datasets with row/column counts.
- `SHOW DATABASES` / `SHOW ALL DATABASES`: List all database instances.
- `SHOW SCHEMA <dataset>`: Display column names and types for a legacy dataset.
- `SHOW SHAPE <name>`: Display only the shape dimensions of a tensor.
- `SHOW LINEAGE <name>`: Display the recursive computation graph that produced a tensor.
- `SHOW INDEXES [<dataset>]`: List all indexes; optionally filter to a specific dataset.

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
- `AUDIT DATASET <name>`: Perform a deep health check on the reference graph — detects dangling tensor references.
- `DELIVER <dataset> [TO '<path>']`: Check whether a dataset is deliverable over the `/delivery` HTTP routes (§10). Errors if the dataset doesn't exist. If it exists but hasn't been persisted yet, reports that and points to `SAVE DATASET`; if a delivery manifest is found (default path `<data_dir>/<db>/datasets/<name>/manifest.json`, or the directory given by `TO`), confirms it's ready to serve.

---

## 10. Server & Job Management

For remote execution and production workloads.

### Background Jobs

| Endpoint | Method | Description |
|---|---|---|
| `/jobs` | `POST` | Submit a DSL command for background execution. Returns `job_id`. |
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
| `/execute` | `POST` | Execute a DSL command synchronously. |
| `/databases` | `GET` | List database instances. |
| `/databases/:name` | `POST` | Create a database instance. |
| `/databases/:name` | `DELETE` | Drop a database instance. |
| `/delivery/...` | `GET` | Read-only dataset delivery endpoints. |

Multi-tenant isolation is provided via the `X-Linal-Database: <db_name>` request header. Each request restores the previous active database after execution, so concurrent requests with different headers do not interfere.

- **Graceful Shutdown**: Server handles `SIGINT`/`SIGTERM` to safely close connections.

---

**LINALDB**: *Where SQL meets Linear Algebra.*
Copyright (c) 2025 gorigami (gorigami.xyz)
Licensed under the LinalDB Community License v1.0
