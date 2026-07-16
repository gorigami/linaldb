# ⚡ LINALDB: The Tensor-First Analytical Engine

**LINALDB** is a high-performance, in-memory analytical engine built to bridge the gap between relational data engineering and scientific computing. It provides a SQL-inspired DSL that treats vectors, matrices, and multi-dimensional tensors as first-class citizens.

---

## One Mental Model: SQL meets Linear Algebra

LINALDB is designed for developers and researchers who need the structure of a database with the mathematical power of a tensor library.

- **Multi-Dimensional Tensors**: Generalized N-dimensional math (Rank > 2) with efficient offset traversal.
- **Tensor-SQL Bridge**: Inline vector literals and SQL vector functions (`COSINE_SIM`, `L2_NORM`, `NORMALIZE`, `DOT`, `VEC_ADD`, `VEC_SCALE`) usable directly inside SELECT, WHERE, and ORDER BY clauses.
- **Vector Aggregates**: `AVG_VEC` and `SUM_VEC` compute element-wise centroid or sum across row groups in a single GROUP BY query.
- **Lazy Evaluation Engine**: Define computation graphs using `LAZY LET` and materialize them on-demand via `SHOW`.
- **Numerical Aggregations**: Native `SUM`, `MEAN`, and `STDEV` operations for powerful statistical analysis.
- **Semantic Transformations**: Build zero-copy views using Reference Graphs and Lineage tracking.
- **Local-First & Portable**: Use it as an embedded library (like SQLite) or a multi-tenant managed server.
- **High-Performance Ingestion**: Native zero-copy ingestion for scientific data (CSV, HDF5, Numpy, Zarr) via the new connector-based architecture.
- **Dataset Delivery & Packages**: Standardized portable packages with Parquet data and JSON metadata (Schema, Stats, Lineage).
- **High Performance**: 2.5x speedup via SIMD, Rayon parallelization, and intelligent tensor pooling.

---

## 30-Second Quick Start

Get LINALDB running on your machine:

```bash
# Clone and build
git clone https://github.com/gorigami/linaldb.git
cd linaldb && cargo build --release

# 1. Start the interactive REPL
./target/release/linal repl

# 2. Run a smoke test script
./target/release/linal run examples/example.lnl

# 3. Start the managed server
./target/release/linal serve --port 8080
```

---

## Core Capabilities

### 1. Unified Hybrid Data Model

Store structured fields alongside high-dimensional tensors in the same dataset.

```sql
DATASET diagnostics COLUMNS (
    id: Int,
    region: String,
    features: Matrix(4, 4),  -- Native Matrix support
    embedding: Vector(128)   -- Native Vector support
)
```

### 2. Tensor-SQL Bridge

Vectors are first-class citizens in SQL expressions — no separate vector query language needed.

```sql
-- Similarity search with an inline query vector
SELECT id, title, COSINE_SIM(embedding, [0.9, 0.1, 0.0]) AS score
FROM docs
WHERE COSINE_SIM(embedding, [0.9, 0.1, 0.0]) > 0.7
ORDER BY score DESC
LIMIT 10

-- Compute per-category centroids with GROUP BY
SELECT category, AVG_VEC(embedding) AS centroid
FROM docs
GROUP BY category

-- Normalize before storing results
SELECT id, NORMALIZE(embedding) AS unit_vec FROM docs
```

### 3. Zero-Copy Reference Graphs

Create semantic views without duplicating data. LINALDB tracks lineage and provenance automatically.

```sql
-- Create a zero-copy alias
BIND scores_alias TO original_scores

-- Perform statistical analysis on high-rank data
LET total_norm = NORMALIZE sensor_3d
LET avg_signal = MEAN total_norm
LAZY LET trend = STDEV (sensor_3d * 1.5)

-- Derive new resources with full lineage
DERIVE clean_data FROM sensor_3d[0:10, :, *]
```

### 4. Named Pipelines with Persistence

Define reusable transformation chains and persist them across sessions.

```sql
-- Define a multi-step pipeline
DEFINE PIPELINE clean_rank AS
    WHERE active = 1
    THEN ORDER BY score DESC
    THEN LIMIT 10

-- Apply to a dataset
APPLY PIPELINE clean_rank ON products INTO top_products

-- Save to disk and restore later
SAVE PIPELINE clean_rank
-- ... restart session ...
LOAD PIPELINE clean_rank
APPLY PIPELINE clean_rank ON new_products INTO results
```

Pipelines are stored as human-readable JSON containing the original DSL source, making them portable and editable.

### 5. High-Concurrency Analytics

Multi-platform server with parallel execution and background workload management.

```bash
# Check server health
linal server status

# Submit a long-running job to the background
curl -X POST "http://localhost:8080/jobs" -d "SHOW ALL"
```

---

## 📖 Documentation Hub

LINALDB is extensively documented to help you scale from local experiments to production services.

- **[Architecture](docs/ARCHITECTURE.md)**: Deep dive into the internal engine design.
- **[Dataset Architecture](docs/DATASET_ARCHITECTURE.md)**: How the two dataset implementations (row-based and zero-copy reference graph) work and interoperate.
- **[DSL Reference](docs/DSL_REFERENCE.md)**: Complete guide to keywords, operators, and syntax.
- **[Examples](examples/README.md)**: Runnable `.lnl` scripts covering every major feature area.
- **[Error Reference](docs/ERROR_REFERENCE.md)**: Troubleshooting guide for engine and DSL errors.

---

## ⚖️ License

LINALDB is licensed under the **LinalDB Community License v1.0**.

- ✅ **Permitted**: Personal use, research, education, and internal organizational use.
- ⚠️ **Restricted**: Commercial redistribution, managed services (DBaaS/SaaS), and direct monetization require a separate **Commercial License**.

For commercial licensing inquiries, please contact: [develop@gorigami.xyz](mailto:develop@gorigami.xyz)

---

**LINALDB**: *Where SQL meets Linear Algebra.*
Copyright (c) 2025 gorigami (gorigami.xyz)
