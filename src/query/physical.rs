use crate::core::tuple::{Schema, Tuple};
use crate::engine::EngineError;
use crate::engine::TensorDb;
use std::sync::Arc;

/// Helper function to evaluate lazy columns in a row
pub(crate) fn evaluate_lazy_columns_in_row(
    dataset: &crate::core::dataset_legacy::Dataset,
    row: &Tuple,
) -> Result<Tuple, EngineError> {
    let mut evaluated_values = row.values.clone();

    // Evaluate any lazy columns
    for (i, field) in dataset.schema.fields.iter().enumerate() {
        if field.is_lazy && i < evaluated_values.len() {
            if let Some(evaluated_val) = dataset.evaluate_lazy_column(&field.name, row) {
                evaluated_values[i] = evaluated_val;
            }
        }
    }

    Tuple::new(dataset.schema.clone(), evaluated_values).map_err(EngineError::InvalidOp)
}

/// Trait for physical execution plan nodes
pub trait PhysicalPlan: Send + Sync + std::fmt::Debug {
    /// Get the schema of the output
    fn schema(&self) -> Arc<Schema>;

    /// Execute the plan and return the result rows
    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError>;
}

/// Sequential Scan Executor
#[derive(Debug)]
pub struct SeqScanExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
}

impl PhysicalPlan for SeqScanExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let dataset = db.get_dataset(&self.dataset_name)?;
        // Clone all rows and evaluate lazy columns
        let mut rows = Vec::with_capacity(dataset.rows.len());
        for row in &dataset.rows {
            rows.push(evaluate_lazy_columns_in_row(dataset, row)?);
        }
        Ok(rows)
    }
}

/// Produces rows computed earlier in the same query (`LogicalPlan::Values`:
/// a CTE or a `FROM` subquery) without reading the database at all.
#[derive(Debug)]
pub struct ValuesExec {
    pub schema: Arc<Schema>,
    pub rows: Arc<Vec<Tuple>>,
}

impl PhysicalPlan for ValuesExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, _db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        Ok(self.rows.as_ref().clone())
    }
}

/// Like `SeqScanExec`, but only reads the given `row_ranges` (start..end
/// index ranges into `Dataset.rows`) instead of every row. Produced by
/// `Planner::try_prune_partitions` when a range predicate's column has
/// per-partition zone-map stats (`Dataset.partitions`) that prove some
/// partitions can't contain a match. This only narrows which rows the
/// wrapping `FilterExec` has to evaluate -- it never decides the query's
/// answer, so correctness never depends on the pruning being exact (unlike
/// `IndexScanExec`, which replaces the filter outright).
#[derive(Debug)]
pub struct PartitionPrunedScanExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
    pub row_ranges: Vec<(usize, usize)>,
}

impl PhysicalPlan for PartitionPrunedScanExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let dataset = db.get_dataset(&self.dataset_name)?;
        let mut rows = Vec::new();
        for &(start, end) in &self.row_ranges {
            let end = end.min(dataset.rows.len());
            if start >= end {
                continue;
            }
            for row in &dataset.rows[start..end] {
                rows.push(evaluate_lazy_columns_in_row(dataset, row)?);
            }
        }
        Ok(rows)
    }
}

/// Filter Executor
pub struct FilterExec {
    pub input: Box<dyn PhysicalPlan>,
    pub predicate: Box<dyn Fn(&Tuple) -> bool + Send + Sync>,
}

impl std::fmt::Debug for FilterExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterExec")
            .field("input", &self.input)
            .field("predicate", &"<closure>")
            .finish()
    }
}

impl PhysicalPlan for FilterExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let input_rows = self.input.execute(db)?;
        crate::query::row_error::clear();
        let filtered = input_rows
            .into_iter()
            .filter(|row| (self.predicate)(row))
            .collect();
        if let Some(e) = crate::query::row_error::take() {
            return Err(EngineError::InvalidOp(e));
        }
        Ok(filtered)
    }
}

/// Index Scan Executor (Optimization)
#[derive(Debug)]
pub struct IndexScanExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
    pub column: String,
    pub value: crate::core::value::Value,
}

impl PhysicalPlan for IndexScanExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let dataset = db.get_dataset(&self.dataset_name)?;

        // Use Index!
        let index = dataset.get_index(&self.column).ok_or_else(|| {
            EngineError::InvalidOp(format!("Index not found on column '{}'", self.column))
        })?;

        let row_ids = index.lookup(&self.value).map_err(EngineError::InvalidOp)?;

        let mut evaluated_rows = Vec::new();
        for row in dataset.get_rows_by_ids(&row_ids) {
            evaluated_rows.push(evaluate_lazy_columns_in_row(dataset, &row)?);
        }
        Ok(evaluated_rows)
    }
}

/// Range scan through a `SORTED` index: the rows whose `column` satisfies
/// every `(op, value)` constraint, in row order. The planner wraps it in a
/// `FilterExec` with the full predicate, so this only has to narrow.
#[derive(Debug)]
pub struct SortedRangeScanExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
    pub column: String,
    pub constraints: Vec<(String, crate::core::value::Value)>,
}

impl PhysicalPlan for SortedRangeScanExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let dataset = db.get_dataset(&self.dataset_name)?;
        let index = dataset
            .get_index(&self.column)
            .and_then(|i| {
                i.as_any()
                    .downcast_ref::<crate::core::index::sorted::SortedIndex>()
            })
            .ok_or_else(|| {
                EngineError::InvalidOp(format!(
                    "SORTED index not found on column '{}'",
                    self.column
                ))
            })?;
        let row_ids = index.range(&self.constraints);
        let mut rows = Vec::with_capacity(row_ids.len());
        for row in dataset.get_rows_by_ids(&row_ids) {
            rows.push(evaluate_lazy_columns_in_row(dataset, &row)?);
        }
        Ok(rows)
    }
}

/// Vector Search Executor
#[derive(Debug)]
pub struct VectorSearchExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
    pub column: String,
    pub query: crate::core::tensor::Tensor,
    pub k: usize,
    /// Which index type the planner found on `column` at plan time (`None`
    /// if there wasn't one yet) -- purely for `EXPLAIN`'s benefit, so it can
    /// report whether HNSW, IVF, or no acceleration will actually be used
    /// without having to execute the query. `execute()` re-resolves the
    /// real index itself and doesn't trust this field.
    pub resolved_index_type: Option<String>,
}

impl PhysicalPlan for VectorSearchExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let dataset = db.get_dataset(&self.dataset_name)?;
        let index = dataset.get_index(&self.column).ok_or_else(|| {
            EngineError::InvalidOp(format!(
                "Vector index not found on column '{}'",
                self.column
            ))
        })?;

        // Top-k search is the one path an HNSW-backed index can safely
        // accelerate too (unlike the exact-predicate paths below, which stay
        // IVF-`Vector`-only -- see `core::index::hnsw::HnswIndex`'s doc
        // comment).
        if !matches!(
            index.index_type(),
            crate::core::index::IndexType::Vector | crate::core::index::IndexType::Hnsw
        ) {
            return Err(EngineError::InvalidOp(format!(
                "Index on '{}' is not a VECTOR index",
                self.column
            )));
        }

        let results = index
            .search(&self.query, self.k)
            .map_err(EngineError::InvalidOp)?;
        let row_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();

        let mut evaluated_rows = Vec::new();
        for row in dataset.get_rows_by_ids(&row_ids) {
            evaluated_rows.push(evaluate_lazy_columns_in_row(dataset, &row)?);
        }
        Ok(evaluated_rows)
    }
}

/// Top-k vector search for many queries at once (`SEARCH ... QUERIES`), and
/// any `SEARCH ... PREFILTER`. Without a prefilter, every query runs the
/// same `Index::search` a single-query `SEARCH` would, in parallel. With
/// one, each query's top-k is an exact cosine ranking over the rows that
/// pass the predicate (narrowed first through a SORTED index when the
/// predicate has a range it can answer), so `k` rows come back whenever `k`
/// rows pass. Rows come out grouped by query in input order, then by rank;
/// ties keep row order.
pub struct BatchVectorSearchExec {
    pub dataset_name: String,
    pub column: String,
    pub queries: Arc<crate::query::logical::QueryBatch>,
    pub k: usize,
    pub schema: Arc<Schema>,
    pub prefilter: Option<Arc<crate::query::logical::Prefilter>>,
    pub rows_only: bool,
    /// Same role as `VectorSearchExec::resolved_index_type`: for `EXPLAIN`.
    pub resolved_index_type: Option<String>,
}

// Hand-written so `EXPLAIN` prints the query count, not every query vector.
impl std::fmt::Debug for BatchVectorSearchExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefilter = self.prefilter.as_ref().map(|p| match &p.sorted_range {
            _ if p.approximate => "HNSW graph over rows passing PREFILTER (APPROX)".to_string(),
            Some((column, _)) => format!(
                "exact over rows passing PREFILTER, narrowed by SORTED index on {}",
                column
            ),
            None => "exact over rows passing PREFILTER, full scan".to_string(),
        });
        f.debug_struct("BatchVectorSearchExec")
            .field("dataset_name", &self.dataset_name)
            .field("column", &self.column)
            .field("queries", &self.queries.0.len())
            .field("k", &self.k)
            .field("resolved_index_type", &self.resolved_index_type)
            .field("prefilter", &prefilter)
            .finish()
    }
}

impl BatchVectorSearchExec {
    fn index_results(
        &self,
        dataset: &crate::core::dataset_legacy::Dataset,
    ) -> Result<Vec<Vec<(usize, f32)>>, EngineError> {
        use rayon::prelude::*;

        let index = dataset.get_index(&self.column).ok_or_else(|| {
            EngineError::InvalidOp(format!(
                "Vector index not found on column '{}'",
                self.column
            ))
        })?;
        if !matches!(
            index.index_type(),
            crate::core::index::IndexType::Vector | crate::core::index::IndexType::Hnsw
        ) {
            return Err(EngineError::InvalidOp(format!(
                "Index on '{}' is not a VECTOR index",
                self.column
            )));
        }
        self.queries
            .0
            .par_iter()
            .map(|(_, v)| {
                let id = crate::core::tensor::TensorId::new();
                let query = crate::core::tensor::Tensor::new(
                    id,
                    crate::core::tensor::Shape::new(vec![v.len()]),
                    v.clone(),
                    crate::core::tensor::TensorMetadata::new(id, None),
                )?;
                index.search(&query, self.k)
            })
            .collect::<Result<_, String>>()
            .map_err(EngineError::InvalidOp)
    }

    fn prefiltered_results(
        &self,
        dataset: &crate::core::dataset_legacy::Dataset,
        pf: &crate::query::logical::Prefilter,
    ) -> Result<Vec<Vec<(usize, f32)>>, EngineError> {
        use crate::core::index::flat::{cosine_with_norms, l2_norm};
        use crate::core::value::Value;
        use rayon::prelude::*;

        let col_idx = dataset
            .schema
            .get_field_index(&self.column)
            .ok_or_else(|| EngineError::InvalidOp(format!("column '{}' not found", self.column)))?;
        let width = dataset.schema.fields.len();
        let sorted = pf.sorted_range.as_ref().and_then(|(column, bounds)| {
            let index = dataset.get_index(column)?;
            let index = index
                .as_any()
                .downcast_ref::<crate::core::index::sorted::SortedIndex>()?;
            Some((index, bounds))
        });
        // Only the columns the predicate reads are copied into each
        // candidate's evaluation row.
        let mut referenced = Vec::new();
        collect_columns(&pf.predicate, &mut referenced);
        let copy_all = referenced.iter().any(|c| c == "\0all");
        let needed: Vec<bool> = (0..width)
            .map(|i| copy_all || referenced.contains(&dataset.schema.fields[i].name))
            .collect();
        let has_lazy = !dataset.lazy_expressions.is_empty();
        let hnsw = if pf.approximate {
            Some(
                dataset
                    .get_index(&self.column)
                    .and_then(|i| i.as_any().downcast_ref::<crate::core::index::hnsw::HnswIndex>())
                    .ok_or_else(|| {
                        EngineError::InvalidOp(format!(
                            "PREFILTER ... APPROX needs an HNSW index on '{}': CREATE VECTOR INDEX ON {}({}) USING HNSW",
                            self.column, self.dataset_name, self.column
                        ))
                    })?,
            )
        } else {
            None
        };

        self.queries
            .0
            .par_iter()
            .enumerate()
            .map(|(qi, (_, query))| {
                let qvals: &[Value] = pf.query_values.get(qi).map_or(&[], |v| v.as_slice());
                let mut eval_values = vec![Value::Null; width];
                eval_values.extend_from_slice(qvals);
                let mut eval_row = Tuple {
                    schema: pf.combined_schema.clone(),
                    values: eval_values,
                };

                let candidates: Vec<usize> = match &sorted {
                    Some((index, bounds)) => {
                        let mut constraints = Vec::with_capacity(bounds.len());
                        for (op, bound) in bounds.iter() {
                            let v = evaluate_expression(bound, &eval_row);
                            if v.is_null() {
                                return Ok(Vec::new()); // NULL bound: nothing passes
                            }
                            constraints.push((op.clone(), v));
                        }
                        index.range(&constraints)
                    }
                    None => (0..dataset.rows.len()).collect(),
                };

                let query_norm = l2_norm(query);
                let mut scored = Vec::new();
                let mut passing: Vec<usize> = Vec::new();
                for id in candidates {
                    let stored = &dataset.rows[id];
                    let evaluated;
                    let row = if has_lazy {
                        evaluated = evaluate_lazy_columns_in_row(dataset, stored)
                            .map_err(|e| e.to_string())?;
                        &evaluated
                    } else {
                        stored
                    };
                    for (i, keep) in needed.iter().enumerate() {
                        if *keep {
                            eval_row.values[i] = row.values[i].clone();
                        }
                    }
                    let passes =
                        crate::query::planner::evaluate_predicate(&pf.predicate, &eval_row);
                    if let Some(e) = crate::query::row_error::take() {
                        return Err(e);
                    }
                    if !passes {
                        continue;
                    }
                    if hnsw.is_some() {
                        passing.push(id);
                        continue;
                    }
                    match &row.values[col_idx] {
                        Value::Vector(v) => {
                            if v.len() != query.len() {
                                return Err(format!(
                                    "SEARCH: row {} has a {}-dimensional vector, the query has {}",
                                    id,
                                    v.len(),
                                    query.len()
                                ));
                            }
                            scored.push((id, cosine_with_norms(query, query_norm, v, l2_norm(v))));
                        }
                        Value::QVector(qv) => {
                            let v = qv.dequantize();
                            if v.len() != query.len() {
                                return Err(format!(
                                    "SEARCH: row {} has a {}-dimensional vector, the query has {}",
                                    id,
                                    v.len(),
                                    query.len()
                                ));
                            }
                            scored.push((id, cosine_with_norms(query, query_norm, &v, l2_norm(&v))));
                        }
                        Value::SparseVector(sv) => {
                            if sv.dim() != query.len() {
                                return Err(format!(
                                    "SEARCH: row {} has a {}-dimensional sparse vector, the query has {}",
                                    id,
                                    sv.dim(),
                                    query.len()
                                ));
                            }
                            let dot = sv.dot_dense(query);
                            let norm = sv.l2_norm();
                            let score = if norm == 0.0 || query_norm == 0.0 {
                                0.0
                            } else {
                                dot / (query_norm * norm)
                            };
                            scored.push((id, score));
                        }
                        Value::Null => {}
                        other => {
                            return Err(format!(
                                "SEARCH: column '{}' row {} is {:?}, not a Vector",
                                self.column,
                                id,
                                other.value_type()
                            ))
                        }
                    }
                }
                if let Some(index) = hnsw {
                    let mut allowed = vec![false; dataset.rows.len()];
                    for &id in &passing {
                        allowed[id] = true;
                    }
                    let allowed_fn = |row_id: usize| allowed.get(row_id).copied().unwrap_or(false);
                    return index.search_filtered(query, self.k, &allowed_fn, passing.len());
                }
                scored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                scored.truncate(self.k);
                Ok(scored)
            })
            .collect::<Result<_, String>>()
            .map_err(EngineError::InvalidOp)
    }
}

/// `(dot, ‖a‖, ‖b‖)` when at least one of `a`, `b` is a `SparseVector`
/// and the other a `SparseVector` or `Vector` of the same dimension; `None`
/// otherwise. A dimension mismatch is recorded in `row_error` (the plan-time
/// check catches it first whenever both dimensions are declared).
fn sparse_dot(
    a: &crate::core::value::Value,
    b: &crate::core::value::Value,
    name: &str,
) -> Option<(f32, f32, f32)> {
    use crate::core::value::Value;
    let dense_norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let (dim_a, dim_b) = match (a, b) {
        (Value::SparseVector(x), Value::SparseVector(y)) => (x.dim(), y.dim()),
        (Value::SparseVector(x), Value::Vector(y)) => (x.dim(), y.len()),
        (Value::Vector(x), Value::SparseVector(y)) => (x.len(), y.dim()),
        _ => return None,
    };
    if dim_a != dim_b {
        crate::query::row_error::record(format!(
            "{}: dimensions differ ({} vs {})",
            name, dim_a, dim_b
        ));
        return None;
    }
    Some(match (a, b) {
        (Value::SparseVector(x), Value::SparseVector(y)) => (x.dot(y), x.l2_norm(), y.l2_norm()),
        (Value::SparseVector(x), Value::Vector(y)) => (x.dot_dense(y), x.l2_norm(), dense_norm(y)),
        (Value::Vector(x), Value::SparseVector(y)) => (y.dot_dense(x), dense_norm(x), y.l2_norm()),
        _ => unreachable!(),
    })
}

/// `SPARSE(dim, [indices], [values])`. Bad input (non-integral or
/// out-of-range indices, unsorted or duplicate indices, non-finite values,
/// mismatched lengths) is recorded in `row_error`.
fn sparse_new(vals: &[crate::core::value::Value]) -> crate::core::value::Value {
    use crate::core::value::Value;
    if vals.iter().any(|v| v.is_null()) {
        return Value::Null;
    }
    let result = (|| -> Result<Value, String> {
        let dim = match vals.first() {
            Some(Value::Int(d)) if *d > 0 => *d as usize,
            _ => return Err("the first argument (dimension) must be a positive integer".into()),
        };
        let (Some(Value::Vector(idx)), Some(Value::Vector(v))) = (vals.get(1), vals.get(2)) else {
            return Err("expects SPARSE(dim, [indices], [values])".into());
        };
        let indices = idx
            .iter()
            .map(|x| {
                if x.fract() == 0.0 && *x >= 0.0 && *x <= u32::MAX as f32 {
                    Ok(*x as u32)
                } else {
                    Err(format!("index {} is not a non-negative integer", x))
                }
            })
            .collect::<Result<Vec<u32>, String>>()?;
        crate::core::sparse::SparseVec::new(dim, indices, v.clone()).map(Value::SparseVector)
    })();
    result.unwrap_or_else(|e| {
        crate::query::row_error::record(format!("SPARSE: {}", e));
        Value::Null
    })
}

/// `SPEC_COSINE` / `SPEC_COSINE_MOD` / `SPEC_MATCHES` (`core::spectral`).
/// A NULL argument gives NULL; bad data (unsorted m/z, a non-finite value,
/// a negative tolerance) is recorded in `row_error` so the statement fails
/// with it.
fn spectral_fn(
    func: crate::query::logical::VectorFnKind,
    vals: &[crate::core::value::Value],
) -> crate::core::value::Value {
    use crate::core::spectral::{cosine_greedy, peaks, Params};
    use crate::core::value::Value;
    use crate::query::logical::VectorFnKind;

    let name = match func {
        VectorFnKind::SpecCosine => "SPEC_COSINE",
        VectorFnKind::SpecCosineMod => "SPEC_COSINE_MOD",
        _ => "SPEC_MATCHES",
    };
    if vals.iter().any(|v| v.is_null()) {
        return Value::Null;
    }
    let num = |i: usize| -> Option<f64> {
        match vals.get(i)? {
            Value::Int(n) => Some(*n as f64),
            Value::Float(f) => Some(*f as f64),
            Value::Float64(f) => Some(*f),
            _ => None,
        }
    };
    let result = (|| -> Result<Value, String> {
        let (Some(Value::Matrix(a)), Some(Value::Matrix(b))) = (vals.first(), vals.get(1)) else {
            return Err("the first two arguments must be peak lists, Matrix(2, n)".to_string());
        };
        let a = peaks(a, "first spectrum")?;
        let b = peaks(b, "second spectrum")?;
        let tolerance = num(2).ok_or("tolerance must be a number")?;
        let (shift, powers_at) = match func {
            VectorFnKind::SpecCosineMod => (Some(num(3).ok_or("shift must be a number")?), 4),
            VectorFnKind::SpecMatches => (num(3), 4),
            _ => (None, 3),
        };
        let mz_power = if vals.len() > powers_at {
            num(powers_at).ok_or("mz_power must be a number")?
        } else {
            0.0
        };
        let intensity_power = if vals.len() > powers_at + 1 {
            num(powers_at + 1).ok_or("intensity_power must be a number")?
        } else {
            1.0
        };
        let params = Params::new(tolerance, mz_power, intensity_power)?;
        if let Some(s) = shift {
            if !s.is_finite() {
                return Err("shift must be finite".to_string());
            }
        }
        let (score, matched) = cosine_greedy(&a, &b, params, shift);
        Ok(match func {
            VectorFnKind::SpecMatches => Value::Int(matched as i64),
            _ => Value::Float64(score),
        })
    })();
    match result {
        Ok(v) => v,
        Err(e) => {
            crate::query::row_error::record(format!("{}: {}", name, e));
            Value::Null
        }
    }
}

/// Every column name `expr` reads.
fn collect_columns(expr: &crate::query::logical::Expr, out: &mut Vec<String>) {
    use crate::query::logical::Expr;
    match expr {
        Expr::Column(c) => out.push(c.clone()),
        Expr::Literal(_) => {}
        Expr::BinaryExpr { left, right, .. } | Expr::And(left, right) | Expr::Or(left, right) => {
            collect_columns(left, out);
            collect_columns(right, out);
        }
        other => {
            // Anything else: fall back to copying every column, which is
            // always correct (just slower).
            let _ = other;
            out.push("\0all".to_string());
        }
    }
}

impl PhysicalPlan for BatchVectorSearchExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let dataset = db.get_dataset(&self.dataset_name)?;
        let results = match &self.prefilter {
            Some(pf) => self.prefiltered_results(dataset, pf)?,
            None => self.index_results(dataset)?,
        };

        let mut out = Vec::new();
        for ((query_id, _), hits) in self.queries.0.iter().zip(results) {
            for (rank, (row_id, score)) in hits.into_iter().enumerate() {
                let row = dataset.rows.get(row_id).ok_or_else(|| {
                    EngineError::InvalidOp(format!(
                        "SEARCH: index returned row {} but '{}' has {} rows",
                        row_id,
                        self.dataset_name,
                        dataset.rows.len()
                    ))
                })?;
                let row = evaluate_lazy_columns_in_row(dataset, row)?;
                let values = if self.rows_only {
                    row.values
                } else {
                    let mut values = Vec::with_capacity(4 + row.values.len());
                    values.push(query_id.clone());
                    values.push(crate::core::value::Value::Int(rank as i64 + 1));
                    values.push(crate::core::value::Value::Float(score));
                    values.push(crate::core::value::Value::Int(row_id as i64));
                    values.extend(row.values);
                    values
                };
                out.push(Tuple::new(self.schema.clone(), values).map_err(EngineError::InvalidOp)?);
            }
        }
        Ok(out)
    }
}

/// Projection Executor
#[derive(Debug)]
pub struct ProjectionExec {
    pub input: Box<dyn PhysicalPlan>,
    pub output_schema: Arc<Schema>,
    pub column_indices: Vec<usize>,
}

impl PhysicalPlan for ProjectionExec {
    fn schema(&self) -> Arc<Schema> {
        self.output_schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let input_rows = self.input.execute(db)?;
        let mut output_rows = Vec::with_capacity(input_rows.len());

        for row in input_rows {
            let new_values: Vec<_> = self
                .column_indices
                .iter()
                .map(|&idx| row.values[idx].clone())
                .collect();
            output_rows.push(
                Tuple::new(self.output_schema.clone(), new_values)
                    .map_err(EngineError::InvalidOp)?,
            );
        }
        Ok(output_rows)
    }
}

/// Limit Executor
#[derive(Debug)]
pub struct LimitExec {
    pub input: Box<dyn PhysicalPlan>,
    pub n: usize,
    pub offset: usize,
}

impl PhysicalPlan for LimitExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let input_rows = self.input.execute(db)?;
        Ok(input_rows
            .into_iter()
            .skip(self.offset)
            .take(self.n)
            .collect())
    }
}

/// Sort Executor — supports multi-column sort with per-column direction.
#[derive(Debug)]
pub struct SortExec {
    pub input: Box<dyn PhysicalPlan>,
    pub columns: Vec<(String, bool)>,
}

impl PhysicalPlan for SortExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let rows = self.input.execute(db)?;
        let schema = self.schema();
        sort_tuples(rows, &schema, &self.columns)
    }
}

/// Shared by `SortExec` (ordering by a base column, baked into the
/// LogicalPlan and run before the query executes) and `execute_select`'s
/// post-processing pass (ordering by a `Computed`/`Window` alias, which
/// doesn't exist in any schema until *after* `apply_window_and_computed_exprs`
/// appends it — see that call site for why this had to be pulled out into a
/// standalone function rather than staying a `SortExec`-only method).
pub fn sort_tuples(
    rows: Vec<Tuple>,
    schema: &Schema,
    columns: &[(String, bool)],
) -> Result<Vec<Tuple>, EngineError> {
    // Pre-resolve column indices so the sort closure is allocation-free.
    let col_refs: Vec<(usize, bool)> = columns
        .iter()
        .map(|(col, asc)| {
            let idx = schema.get_field_index(col).ok_or_else(|| {
                EngineError::InvalidOp(format!("Column not found for sorting: {}", col))
            })?;
            match &schema.fields[idx].value_type {
                crate::core::value::ValueType::Vector(_)
                | crate::core::value::ValueType::Matrix(_, _) => {
                    Err(EngineError::InvalidOp(format!(
                        "Cannot ORDER BY column '{}': Vector and Matrix values have no defined ordering. \
                         Sort by a scalar expression instead (e.g. a similarity/distance function).",
                        col
                    )))
                }
                // Complex has real equality (Value::equals()) but, like
                // Vector/Matrix, no total order -- Value::compare() already
                // correctly returns None for it, but silently treating that
                // as "tied" here (the `unwrap_or(Equal)` below) would leave
                // rows in an arbitrary, not-actually-sorted order instead of
                // the same clear error Vector/Matrix already get. Caught by
                // this phase's proactive wildcard-arm audit.
                crate::core::value::ValueType::Complex => Err(EngineError::InvalidOp(format!(
                    "Cannot ORDER BY column '{}': Complex values have no defined ordering. \
                     Sort by REAL(...)/IMAG(...)/ABS(...) instead.",
                    col
                ))),
                _ => Ok((idx, *asc)),
            }
        })
        .collect::<Result<_, _>>()?;

    let mut sorted_rows = rows;
    sorted_rows.sort_by(|a, b| {
        for &(col_idx, asc) in &col_refs {
            let cmp = a.values[col_idx]
                .compare(&b.values[col_idx])
                .unwrap_or(std::cmp::Ordering::Equal);
            let ord = if asc { cmp } else { cmp.reverse() };
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });

    Ok(sorted_rows)
}

/// Aggregation Executor
#[derive(Debug)]
pub struct AggregateExec {
    pub input: Box<dyn PhysicalPlan>,
    pub group_expr: Vec<crate::query::logical::Expr>,
    pub aggr_expr: Vec<crate::query::logical::Expr>,
    pub schema: Arc<Schema>,
}

impl PhysicalPlan for AggregateExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let rows = self.input.execute(db)?;
        crate::query::row_error::clear();

        // If no rows and no group by, return empty result set
        // (Aggregations on empty sets typically return no rows, not NULL rows)
        if rows.is_empty() {
            return Ok(vec![]);
        }

        // If no group by, global aggregation (1 group)
        // If group by, hash aggregation

        use crate::core::value::Value;
        use std::collections::HashMap;

        // Map GroupKey -> Accumulators
        // GroupKey is Vec<Value>
        type GroupKey = Vec<Value>;
        type Accumulators = Vec<Value>; // Accumulator state for SUM, COUNT, MIN, MAX

        // Separate tracking for AVG: (sum, count) pairs for each AVG aggregate
        // Indexed by position in aggr_expr
        type AvgAccumulators = Vec<(Value, usize)>; // (sum, count) for AVG

        // VARIANCE: Welford's online algorithm (mean, M2, count) per group
        // per aggregate position -- a single pass, no need to retain raw
        // values (unlike MEDIAN below). Population variance = M2 / count.
        type VarianceAccumulators = Vec<(f64, f64, u64)>;

        // MEDIAN: no online algorithm computes an exact median, so this
        // collects every scalar value seen per group per aggregate
        // position, sorted at finalization time.
        type MedianAccumulators = Vec<Vec<f64>>;

        // ARG_MAX/ARG_MIN: the best `by` key seen so far per aggregate
        // position (Null until a non-NULL key arrives); the value selected
        // alongside it lives in the regular `Accumulators` slot.
        type ArgKeyAccumulators = Vec<Value>;

        let mut groups: HashMap<
            GroupKey,
            (
                Accumulators,
                AvgAccumulators,
                VarianceAccumulators,
                MedianAccumulators,
                ArgKeyAccumulators,
            ),
        > = HashMap::new();
        // Group keys in first-appearance order. `groups` is a HashMap, whose
        // iteration order is random per process, so emitting groups by
        // iterating it made GROUP BY's row order (and every content hash
        // derived from it, e.g. provenance) differ run to run. Emitting in
        // first-appearance order makes the result deterministic.
        let mut group_order: Vec<GroupKey> = Vec::new();

        // VARIANCE/MEDIAN are scalar-only (Int/Float/Float64) -- promotes
        // any of the three to `f64`, or `None` for a non-scalar `Value`
        // (Vector/Matrix/String/Bool/Null), which callers turn into a real
        // engine error rather than silently skipping the row.
        fn scalar_as_f64(v: &Value) -> Option<f64> {
            match v {
                Value::Int(i) => Some(*i as f64),
                Value::Float(f) => Some(*f as f64),
                Value::Float64(f) => Some(*f),
                _ => None,
            }
        }

        // 1. Initialize groups
        // Iterate rows
        for row in rows {
            // Eval group key
            let key: GroupKey = self
                .group_expr
                .iter()
                .map(|expr| evaluate_expression(expr, &row))
                .collect();

            if !groups.contains_key(&key) {
                group_order.push(key.clone());
            }
            let (accs, avg_accs, var_accs, median_accs, arg_keys) =
                groups.entry(key).or_insert_with(|| {
                    // Init accumulators
                    let mut regular_accs = Vec::new();
                    let mut avg_accumulators = Vec::new();
                    let mut var_accumulators: VarianceAccumulators = Vec::new();
                    let mut median_accumulators: MedianAccumulators = Vec::new();
                    // Every aggregate position gets a (normally unused) Null key,
                    // keeping all accumulator vectors index-aligned.
                    let arg_key_accumulators: ArgKeyAccumulators =
                        vec![Value::Null; self.aggr_expr.len()];

                    for expr in &self.aggr_expr {
                        match expr {
                            crate::query::logical::Expr::AggregateExpr {
                                func,
                                expr: inner,
                                ..
                            } => match func {
                                crate::query::logical::AggregateFunction::Count => {
                                    regular_accs.push(Value::Int(0));
                                    avg_accumulators.push((Value::Null, 0));
                                    var_accumulators.push((0.0, 0.0, 0));
                                    median_accumulators.push(Vec::new());
                                }
                                crate::query::logical::AggregateFunction::Sum
                                | crate::query::logical::AggregateFunction::SumVec => {
                                    let val = evaluate_expression(inner, &row);
                                    if let Value::Vector(v) = val {
                                        regular_accs.push(Value::Vector(vec![0.0; v.len()]));
                                    } else if let Value::Matrix(m) = val {
                                        if m.is_empty() {
                                            regular_accs.push(Value::Matrix(vec![]));
                                        } else {
                                            let r = m.len();
                                            let c = m[0].len();
                                            regular_accs.push(Value::Matrix(vec![vec![0.0; c]; r]));
                                        }
                                    } else if let Value::Complex(_) = val {
                                        regular_accs.push(Value::Complex(
                                            crate::core::value::Complex64::new(0.0, 0.0),
                                        ));
                                    } else {
                                        regular_accs.push(Value::Int(0));
                                    }
                                    avg_accumulators.push((Value::Null, 0));
                                    var_accumulators.push((0.0, 0.0, 0));
                                    median_accumulators.push(Vec::new());
                                }
                                crate::query::logical::AggregateFunction::Min => {
                                    regular_accs.push(Value::Null);
                                    avg_accumulators.push((Value::Null, 0));
                                    var_accumulators.push((0.0, 0.0, 0));
                                    median_accumulators.push(Vec::new());
                                }
                                crate::query::logical::AggregateFunction::Max => {
                                    regular_accs.push(Value::Null);
                                    avg_accumulators.push((Value::Null, 0));
                                    var_accumulators.push((0.0, 0.0, 0));
                                    median_accumulators.push(Vec::new());
                                }
                                crate::query::logical::AggregateFunction::Avg
                                | crate::query::logical::AggregateFunction::AvgVec => {
                                    let val = evaluate_expression(inner, &row);
                                    let initial_sum = if let Value::Vector(v) = val {
                                        Value::Vector(vec![0.0; v.len()])
                                    } else if let Value::Matrix(m) = val {
                                        if m.is_empty() {
                                            Value::Matrix(vec![])
                                        } else {
                                            let r = m.len();
                                            let c = m[0].len();
                                            Value::Matrix(vec![vec![0.0; c]; r])
                                        }
                                    } else if let Value::Float64(_) = val {
                                        Value::Float64(0.0)
                                    } else if let Value::Complex(_) = val {
                                        Value::Complex(crate::core::value::Complex64::new(0.0, 0.0))
                                    } else {
                                        Value::Float(0.0)
                                    };
                                    avg_accumulators.push((initial_sum, 0));
                                    regular_accs.push(Value::Null);
                                    var_accumulators.push((0.0, 0.0, 0));
                                    median_accumulators.push(Vec::new());
                                }
                                crate::query::logical::AggregateFunction::Variance => {
                                    regular_accs.push(Value::Null);
                                    avg_accumulators.push((Value::Null, 0));
                                    var_accumulators.push((0.0, 0.0, 0));
                                    median_accumulators.push(Vec::new());
                                }
                                crate::query::logical::AggregateFunction::Median
                                | crate::query::logical::AggregateFunction::ArgMax(_)
                                | crate::query::logical::AggregateFunction::ArgMin(_) => {
                                    regular_accs.push(Value::Null);
                                    avg_accumulators.push((Value::Null, 0));
                                    var_accumulators.push((0.0, 0.0, 0));
                                    median_accumulators.push(Vec::new());
                                }
                                crate::query::logical::AggregateFunction::Rrf(_) => {
                                    regular_accs.push(Value::Float64(0.0));
                                    avg_accumulators.push((Value::Null, 0));
                                    var_accumulators.push((0.0, 0.0, 0));
                                    median_accumulators.push(Vec::new());
                                }
                            },
                            _ => {
                                regular_accs.push(Value::Null);
                                avg_accumulators.push((Value::Null, 0));
                                var_accumulators.push((0.0, 0.0, 0));
                                median_accumulators.push(Vec::new());
                            }
                        }
                    }

                    (
                        regular_accs,
                        avg_accumulators,
                        var_accumulators,
                        median_accumulators,
                        arg_key_accumulators,
                    )
                });

            // Update accumulators
            for (i, expr) in self.aggr_expr.iter().enumerate() {
                if let crate::query::logical::Expr::AggregateExpr {
                    func,
                    expr: inner_expr,
                    ..
                } = expr
                {
                    // Eval inner expr
                    let val = evaluate_expression(inner_expr, &row);

                    match func {
                        crate::query::logical::AggregateFunction::Count => {
                            if let Value::Int(c) = accs[i] {
                                accs[i] = Value::Int(c + 1);
                            }
                        }
                        crate::query::logical::AggregateFunction::Sum
                        | crate::query::logical::AggregateFunction::SumVec => {
                            match (&mut accs[i], &val) {
                                (Value::Int(ref mut sum), Value::Int(v)) => *sum += v,
                                (Value::Float(ref mut sum), Value::Float(v)) => *sum += v,
                                (Value::Int(sum), Value::Float(v)) => {
                                    let new_val = *sum as f32 + v;
                                    accs[i] = Value::Float(new_val);
                                }
                                (Value::Float(ref mut sum), Value::Int(v)) => *sum += *v as f32,
                                // Any pairing touching Float64 promotes the
                                // accumulator to Float64, never demoting
                                // back once seen.
                                (Value::Float64(ref mut sum), Value::Float64(v)) => *sum += v,
                                (Value::Float64(ref mut sum), Value::Int(v)) => *sum += *v as f64,
                                (Value::Float64(ref mut sum), Value::Float(v)) => *sum += *v as f64,
                                (Value::Int(sum), Value::Float64(v)) => {
                                    let new_val = *sum as f64 + v;
                                    accs[i] = Value::Float64(new_val);
                                }
                                (Value::Float(sum), Value::Float64(v)) => {
                                    let new_val = *sum as f64 + v;
                                    accs[i] = Value::Float64(new_val);
                                }
                                // Complex is checked before the generic
                                // catch-all below (never silently dropped,
                                // per this phase's wildcard-arm audit): SUM
                                // of a Complex column is well-defined (unlike
                                // MIN/MAX), same promote-and-never-demote
                                // policy Float64 above already has.
                                (Value::Complex(ref mut sum), Value::Complex(v)) => *sum += v,
                                (Value::Complex(ref mut sum), v) => {
                                    if let Some(addend) = v.as_complex() {
                                        *sum += addend;
                                    }
                                }
                                (
                                    Value::Int(_) | Value::Float(_) | Value::Float64(_),
                                    Value::Complex(_),
                                ) => {
                                    let sum_so_far = accs[i]
                                        .as_complex()
                                        .unwrap_or(crate::core::value::Complex64::new(0.0, 0.0));
                                    let addend = val.as_complex().unwrap();
                                    accs[i] = Value::Complex(sum_so_far + addend);
                                }
                                (Value::Vector(sum_vec), Value::Vector(v)) => {
                                    if sum_vec.len() != v.len() {
                                        return Err(EngineError::InvalidOp(format!(
                                            "SUM: vector dimension mismatch in aggregate — expected {}, got {}",
                                            sum_vec.len(),
                                            v.len()
                                        )));
                                    }
                                    for (opt, val) in sum_vec.iter_mut().zip(v.iter()) {
                                        *opt += val;
                                    }
                                }
                                (Value::Matrix(sum_mat), Value::Matrix(v)) => {
                                    let expected_shape =
                                        (sum_mat.len(), sum_mat.first().map_or(0, |r| r.len()));
                                    let actual_shape = (v.len(), v.first().map_or(0, |r| r.len()));
                                    if expected_shape != actual_shape {
                                        return Err(EngineError::InvalidOp(format!(
                                            "SUM: matrix shape mismatch in aggregate — expected {:?}, got {:?}",
                                            expected_shape, actual_shape
                                        )));
                                    }
                                    for i in 0..sum_mat.len() {
                                        for j in 0..sum_mat[i].len() {
                                            sum_mat[i][j] += v[i][j];
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        crate::query::logical::AggregateFunction::Avg
                        | crate::query::logical::AggregateFunction::AvgVec => {
                            // Track sum and count for AVG
                            let (sum_ref, count_ref) = &mut avg_accs[i];
                            *count_ref += 1;

                            // Add to sum - need to handle type conversions
                            match sum_ref {
                                Value::Float(ref mut sum) => match &val {
                                    Value::Int(v) => *sum += *v as f32,
                                    Value::Float(v) => *sum += v,
                                    // Seeing a Float64 addend promotes the
                                    // whole accumulator to Float64, never
                                    // demoting back once promoted.
                                    Value::Float64(v) => {
                                        *sum_ref = Value::Float64(*sum as f64 + v);
                                    }
                                    Value::Complex(v) => {
                                        *sum_ref = Value::Complex(
                                            crate::core::value::Complex64::new(*sum as f64, 0.0)
                                                + v,
                                        );
                                    }
                                    _ => {}
                                },
                                Value::Float64(ref mut sum) => match &val {
                                    Value::Int(v) => *sum += *v as f64,
                                    Value::Float(v) => *sum += *v as f64,
                                    Value::Float64(v) => *sum += v,
                                    // Complex is checked before the generic
                                    // catch-all (never silently dropped, per
                                    // this phase's wildcard-arm audit) --
                                    // AVG of a Complex column is
                                    // well-defined, same promote-and-never-
                                    // demote policy as Float64 above.
                                    Value::Complex(v) => {
                                        *sum_ref = Value::Complex(
                                            crate::core::value::Complex64::new(*sum, 0.0) + v,
                                        );
                                    }
                                    _ => {}
                                },
                                Value::Complex(ref mut sum) => {
                                    if let Some(addend) = val.as_complex() {
                                        *sum += addend;
                                    }
                                }
                                Value::Int(ref mut sum) => {
                                    match &val {
                                        Value::Int(v) => {
                                            // Convert to Float for precision
                                            *sum_ref = Value::Float(*sum as f32 + *v as f32);
                                        }
                                        Value::Float(v) => {
                                            *sum_ref = Value::Float(*sum as f32 + v);
                                        }
                                        Value::Float64(v) => {
                                            *sum_ref = Value::Float64(*sum as f64 + v);
                                        }
                                        Value::Complex(v) => {
                                            *sum_ref = Value::Complex(
                                                crate::core::value::Complex64::new(
                                                    *sum as f64,
                                                    0.0,
                                                ) + v,
                                            );
                                        }
                                        _ => {}
                                    }
                                }
                                Value::Vector(ref mut sum_vec) => {
                                    if let Value::Vector(v) = &val {
                                        if sum_vec.len() != v.len() {
                                            return Err(EngineError::InvalidOp(format!(
                                                "AVG: vector dimension mismatch in aggregate — expected {}, got {}",
                                                sum_vec.len(),
                                                v.len()
                                            )));
                                        }
                                        for (s, val) in sum_vec.iter_mut().zip(v.iter()) {
                                            *s += val;
                                        }
                                    }
                                }
                                Value::Matrix(ref mut sum_mat) => {
                                    if let Value::Matrix(v) = &val {
                                        let expected_shape =
                                            (sum_mat.len(), sum_mat.first().map_or(0, |r| r.len()));
                                        let actual_shape =
                                            (v.len(), v.first().map_or(0, |r| r.len()));
                                        if expected_shape != actual_shape {
                                            return Err(EngineError::InvalidOp(format!(
                                                "AVG: matrix shape mismatch in aggregate — expected {:?}, got {:?}",
                                                expected_shape, actual_shape
                                            )));
                                        }
                                        // Element-wise sum
                                        for i in 0..sum_mat.len() {
                                            for j in 0..sum_mat[i].len() {
                                                sum_mat[i][j] += v[i][j];
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    // Initialize with first value
                                    *sum_ref = val.clone();
                                }
                            }
                        }
                        crate::query::logical::AggregateFunction::Max => {
                            match (&mut accs[i], &val) {
                                (Value::Null, _) => accs[i] = val.clone(),
                                (current, v) if !v.is_null() => {
                                    // Handle Vector element-wise MAX? Or Magnitude?
                                    // User said "element-wise aggregation".
                                    // MAX([1, 5], [2, 3]) -> [2, 5].
                                    match (current, v) {
                                        (Value::Vector(curr_vec), Value::Vector(v_vec)) => {
                                            if curr_vec.len() == v_vec.len() {
                                                for (c, n) in curr_vec.iter_mut().zip(v_vec.iter())
                                                {
                                                    if *n > *c {
                                                        *c = *n;
                                                    }
                                                }
                                            }
                                        }
                                        // Complex has no total order (see
                                        // Value::compare()'s doc comment) --
                                        // silently letting it fall to the
                                        // generic compare()-based arm below
                                        // would always report the *first*
                                        // row's value as "the max" (compare()
                                        // returns None, so the update never
                                        // fires), which looks like a real
                                        // answer but isn't one. Loud error
                                        // instead, same philosophy as every
                                        // other genuinely-undefined operation
                                        // in this codebase. Caught by this
                                        // phase's wildcard-arm audit.
                                        (Value::Complex(_), _) | (_, Value::Complex(_)) => {
                                            return Err(EngineError::InvalidOp(
                                                "MAX: Complex values have no defined ordering"
                                                    .to_string(),
                                            ));
                                        }
                                        (c, n) => {
                                            if let Some(std::cmp::Ordering::Greater) = n.compare(c)
                                            {
                                                *c = n.clone();
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        crate::query::logical::AggregateFunction::Min => {
                            match (&mut accs[i], &val) {
                                (Value::Null, _) => accs[i] = val.clone(),
                                (current, v) if !v.is_null() => match (current, v) {
                                    (Value::Vector(curr_vec), Value::Vector(v_vec)) => {
                                        if curr_vec.len() == v_vec.len() {
                                            for (c, n) in curr_vec.iter_mut().zip(v_vec.iter()) {
                                                if *n < *c {
                                                    *c = *n;
                                                }
                                            }
                                        }
                                    }
                                    // See the matching MAX comment above.
                                    (Value::Complex(_), _) | (_, Value::Complex(_)) => {
                                        return Err(EngineError::InvalidOp(
                                            "MIN: Complex values have no defined ordering"
                                                .to_string(),
                                        ));
                                    }
                                    (c, n) => {
                                        if let Some(std::cmp::Ordering::Less) = n.compare(c) {
                                            *c = n.clone();
                                        }
                                    }
                                },
                                _ => {}
                            }
                        }
                        crate::query::logical::AggregateFunction::Variance => {
                            let x = scalar_as_f64(&val).ok_or_else(|| {
                                EngineError::InvalidOp(format!(
                                    "VARIANCE: expected a scalar (Int/Float/Float64) column, got {:?}",
                                    val.value_type()
                                ))
                            })?;
                            // Welford's online algorithm.
                            let (mean, m2, count) = &mut var_accs[i];
                            *count += 1;
                            let delta = x - *mean;
                            *mean += delta / *count as f64;
                            let delta2 = x - *mean;
                            *m2 += delta * delta2;
                        }
                        crate::query::logical::AggregateFunction::Median => {
                            let x = scalar_as_f64(&val).ok_or_else(|| {
                                EngineError::InvalidOp(format!(
                                    "MEDIAN: expected a scalar (Int/Float/Float64) column, got {:?}",
                                    val.value_type()
                                ))
                            })?;
                            median_accs[i].push(x);
                        }
                        crate::query::logical::AggregateFunction::ArgMax(by)
                        | crate::query::logical::AggregateFunction::ArgMin(by) => {
                            let is_max =
                                matches!(func, crate::query::logical::AggregateFunction::ArgMax(_));
                            let name = if is_max { "ARG_MAX" } else { "ARG_MIN" };
                            let key = evaluate_expression(by, &row);
                            if key.is_null() {
                                continue;
                            }
                            if !matches!(
                                key,
                                Value::Int(_)
                                    | Value::Float(_)
                                    | Value::Float64(_)
                                    | Value::String(_)
                                    | Value::Bool(_)
                            ) {
                                return Err(EngineError::InvalidOp(format!(
                                    "{}: the `by` argument must be a scalar (Int/Float/Float64/String/Bool), got {:?}",
                                    name,
                                    key.value_type()
                                )));
                            }
                            if let Value::Float(f) = key {
                                if f.is_nan() {
                                    return Err(EngineError::InvalidOp(format!(
                                        "{}: the `by` argument is NaN, which has no ordering",
                                        name
                                    )));
                                }
                            }
                            if let Value::Float64(f) = key {
                                if f.is_nan() {
                                    return Err(EngineError::InvalidOp(format!(
                                        "{}: the `by` argument is NaN, which has no ordering",
                                        name
                                    )));
                                }
                            }
                            let replace = if arg_keys[i].is_null() {
                                true
                            } else {
                                // Strict comparison: an equal key never
                                // replaces, so the first row wins ties.
                                match key.compare(&arg_keys[i]) {
                                    Some(std::cmp::Ordering::Greater) => is_max,
                                    Some(std::cmp::Ordering::Less) => !is_max,
                                    Some(std::cmp::Ordering::Equal) => false,
                                    None => {
                                        return Err(EngineError::InvalidOp(format!(
                                            "{}: cannot compare `by` values of types {:?} and {:?}",
                                            name,
                                            key.value_type(),
                                            arg_keys[i].value_type()
                                        )))
                                    }
                                }
                            };
                            if replace {
                                arg_keys[i] = key;
                                accs[i] = val;
                            }
                        }
                        crate::query::logical::AggregateFunction::Rrf(k) => {
                            if val.is_null() {
                                continue;
                            }
                            let rank = scalar_as_f64(&val).ok_or_else(|| {
                                EngineError::InvalidOp(format!(
                                    "RRF: expected a numeric rank (Int/Float/Float64), got {:?}",
                                    val.value_type()
                                ))
                            })?;
                            let denom = k + rank;
                            if !denom.is_finite() || denom <= 0.0 {
                                return Err(EngineError::InvalidOp(format!(
                                    "RRF: k + rank must be positive and finite, got k={} rank={}",
                                    k, rank
                                )));
                            }
                            if let Value::Float64(sum) = &mut accs[i] {
                                *sum += 1.0 / denom;
                            }
                        }
                    }
                }
            }
        }

        if let Some(e) = crate::query::row_error::take() {
            return Err(EngineError::InvalidOp(e));
        }

        // Output rows - compute AVG/VARIANCE/MEDIAN from their accumulators
        // before outputting
        let mut output_rows = Vec::new();
        for key in group_order {
            let (accs, avg_accs, var_accs, median_accs, _arg_keys) = groups
                .remove(&key)
                .expect("every key in group_order was inserted into groups");
            let mut values = key; // Group keys first

            // Build final accumulator values, computing AVG/VARIANCE/MEDIAN where needed
            let mut final_accs = Vec::new();
            for (i, expr) in self.aggr_expr.iter().enumerate() {
                if let crate::query::logical::Expr::AggregateExpr { func, .. } = expr {
                    if matches!(
                        func,
                        crate::query::logical::AggregateFunction::Avg
                            | crate::query::logical::AggregateFunction::AvgVec
                    ) {
                        // Compute average: sum / count
                        let (sum, count) = &avg_accs[i];
                        if *count > 0 {
                            let avg = match sum {
                                Value::Float(s) => Value::Float(*s / *count as f32),
                                Value::Float64(s) => Value::Float64(*s / *count as f64),
                                Value::Int(s) => Value::Float(*s as f32 / *count as f32),
                                Value::Vector(v) => {
                                    Value::Vector(v.iter().map(|x| x / *count as f32).collect())
                                }
                                Value::Matrix(m) => Value::Matrix(
                                    m.iter()
                                        .map(|row| row.iter().map(|x| x / *count as f32).collect())
                                        .collect(),
                                ),
                                // Caught by this phase's wildcard-arm audit
                                // -- without this, AVG of a Complex column
                                // would finalize to Null despite the sum
                                // accumulator having correctly tracked a
                                // real Complex sum above.
                                Value::Complex(c) => Value::Complex(c / *count as f64),
                                _ => Value::Null,
                            };
                            final_accs.push(avg);
                        } else {
                            final_accs.push(Value::Null);
                        }
                    } else if matches!(func, crate::query::logical::AggregateFunction::Variance) {
                        let (_, m2, count) = &var_accs[i];
                        if *count > 0 {
                            final_accs.push(Value::Float64(m2 / *count as f64));
                        } else {
                            final_accs.push(Value::Null);
                        }
                    } else if matches!(func, crate::query::logical::AggregateFunction::Median) {
                        let mut sorted = median_accs[i].clone();
                        if sorted.is_empty() {
                            final_accs.push(Value::Null);
                        } else {
                            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let n = sorted.len();
                            let mid = n / 2;
                            let median = if n.is_multiple_of(2) {
                                (sorted[mid - 1] + sorted[mid]) / 2.0
                            } else {
                                sorted[mid]
                            };
                            final_accs.push(Value::Float64(median));
                        }
                    } else {
                        final_accs.push(accs[i].clone());
                    }
                } else {
                    final_accs.push(accs[i].clone());
                }
            }

            values.extend(final_accs); // Then aggregates
            output_rows
                .push(Tuple::new(self.schema.clone(), values).map_err(EngineError::InvalidOp)?);
        }

        Ok(output_rows)
    }
}

/// Cosine similarity threshold filter using a vector index.
#[derive(Debug)]
pub struct CosineFilterExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
    pub column: String,
    pub query: Vec<f32>,
    pub threshold: f32,
    pub strict: bool,
}

impl PhysicalPlan for CosineFilterExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        use crate::core::tensor::{Shape, TensorId, TensorMetadata};

        let dataset = db.get_dataset(&self.dataset_name)?;
        let index = dataset.get_index(&self.column).ok_or_else(|| {
            EngineError::InvalidOp(format!(
                "Vector index not found on column '{}'",
                self.column
            ))
        })?;

        if index.index_type() != crate::core::index::IndexType::Vector {
            return Err(EngineError::InvalidOp(format!(
                "Index on '{}' is not a VECTOR index",
                self.column
            )));
        }

        let n = self.query.len();
        let id = TensorId::new();
        let meta = TensorMetadata::new(id, None);
        let query_tensor =
            crate::core::tensor::Tensor::new(id, Shape::new(vec![n]), self.query.clone(), meta)
                .map_err(EngineError::InvalidOp)?;

        // `search_threshold` (not `search`) because this is a boolean
        // predicate, not a top-k ranking: it must return every row that
        // qualifies, not just however many an arbitrary `k` would keep. On a
        // clustered `VectorIndex` this also lets whole clusters be skipped
        // via a provable similarity bound, rather than scoring every row.
        let results = index
            .search_threshold(&query_tensor, self.threshold, self.strict)
            .map_err(EngineError::InvalidOp)?;

        let row_ids: Vec<usize> = results.into_iter().map(|(id, _)| id).collect();

        let mut evaluated_rows = Vec::new();
        for row in dataset.get_rows_by_ids(&row_ids) {
            evaluated_rows.push(evaluate_lazy_columns_in_row(dataset, &row)?);
        }
        Ok(evaluated_rows)
    }
}

pub fn evaluate_expression(
    expr: &crate::query::logical::Expr,
    row: &crate::core::tuple::Tuple,
) -> crate::core::value::Value {
    use crate::core::value::Value;
    match expr {
        // The one place a quantized column is turned back into f32 for
        // expressions; projections of the bare column keep the stored value.
        crate::query::logical::Expr::Column(name) => match row.get(name) {
            Some(Value::QVector(q)) => Value::Vector(q.dequantize()),
            Some(v) => v.clone(),
            None => Value::Null,
        },
        crate::query::logical::Expr::Literal(val) => val.clone(),
        crate::query::logical::Expr::And(l, r) => {
            match (evaluate_expression(l, row), evaluate_expression(r, row)) {
                (Value::Bool(a), Value::Bool(b)) => Value::Bool(a && b),
                _ => Value::Null,
            }
        }
        crate::query::logical::Expr::Or(l, r) => {
            match (evaluate_expression(l, row), evaluate_expression(r, row)) {
                (Value::Bool(a), Value::Bool(b)) => Value::Bool(a || b),
                _ => Value::Null,
            }
        }
        crate::query::logical::Expr::BinaryExpr { left, op, right } => {
            let left_val = evaluate_expression(left, row);
            let right_val = evaluate_expression(right, row);

            // Comparison operators apply generically via Value::compare(),
            // regardless of operand type — mirrors the separate WHERE-predicate
            // evaluator (query/planner.rs::evaluate_expr). Without this, every
            // comparison fell through the arithmetic-only type-pair match below
            // straight to `_ => Value::Null`, silently breaking any CASE WHEN
            // condition, computed SELECT column, or aggregate inner expression
            // that used a comparison (e.g. `CASE WHEN score > 90 THEN ...`) —
            // always taking the ELSE branch instead of erroring or evaluating
            // correctly. WHERE clauses were unaffected: they route through
            // `evaluate_expr`, not this function.
            if op == "=" || op == "!=" {
                // Value::equals() (not compare()'s Ordering) -- handles
                // Complex's real equality-without-order correctly; see its
                // doc comment. compare()-based Ordering is still exactly
                // right for every other type's = / != (its own cross-type
                // numeric/bool promotions are unchanged), equals() just
                // delegates to it for anything that isn't Complex.
                let eq = left_val.equals(&right_val);
                return Value::Bool(if op == "=" {
                    eq == Some(true)
                } else {
                    eq == Some(false)
                });
            }
            if matches!(op.as_str(), ">" | "<" | ">=" | "<=") {
                let ord = left_val.compare(&right_val);
                return Value::Bool(match op.as_str() {
                    ">" => ord == Some(std::cmp::Ordering::Greater),
                    "<" => ord == Some(std::cmp::Ordering::Less),
                    ">=" => matches!(
                        ord,
                        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
                    ),
                    _ => matches!(
                        ord,
                        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
                    ), // "<="
                });
            }

            // Any pairing touching Complex promotes to Complex (num_complex's
            // own +/-/*// impls, real numbers zero-extended via as_complex())
            // -- checked before the Float64 promotion below so it always
            // takes priority: Complex is strictly wider, Int/Float/Float64
            // all promote into it, never the reverse.
            if matches!(left_val, Value::Complex(_)) || matches!(right_val, Value::Complex(_)) {
                return match (left_val.as_complex(), right_val.as_complex()) {
                    (Some(l), Some(r)) => match op.as_str() {
                        "+" => Value::Complex(l + r),
                        "-" => Value::Complex(l - r),
                        "*" => Value::Complex(l * r),
                        "/" => Value::Complex(l / r),
                        _ => Value::Null,
                    },
                    _ => Value::Null,
                };
            }

            // Any pairing touching Float64 promotes to Float64 (widening the
            // other side at full f64 precision), checked before the
            // f32/Int-only arms below so it always takes priority over them
            // — mirrors the same policy in dsl/executor/query.rs's
            // eval_row_expr and window_running_sum.
            if matches!(left_val, Value::Float64(_)) || matches!(right_val, Value::Float64(_)) {
                return match (left_val.as_float64(), right_val.as_float64()) {
                    (Some(l), Some(r)) => match op.as_str() {
                        "+" => Value::Float64(l + r),
                        "-" => Value::Float64(l - r),
                        "*" => Value::Float64(l * r),
                        "/" => Value::Float64(l / r),
                        _ => Value::Null,
                    },
                    _ => Value::Null,
                };
            }

            match (left_val, right_val) {
                (Value::Int(l), Value::Int(r)) => match op.as_str() {
                    "+" => Value::Int(l + r),
                    "-" => Value::Int(l - r),
                    "*" => Value::Int(l * r),
                    "/" => {
                        if r != 0 {
                            Value::Int(l / r)
                        } else {
                            Value::Null
                        }
                    }
                    _ => Value::Null,
                },
                (Value::Float(l), Value::Float(r)) => match op.as_str() {
                    "+" => Value::Float(l + r),
                    "-" => Value::Float(l - r),
                    "*" => Value::Float(l * r),
                    "/" => Value::Float(l / r),
                    _ => Value::Null,
                },
                (Value::Int(l), Value::Float(r)) => {
                    let l = l as f32;
                    match op.as_str() {
                        "+" => Value::Float(l + r),
                        "-" => Value::Float(l - r),
                        "*" => Value::Float(l * r),
                        "/" => Value::Float(l / r),
                        _ => Value::Null,
                    }
                }
                (Value::Float(l), Value::Int(r)) => {
                    let r = r as f32;
                    match op.as_str() {
                        "+" => Value::Float(l + r),
                        "-" => Value::Float(l - r),
                        "*" => Value::Float(l * r),
                        "/" => Value::Float(l / r),
                        _ => Value::Null,
                    }
                }
                (Value::Matrix(l), Value::Matrix(r)) => {
                    // Element-wise ops
                    if l.len() != r.len() || (!l.is_empty() && l[0].len() != r[0].len()) {
                        return Value::Null; // Mismatch
                    }
                    let mut res = l.clone();
                    for i in 0..l.len() {
                        for j in 0..l[i].len() {
                            match op.as_str() {
                                "+" => res[i][j] += r[i][j],
                                "-" => res[i][j] -= r[i][j],
                                "*" => res[i][j] *= r[i][j], // Element-wise mul
                                "/" if r[i][j] != 0.0 => res[i][j] /= r[i][j],
                                _ => {}
                            }
                        }
                    }
                    Value::Matrix(res)
                }
                (Value::Matrix(m), Value::Int(scalar)) => {
                    let s = scalar as f32;
                    let mut res = m.clone();
                    for row in res.iter_mut() {
                        for val in row.iter_mut() {
                            match op.as_str() {
                                "+" => *val += s,
                                "-" => *val -= s,
                                "*" => *val *= s,
                                "/" if s != 0.0 => *val /= s,
                                _ => {}
                            }
                        }
                    }
                    Value::Matrix(res)
                }
                (Value::Matrix(m), Value::Float(scalar)) => {
                    let mut res = m.clone();
                    for row in res.iter_mut() {
                        for val in row.iter_mut() {
                            match op.as_str() {
                                "+" => *val += scalar,
                                "-" => *val -= scalar,
                                "*" => *val *= scalar,
                                "/" if scalar != 0.0 => *val /= scalar,
                                _ => {}
                            }
                        }
                    }
                    Value::Matrix(res)
                }
                _ => Value::Null,
            }
        }
        crate::query::logical::Expr::Not(inner) => match evaluate_expression(inner, row) {
            Value::Bool(b) => Value::Bool(!b),
            _ => Value::Null,
        },
        crate::query::logical::Expr::IsNull(inner) => match evaluate_expression(inner, row) {
            Value::Null => Value::Bool(true),
            _ => Value::Bool(false),
        },
        crate::query::logical::Expr::IsNotNull(inner) => match evaluate_expression(inner, row) {
            Value::Null => Value::Bool(false),
            _ => Value::Bool(true),
        },
        crate::query::logical::Expr::In { expr, list } => {
            let val = evaluate_expression(expr, row);
            let found = list.iter().any(|item| {
                val.compare(&evaluate_expression(item, row)) == Some(std::cmp::Ordering::Equal)
            });
            Value::Bool(found)
        }
        crate::query::logical::Expr::Between { expr, low, high } => {
            let val = evaluate_expression(expr, row);
            let lo = evaluate_expression(low, row);
            let hi = evaluate_expression(high, row);
            let ge = matches!(
                val.compare(&lo),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            );
            let le = matches!(
                val.compare(&hi),
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
            );
            Value::Bool(ge && le)
        }
        crate::query::logical::Expr::Case {
            operand,
            branches,
            else_expr,
        } => {
            let operand_val = operand.as_ref().map(|e| evaluate_expression(e, row));
            for (cond, result) in branches {
                let matched = if let Some(ref ov) = operand_val {
                    let cv = evaluate_expression(cond, row);
                    ov.compare(&cv) == Some(std::cmp::Ordering::Equal)
                } else {
                    matches!(evaluate_expression(cond, row), Value::Bool(true))
                };
                if matched {
                    return evaluate_expression(result, row);
                }
            }
            else_expr
                .as_ref()
                .map_or(Value::Null, |e| evaluate_expression(e, row))
        }
        crate::query::logical::Expr::Coalesce(args) => {
            for arg in args {
                let v = evaluate_expression(arg, row);
                if !v.is_null() {
                    return v;
                }
            }
            Value::Null
        }
        crate::query::logical::Expr::Nullif(a, b) => {
            let va = evaluate_expression(a, row);
            let vb = evaluate_expression(b, row);
            if va.compare(&vb) == Some(std::cmp::Ordering::Equal) {
                Value::Null
            } else {
                va
            }
        }
        crate::query::logical::Expr::ScalarFn { func, args } => {
            use crate::query::logical::ScalarFnKind;
            let vals: Vec<Value> = args.iter().map(|a| evaluate_expression(a, row)).collect();
            match func {
                ScalarFnKind::Upper => match vals.first() {
                    Some(Value::String(s)) => Value::String(s.to_uppercase()),
                    _ => Value::Null,
                },
                ScalarFnKind::Lower => match vals.first() {
                    Some(Value::String(s)) => Value::String(s.to_lowercase()),
                    _ => Value::Null,
                },
                ScalarFnKind::Length => match vals.first() {
                    Some(Value::String(s)) => Value::Int(s.len() as i64),
                    _ => Value::Null,
                },
                ScalarFnKind::Trim => match vals.first() {
                    Some(Value::String(s)) => Value::String(s.trim().to_string()),
                    _ => Value::Null,
                },
                ScalarFnKind::Concat => {
                    let parts: String = vals
                        .iter()
                        .filter_map(|v| {
                            if let Value::String(s) = v {
                                Some(s.as_str())
                            } else {
                                None
                            }
                        })
                        .collect();
                    Value::String(parts)
                }
                ScalarFnKind::Substr => {
                    if let (Some(Value::String(s)), Some(Value::Int(start))) =
                        (vals.first(), vals.get(1))
                    {
                        let start = (*start as usize).saturating_sub(1); // 1-based
                        if let Some(Value::Int(n)) = vals.get(2) {
                            Value::String(s.chars().skip(start).take(*n as usize).collect())
                        } else {
                            Value::String(s.chars().skip(start).collect())
                        }
                    } else {
                        Value::Null
                    }
                }
            }
        }
        crate::query::logical::Expr::Cast { expr, to } => {
            use crate::query::logical::CastTarget;
            let val = evaluate_expression(expr, row);
            match to {
                CastTarget::Int => match val {
                    Value::Int(n) => Value::Int(n),
                    Value::Float(f) => Value::Int(f as i64),
                    Value::Float64(f) => Value::Int(f as i64),
                    Value::String(s) => s.parse::<i64>().map(Value::Int).unwrap_or(Value::Null),
                    Value::Bool(b) => Value::Int(if b { 1 } else { 0 }),
                    _ => Value::Null,
                },
                CastTarget::Float => match val {
                    Value::Float(f) => Value::Float(f),
                    Value::Float64(f) => Value::Float(f as f32),
                    Value::Int(n) => Value::Float(n as f32),
                    Value::String(s) => s.parse::<f32>().map(Value::Float).unwrap_or(Value::Null),
                    _ => Value::Null,
                },
                // CAST(... AS DOUBLE) — parses/widens at full f64 precision,
                // the whole point of choosing this target over Float.
                CastTarget::Double => match val {
                    Value::Float64(f) => Value::Float64(f),
                    Value::Float(f) => Value::Float64(f as f64),
                    Value::Int(n) => Value::Float64(n as f64),
                    Value::String(s) => s.parse::<f64>().map(Value::Float64).unwrap_or(Value::Null),
                    _ => Value::Null,
                },
                CastTarget::Text => Value::String(match val {
                    Value::String(s) => s,
                    Value::Int(n) => n.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Float64(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::BitVector(b) => b.to_bit_string(),
                    Value::SparseVector(s) => s.to_string(),
                    _ => return Value::Null,
                }),
                CastTarget::Bool => match val {
                    Value::Bool(b) => Value::Bool(b),
                    Value::Int(n) => Value::Bool(n != 0),
                    Value::String(s) => Value::Bool(!s.is_empty()),
                    _ => Value::Null,
                },
                // Reshape/flatten between Vector and Matrix. Total element
                // count must match exactly — CAST doesn't resize or pad,
                // it only reinterprets the same data under a new shape.
                CastTarget::Vector(n) => match val {
                    Value::Vector(v) if v.len() == *n => Value::Vector(v),
                    Value::SparseVector(s) if s.dim() == *n => Value::Vector(s.to_dense()),
                    Value::BitVector(b) if b.len() == *n => Value::Vector(b.to_floats()),
                    Value::Matrix(m) => {
                        let flat: Vec<f32> = m.into_iter().flatten().collect();
                        if flat.len() == *n {
                            Value::Vector(flat)
                        } else {
                            Value::Null
                        }
                    }
                    _ => Value::Null,
                },
                CastTarget::Matrix(r, c) => match val {
                    Value::Matrix(m) if m.len() == *r && m.iter().all(|row| row.len() == *c) => {
                        Value::Matrix(m)
                    }
                    Value::Vector(v) if v.len() == r * c => {
                        let rows: Vec<Vec<f32>> =
                            v.chunks(*c).map(|chunk| chunk.to_vec()).collect();
                        Value::Matrix(rows)
                    }
                    _ => Value::Null,
                },
                CastTarget::QVector(n, e) => match val {
                    Value::Vector(v) if v.len() == *n => {
                        match crate::core::quant::QuantVec::quantize(&v, *e) {
                            Ok(q) => Value::QVector(q),
                            Err(err) => {
                                crate::query::row_error::record(format!(
                                    "CAST AS VECTOR({}, {}): {}",
                                    n, e, err
                                ));
                                Value::Null
                            }
                        }
                    }
                    Value::QVector(q) if q.len() == *n && q.encoding() == *e => Value::QVector(q),
                    _ => Value::Null,
                },
                CastTarget::SparseVector(n) => match val {
                    Value::SparseVector(s) if s.dim() == *n => Value::SparseVector(s),
                    Value::Vector(v) if v.len() == *n && v.iter().all(|x| x.is_finite()) => {
                        Value::SparseVector(crate::core::sparse::SparseVec::from_dense(&v))
                    }
                    _ => Value::Null,
                },
                CastTarget::BitVector(n) => {
                    let bits = match val {
                        Value::BitVector(b) => Some(b),
                        Value::String(s) => crate::core::bitvec::BitVec::from_bit_string(&s).ok(),
                        Value::Vector(v) if v.iter().all(|x| *x == 0.0 || *x == 1.0) => {
                            Some(crate::core::bitvec::BitVec::from_floats(&v))
                        }
                        _ => None,
                    };
                    match bits {
                        Some(b) if n.is_none_or(|n| b.len() == n) => Value::BitVector(b),
                        _ => Value::Null,
                    }
                }
            }
        }
        crate::query::logical::Expr::VecLiteral(vals) => {
            Value::Vector(vals.iter().map(|&v| v as f32).collect())
        }
        crate::query::logical::Expr::MatLiteral(rows) => Value::Matrix(
            rows.iter()
                .map(|r| r.iter().map(|&v| v as f32).collect())
                .collect(),
        ),
        crate::query::logical::Expr::VectorFn { func, args } => {
            use crate::query::logical::VectorFnKind;
            let vals: Vec<Value> = args.iter().map(|a| evaluate_expression(a, row)).collect();
            match func {
                VectorFnKind::Normalize => match vals.first() {
                    Some(Value::SparseVector(s)) => {
                        let norm = s.l2_norm();
                        if norm == 0.0 {
                            Value::SparseVector(s.clone())
                        } else {
                            Value::SparseVector(s.divide(norm))
                        }
                    }
                    Some(Value::Vector(v)) => {
                        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm == 0.0 {
                            Value::Vector(v.clone())
                        } else {
                            Value::Vector(v.iter().map(|x| x / norm).collect())
                        }
                    }
                    _ => Value::Null,
                },
                VectorFnKind::L2Norm => match vals.first() {
                    Some(Value::SparseVector(s)) => Value::Float(s.l2_norm()),
                    Some(Value::Vector(v)) => {
                        Value::Float(v.iter().map(|x| x * x).sum::<f32>().sqrt())
                    }
                    _ => Value::Null,
                },
                VectorFnKind::CosineSim => match (vals.first(), vals.get(1)) {
                    (Some(Value::Vector(a)), Some(Value::Vector(b))) => {
                        Value::Float(cosine_sim(a, b))
                    }
                    (Some(a), Some(b)) => match sparse_dot(a, b, "COSINE_SIM") {
                        Some((dot, na, nb)) => Value::Float(if na == 0.0 || nb == 0.0 {
                            0.0
                        } else {
                            dot / (na * nb)
                        }),
                        None => Value::Null,
                    },
                    _ => Value::Null,
                },
                VectorFnKind::Dot => match (vals.first(), vals.get(1)) {
                    (Some(Value::Vector(a)), Some(Value::Vector(b))) => {
                        Value::Float(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>())
                    }
                    (Some(a), Some(b)) => match sparse_dot(a, b, "DOT") {
                        Some((dot, _, _)) => Value::Float(dot),
                        None => Value::Null,
                    },
                    _ => Value::Null,
                },
                // A length mismatch between two typed BitVector columns is
                // rejected before execution (`query::typecheck`); one that
                // only shows up at runtime (CAST of a string column) gives
                // NULL here, since this evaluator has no error channel.
                VectorFnKind::Tanimoto | VectorFnKind::Jaccard => {
                    match (vals.first(), vals.get(1)) {
                        (Some(Value::BitVector(a)), Some(Value::BitVector(b))) => {
                            a.tanimoto(b).map(Value::Float64).unwrap_or(Value::Null)
                        }
                        _ => Value::Null,
                    }
                }
                VectorFnKind::Hamming => match (vals.first(), vals.get(1)) {
                    (Some(Value::BitVector(a)), Some(Value::BitVector(b))) => a
                        .hamming(b)
                        .map(|h| Value::Int(h as i64))
                        .unwrap_or(Value::Null),
                    _ => Value::Null,
                },
                VectorFnKind::BitCount => match vals.first() {
                    Some(Value::BitVector(a)) => Value::Int(a.count_ones() as i64),
                    _ => Value::Null,
                },
                VectorFnKind::SpecCosine
                | VectorFnKind::SpecCosineMod
                | VectorFnKind::SpecMatches => spectral_fn(*func, &vals),
                VectorFnKind::SparseNew => sparse_new(&vals),
                VectorFnKind::Distance => match (vals.first(), vals.get(1)) {
                    (Some(Value::Vector(a)), Some(Value::Vector(b))) => Value::Float(
                        a.iter()
                            .zip(b.iter())
                            .map(|(x, y)| (x - y).powi(2))
                            .sum::<f32>()
                            .sqrt(),
                    ),
                    _ => Value::Null,
                },
                VectorFnKind::VecAdd => match (vals.first(), vals.get(1)) {
                    (Some(Value::Vector(a)), Some(Value::Vector(b))) => {
                        Value::Vector(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
                    }
                    _ => Value::Null,
                },
                VectorFnKind::VecScale => match (vals.first(), vals.get(1)) {
                    (Some(Value::SparseVector(s)), Some(factor_val)) => match factor_val {
                        Value::Float(f) => Value::SparseVector(s.scale(*f)),
                        Value::Int(i) => Value::SparseVector(s.scale(*i as f32)),
                        _ => Value::Null,
                    },
                    (Some(Value::Vector(v)), Some(factor_val)) => {
                        let factor = match factor_val {
                            Value::Float(f) => *f,
                            Value::Int(i) => *i as f32,
                            _ => return Value::Null,
                        };
                        Value::Vector(v.iter().map(|x| x * factor).collect())
                    }
                    _ => Value::Null,
                },
                VectorFnKind::Matmul => match (vals.first(), vals.get(1)) {
                    (Some(Value::Matrix(a)), Some(Value::Matrix(b))) => {
                        if a.is_empty() || b.is_empty() || a[0].len() != b.len() {
                            return Value::Null;
                        }
                        let rows = a.len();
                        let cols = b[0].len();
                        let inner = b.len();
                        let mut result = vec![vec![0.0f32; cols]; rows];
                        for i in 0..rows {
                            for j in 0..cols {
                                for k in 0..inner {
                                    result[i][j] += a[i][k] * b[k][j];
                                }
                            }
                        }
                        Value::Matrix(result)
                    }
                    (Some(Value::Matrix(m)), Some(Value::Vector(v))) => {
                        if m.is_empty() || m[0].len() != v.len() {
                            return Value::Null;
                        }
                        let result: Vec<f32> = m
                            .iter()
                            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
                            .collect();
                        Value::Vector(result)
                    }
                    _ => Value::Null,
                },
                VectorFnKind::Transpose => match vals.first() {
                    Some(Value::Matrix(m)) => {
                        if m.is_empty() {
                            return Value::Matrix(vec![]);
                        }
                        let rows = m.len();
                        let cols = m[0].len();
                        let mut result = vec![vec![0.0f32; rows]; cols];
                        for i in 0..rows {
                            for j in 0..cols {
                                result[j][i] = m[i][j];
                            }
                        }
                        Value::Matrix(result)
                    }
                    _ => Value::Null,
                },
                VectorFnKind::MatShape => match vals.first() {
                    Some(Value::Matrix(m)) => {
                        let r = m.len();
                        let c = m.first().map_or(0, |row| row.len());
                        Value::String(format!("{}x{}", r, c))
                    }
                    Some(Value::Vector(v)) => Value::String(format!("{}x1", v.len())),
                    _ => Value::Null,
                },
                VectorFnKind::Flatten => match vals.first() {
                    Some(Value::Vector(v)) => Value::Vector(v.clone()),
                    Some(Value::Matrix(m)) => Value::Vector(m.iter().flatten().copied().collect()),
                    _ => Value::Null,
                },
                VectorFnKind::Real => match vals.first().and_then(|v| v.as_complex()) {
                    Some(c) => Value::Float64(c.re),
                    None => Value::Null,
                },
                VectorFnKind::Imag => match vals.first().and_then(|v| v.as_complex()) {
                    Some(c) => Value::Float64(c.im),
                    None => Value::Null,
                },
                VectorFnKind::ComplexAbs => match vals.first().and_then(|v| v.as_complex()) {
                    Some(c) => Value::Float64(c.norm()),
                    None => Value::Null,
                },
                VectorFnKind::Phase => match vals.first().and_then(|v| v.as_complex()) {
                    Some(c) => Value::Float64(c.arg()),
                    None => Value::Null,
                },
                VectorFnKind::Conj => match vals.first().and_then(|v| v.as_complex()) {
                    Some(c) => Value::Complex(c.conj()),
                    None => Value::Null,
                },
                VectorFnKind::ComplexNew => match (
                    vals.first().and_then(|v| v.as_float64()),
                    vals.get(1).and_then(|v| v.as_float64()),
                ) {
                    (Some(re), Some(im)) => {
                        Value::Complex(crate::core::value::Complex64::new(re, im))
                    }
                    _ => Value::Null,
                },
            }
        }
        _ => Value::Null,
    }
}

/// Returns `true` if the hash table should be built on the left side, based
/// purely on which materialized side has fewer rows (ties build right, to
/// minimize behavior change from before this side-selection existed).
fn build_side_is_left(left_len: usize, right_len: usize) -> bool {
    left_len < right_len
}

/// Hash join executor — INNER/LEFT/RIGHT/FULL join on an equi-condition.
/// Builds a hash table on whichever materialized side has fewer rows
/// (see `build_side_is_left`) and probes with the other side, so the cost
/// of building the table is always paid on the smaller relation regardless
/// of which side is syntactically left/right or which `JoinType` is used.
/// Which side must be NULL-padded for unmatched rows is a separate,
/// semantics-only decision driven by `JoinType`, fully decoupled from which
/// side happens to be the hash build side.
#[derive(Debug)]
pub struct HashJoinExec {
    pub left: Box<dyn PhysicalPlan>,
    pub right: Box<dyn PhysicalPlan>,
    pub left_col: String,
    pub right_col: String,
    pub join_type: crate::query::logical::JoinType,
    pub output_schema: Arc<Schema>,
}

impl PhysicalPlan for HashJoinExec {
    fn schema(&self) -> Arc<Schema> {
        self.output_schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        use crate::core::value::Value;
        use crate::query::logical::JoinType;
        use std::collections::{HashMap, HashSet};

        let left_rows = self.left.execute(db)?;
        let right_rows = self.right.execute(db)?;
        let out_schema = self.output_schema.clone();

        let left_schema = self.left.schema();
        let right_schema = self.right.schema();

        let left_col_idx = left_schema.get_field_index(&self.left_col).ok_or_else(|| {
            EngineError::InvalidOp(format!(
                "Join column '{}' not found in left dataset",
                self.left_col
            ))
        })?;
        let right_col_idx = right_schema
            .get_field_index(&self.right_col)
            .ok_or_else(|| {
                EngineError::InvalidOp(format!(
                    "Join column '{}' not found in right dataset",
                    self.right_col
                ))
            })?;

        let left_nulls: Vec<Value> = left_schema.fields.iter().map(|_| Value::Null).collect();
        let right_nulls: Vec<Value> = right_schema.fields.iter().map(|_| Value::Null).collect();

        let preserve_left = matches!(self.join_type, JoinType::Left | JoinType::Full);
        let preserve_right = matches!(self.join_type, JoinType::Right | JoinType::Full);

        let mut output = Vec::new();

        if build_side_is_left(left_rows.len(), right_rows.len()) {
            // Build on LEFT (smaller), probe with RIGHT.
            let mut build_map: HashMap<Value, Vec<usize>> = HashMap::new();
            for (i, row) in left_rows.iter().enumerate() {
                build_map
                    .entry(row.values[left_col_idx].clone())
                    .or_default()
                    .push(i);
            }

            let mut matched_build: HashSet<usize> = HashSet::new();

            for right_row in &right_rows {
                match build_map.get(&right_row.values[right_col_idx]) {
                    Some(build_indices) => {
                        for &li in build_indices {
                            let combined = merge_row_values(
                                &left_rows[li],
                                right_row,
                                &left_schema,
                                &right_schema,
                            );
                            output.push(
                                Tuple::new(out_schema.clone(), combined)
                                    .map_err(EngineError::InvalidOp)?,
                            );
                            if preserve_left {
                                matched_build.insert(li);
                            }
                        }
                    }
                    None if preserve_right => {
                        let mut combined = left_nulls.clone();
                        combined.extend_from_slice(&right_row.values);
                        output.push(
                            Tuple::new(out_schema.clone(), combined)
                                .map_err(EngineError::InvalidOp)?,
                        );
                    }
                    None => {}
                }
            }

            if preserve_left {
                for (i, left_row) in left_rows.iter().enumerate() {
                    if !matched_build.contains(&i) {
                        let mut combined = left_row.values.clone();
                        combined.extend_from_slice(&right_nulls);
                        output.push(
                            Tuple::new(out_schema.clone(), combined)
                                .map_err(EngineError::InvalidOp)?,
                        );
                    }
                }
            }
        } else {
            // Build on RIGHT (smaller-or-tied), probe with LEFT.
            let mut build_map: HashMap<Value, Vec<usize>> = HashMap::new();
            for (i, row) in right_rows.iter().enumerate() {
                build_map
                    .entry(row.values[right_col_idx].clone())
                    .or_default()
                    .push(i);
            }

            let mut matched_build: HashSet<usize> = HashSet::new();

            for left_row in &left_rows {
                match build_map.get(&left_row.values[left_col_idx]) {
                    Some(build_indices) => {
                        for &ri in build_indices {
                            let combined = merge_row_values(
                                left_row,
                                &right_rows[ri],
                                &left_schema,
                                &right_schema,
                            );
                            output.push(
                                Tuple::new(out_schema.clone(), combined)
                                    .map_err(EngineError::InvalidOp)?,
                            );
                            if preserve_right {
                                matched_build.insert(ri);
                            }
                        }
                    }
                    None if preserve_left => {
                        let mut combined = left_row.values.clone();
                        combined.extend_from_slice(&right_nulls);
                        output.push(
                            Tuple::new(out_schema.clone(), combined)
                                .map_err(EngineError::InvalidOp)?,
                        );
                    }
                    None => {}
                }
            }

            if preserve_right {
                for (i, right_row) in right_rows.iter().enumerate() {
                    if !matched_build.contains(&i) {
                        let mut combined = left_nulls.clone();
                        combined.extend_from_slice(&right_row.values);
                        output.push(
                            Tuple::new(out_schema.clone(), combined)
                                .map_err(EngineError::InvalidOp)?,
                        );
                    }
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod hash_join_build_side_tests {
    use super::build_side_is_left;

    #[test]
    fn smaller_left_builds_left() {
        assert!(build_side_is_left(2, 10));
    }

    #[test]
    fn smaller_right_builds_right() {
        assert!(!build_side_is_left(10, 2));
    }

    #[test]
    fn tie_builds_right() {
        assert!(!build_side_is_left(5, 5));
    }
}

/// Nested-loop similarity join — INNER/LEFT/RIGHT/FULL join where the
/// condition is `COSINE_SIM(left_col, right_col) > threshold` instead of
/// equality. Uses a `Vector` index on the right dataset's `right_col`
/// (if one exists) to accelerate each left row's match search via
/// `Index::search`, the same index-or-fallback pattern as
/// `CosineFilterExec`/`IndexScanExec` (see ARCHITECTURE.md's "Index-Aware
/// Execution" and `Planner::try_optimize_filter`); falls back to a
/// brute-force O(n·m) comparison when no matching index exists.
#[derive(Debug)]
pub struct SimilarityJoinExec {
    pub left: Box<dyn PhysicalPlan>,
    pub right: Box<dyn PhysicalPlan>,
    pub left_col: String,
    pub right_col: String,
    /// Needed to look up a `Vector` index on `right_col` for the
    /// index-accelerated path.
    pub right_dataset_name: String,
    pub threshold: f32,
    pub join_type: crate::query::logical::JoinType,
    pub output_schema: Arc<Schema>,
}

impl SimilarityJoinExec {
    /// For each left row, the right-row indices where
    /// `COSINE_SIM(left, right) > threshold`.
    fn compute_matches(
        &self,
        db: &TensorDb,
        left_rows: &[Tuple],
        right_rows: &[Tuple],
        left_col_idx: usize,
        right_col_idx: usize,
    ) -> Result<Vec<Vec<usize>>, EngineError> {
        let vector_index = db
            .get_dataset(&self.right_dataset_name)
            .ok()
            .and_then(|ds| ds.get_index(&self.right_col))
            .filter(|idx| idx.index_type() == crate::core::index::IndexType::Vector);

        if let Some(index) = vector_index {
            use crate::core::tensor::{Shape, TensorId, TensorMetadata};
            let k = right_rows.len().max(1);

            left_rows
                .iter()
                .map(|left_row| {
                    let crate::core::value::Value::Vector(lv) = &left_row.values[left_col_idx]
                    else {
                        return Ok(vec![]);
                    };
                    let n = lv.len();
                    let id = TensorId::new();
                    let meta = TensorMetadata::new(id, None);
                    let query_tensor =
                        crate::core::tensor::Tensor::new(id, Shape::new(vec![n]), lv.clone(), meta)
                            .map_err(EngineError::InvalidOp)?;
                    let results = index
                        .search(&query_tensor, k)
                        .map_err(EngineError::InvalidOp)?;
                    Ok(results
                        .into_iter()
                        .filter(|(_, score)| *score > self.threshold)
                        .map(|(id, _)| id)
                        .collect())
                })
                .collect()
        } else {
            Ok(left_rows
                .iter()
                .map(|left_row| {
                    let crate::core::value::Value::Vector(lv) = &left_row.values[left_col_idx]
                    else {
                        return vec![];
                    };
                    right_rows
                        .iter()
                        .enumerate()
                        .filter_map(|(ri, right_row)| match &right_row.values[right_col_idx] {
                            crate::core::value::Value::Vector(rv) if rv.len() == lv.len() => {
                                (cosine_sim(lv, rv) > self.threshold).then_some(ri)
                            }
                            _ => None,
                        })
                        .collect()
                })
                .collect())
        }
    }
}

impl PhysicalPlan for SimilarityJoinExec {
    fn schema(&self) -> Arc<Schema> {
        self.output_schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        use crate::query::logical::JoinType;
        use std::collections::{HashMap, HashSet};

        let left_rows = self.left.execute(db)?;
        let right_rows = self.right.execute(db)?;
        let out_schema = self.output_schema.clone();
        let left_schema = self.left.schema();
        let right_schema = self.right.schema();

        let left_col_idx = left_schema.get_field_index(&self.left_col).ok_or_else(|| {
            EngineError::InvalidOp(format!(
                "Join column '{}' not found in left dataset",
                self.left_col
            ))
        })?;
        let right_col_idx = right_schema
            .get_field_index(&self.right_col)
            .ok_or_else(|| {
                EngineError::InvalidOp(format!(
                    "Join column '{}' not found in right dataset",
                    self.right_col
                ))
            })?;

        let left_nulls: Vec<crate::core::value::Value> = left_schema
            .fields
            .iter()
            .map(|_| crate::core::value::Value::Null)
            .collect();
        let right_nulls: Vec<crate::core::value::Value> = right_schema
            .fields
            .iter()
            .map(|_| crate::core::value::Value::Null)
            .collect();

        let matches =
            self.compute_matches(db, &left_rows, &right_rows, left_col_idx, right_col_idx)?;

        let mut output = Vec::new();
        let mut matched_right: HashSet<usize> = HashSet::new();

        match self.join_type {
            JoinType::Inner | JoinType::Left | JoinType::Full => {
                for (li, left_row) in left_rows.iter().enumerate() {
                    let right_indices = &matches[li];
                    if right_indices.is_empty() {
                        if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                            let mut combined = left_row.values.clone();
                            combined.extend_from_slice(&right_nulls);
                            output.push(
                                Tuple::new(out_schema.clone(), combined)
                                    .map_err(EngineError::InvalidOp)?,
                            );
                        }
                    } else {
                        for &ri in right_indices {
                            let combined = merge_row_values(
                                left_row,
                                &right_rows[ri],
                                &left_schema,
                                &right_schema,
                            );
                            output.push(
                                Tuple::new(out_schema.clone(), combined)
                                    .map_err(EngineError::InvalidOp)?,
                            );
                            if self.join_type == JoinType::Full {
                                matched_right.insert(ri);
                            }
                        }
                    }
                }
                if self.join_type == JoinType::Full {
                    for (ri, right_row) in right_rows.iter().enumerate() {
                        if !matched_right.contains(&ri) {
                            let mut combined = left_nulls.clone();
                            combined.extend_from_slice(&right_row.values);
                            output.push(
                                Tuple::new(out_schema.clone(), combined)
                                    .map_err(EngineError::InvalidOp)?,
                            );
                        }
                    }
                }
            }
            JoinType::Right => {
                let mut right_to_left: HashMap<usize, Vec<usize>> = HashMap::new();
                for (li, right_indices) in matches.iter().enumerate() {
                    for &ri in right_indices {
                        right_to_left.entry(ri).or_default().push(li);
                    }
                }
                for (ri, right_row) in right_rows.iter().enumerate() {
                    match right_to_left.get(&ri) {
                        Some(left_indices) => {
                            for &li in left_indices {
                                let combined = merge_row_values(
                                    &left_rows[li],
                                    right_row,
                                    &left_schema,
                                    &right_schema,
                                );
                                output.push(
                                    Tuple::new(out_schema.clone(), combined)
                                        .map_err(EngineError::InvalidOp)?,
                                );
                            }
                        }
                        None => {
                            let mut combined = left_nulls.clone();
                            combined.extend_from_slice(&right_row.values);
                            output.push(
                                Tuple::new(out_schema.clone(), combined)
                                    .map_err(EngineError::InvalidOp)?,
                            );
                        }
                    }
                }
            }
        }

        Ok(output)
    }
}

/// UNION / UNION ALL Executor
#[derive(Debug)]
pub struct UnionExec {
    pub left: Box<dyn PhysicalPlan>,
    pub right: Box<dyn PhysicalPlan>,
    pub all: bool,
}

impl PhysicalPlan for UnionExec {
    fn schema(&self) -> Arc<Schema> {
        self.left.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let mut rows = self.left.execute(db)?;
        rows.extend(self.right.execute(db)?);
        if !self.all {
            let mut seen = std::collections::HashSet::new();
            rows.retain(|row| {
                let key: String = row
                    .values
                    .iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join("|");
                seen.insert(key)
            });
        }
        Ok(rows)
    }
}

/// DISTINCT Executor — removes duplicate rows
#[derive(Debug)]
pub struct DistinctExec {
    pub input: Box<dyn PhysicalPlan>,
}

impl PhysicalPlan for DistinctExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let rows = self.input.execute(db)?;
        let mut seen = std::collections::HashSet::new();
        let deduped = rows
            .into_iter()
            .filter(|row| {
                let key: String = row
                    .values
                    .iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join("|");
                seen.insert(key)
            })
            .collect();
        Ok(deduped)
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn merge_row_values(
    left: &Tuple,
    right: &Tuple,
    left_schema: &Schema,
    right_schema: &Schema,
) -> Vec<crate::core::value::Value> {
    let left_names: std::collections::HashSet<&str> =
        left_schema.fields.iter().map(|f| f.name.as_str()).collect();
    let mut values = left.values.clone();
    for (field, val) in right_schema.fields.iter().zip(right.values.iter()) {
        // Use the same collision-renaming logic as the logical schema
        if left_names.contains(field.name.as_str()) {
            values.push(val.clone()); // renamed in schema as `r_<col>`
        } else {
            values.push(val.clone());
        }
    }
    values
}
