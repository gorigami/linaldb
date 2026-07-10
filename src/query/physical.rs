use crate::core::tuple::{Schema, Tuple};
use crate::engine::EngineError;
use crate::engine::TensorDb;
use std::sync::Arc;

/// Helper function to evaluate lazy columns in a row
fn evaluate_lazy_columns_in_row(
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
        let filtered = input_rows
            .into_iter()
            .filter(|row| (self.predicate)(row))
            .collect();
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

/// Vector Search Executor
#[derive(Debug)]
pub struct VectorSearchExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
    pub column: String,
    pub query: crate::core::tensor::Tensor,
    pub k: usize,
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

        if index.index_type() != crate::core::index::IndexType::Vector {
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

        // Pre-resolve column indices so the sort closure is allocation-free.
        let col_refs: Vec<(usize, bool)> = self
            .columns
            .iter()
            .map(|(col, asc)| {
                schema
                    .get_field_index(col)
                    .ok_or_else(|| {
                        EngineError::InvalidOp(format!("Column not found for sorting: {}", col))
                    })
                    .map(|idx| (idx, *asc))
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

        let mut groups: HashMap<GroupKey, (Accumulators, AvgAccumulators)> = HashMap::new();

        // 1. Initialize groups
        // Iterate rows
        for row in rows {
            // Eval group key
            let key: GroupKey = self
                .group_expr
                .iter()
                .map(|expr| evaluate_expression(expr, &row))
                .collect();

            let (accs, avg_accs) = groups.entry(key).or_insert_with(|| {
                // Init accumulators
                let mut regular_accs = Vec::new();
                let mut avg_accumulators = Vec::new();

                for expr in &self.aggr_expr {
                    match expr {
                        crate::query::logical::Expr::AggregateExpr { func, expr: inner } => {
                            match func {
                                crate::query::logical::AggregateFunction::Count => {
                                    regular_accs.push(Value::Int(0));
                                    avg_accumulators.push((Value::Null, 0));
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
                                    } else {
                                        regular_accs.push(Value::Int(0));
                                    }
                                    avg_accumulators.push((Value::Null, 0));
                                }
                                crate::query::logical::AggregateFunction::Min => {
                                    regular_accs.push(Value::Null);
                                    avg_accumulators.push((Value::Null, 0));
                                }
                                crate::query::logical::AggregateFunction::Max => {
                                    regular_accs.push(Value::Null);
                                    avg_accumulators.push((Value::Null, 0));
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
                                    } else {
                                        Value::Float(0.0)
                                    };
                                    avg_accumulators.push((initial_sum, 0));
                                    regular_accs.push(Value::Null);
                                }
                            }
                        }
                        _ => {
                            regular_accs.push(Value::Null);
                            avg_accumulators.push((Value::Null, 0));
                        }
                    }
                }

                (regular_accs, avg_accumulators)
            });

            // Update accumulators
            for (i, expr) in self.aggr_expr.iter().enumerate() {
                if let crate::query::logical::Expr::AggregateExpr {
                    func,
                    expr: inner_expr,
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
                                (Value::Vector(sum_vec), Value::Vector(v)) => {
                                    if sum_vec.len() == v.len() {
                                        for (opt, val) in sum_vec.iter_mut().zip(v.iter()) {
                                            *opt += val;
                                        }
                                    }
                                }
                                (Value::Matrix(sum_mat), Value::Matrix(v))
                                    if sum_mat.len() == v.len()
                                        && !sum_mat.is_empty()
                                        && sum_mat[0].len() == v[0].len() =>
                                {
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
                                    _ => {}
                                },
                                Value::Int(ref mut sum) => {
                                    match &val {
                                        Value::Int(v) => {
                                            // Convert to Float for precision
                                            *sum_ref = Value::Float(*sum as f32 + *v as f32);
                                        }
                                        Value::Float(v) => {
                                            *sum_ref = Value::Float(*sum as f32 + v);
                                        }
                                        _ => {}
                                    }
                                }
                                Value::Vector(ref mut sum_vec) => {
                                    if let Value::Vector(v) = &val {
                                        if sum_vec.len() == v.len() {
                                            for (s, val) in sum_vec.iter_mut().zip(v.iter()) {
                                                *s += val;
                                            }
                                        }
                                    }
                                }
                                Value::Matrix(ref mut sum_mat) => {
                                    if let Value::Matrix(v) = &val {
                                        // Element-wise sum
                                        if sum_mat.len() == v.len()
                                            && !sum_mat.is_empty()
                                            && sum_mat[0].len() == v[0].len()
                                        {
                                            for i in 0..sum_mat.len() {
                                                for j in 0..sum_mat[i].len() {
                                                    sum_mat[i][j] += v[i][j];
                                                }
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
                                    (c, n) => {
                                        if let Some(std::cmp::Ordering::Less) = n.compare(c) {
                                            *c = n.clone();
                                        }
                                    }
                                },
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // Output rows - compute AVG from sum/count before outputting
        let mut output_rows = Vec::new();
        for (key, (accs, avg_accs)) in groups {
            let mut values = key; // Group keys first

            // Build final accumulator values, computing AVG where needed
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
                                Value::Int(s) => Value::Float(*s as f32 / *count as f32),
                                Value::Vector(v) => {
                                    Value::Vector(v.iter().map(|x| x / *count as f32).collect())
                                }
                                Value::Matrix(m) => Value::Matrix(
                                    m.iter()
                                        .map(|row| row.iter().map(|x| x / *count as f32).collect())
                                        .collect(),
                                ),
                                _ => Value::Null,
                            };
                            final_accs.push(avg);
                        } else {
                            final_accs.push(Value::Null);
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

        let k = dataset.rows.len().max(1);
        let n = self.query.len();
        let id = TensorId::new();
        let meta = TensorMetadata::new(id, None);
        let query_tensor =
            crate::core::tensor::Tensor::new(id, Shape::new(vec![n]), self.query.clone(), meta)
                .map_err(EngineError::InvalidOp)?;

        let results = index
            .search(&query_tensor, k)
            .map_err(EngineError::InvalidOp)?;

        let row_ids: Vec<usize> = results
            .into_iter()
            .filter(|(_, score)| {
                if self.strict {
                    *score > self.threshold
                } else {
                    *score >= self.threshold
                }
            })
            .map(|(id, _)| id)
            .collect();

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
        crate::query::logical::Expr::Column(name) => row.get(name).cloned().unwrap_or(Value::Null),
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
                    Value::String(s) => s.parse::<i64>().map(Value::Int).unwrap_or(Value::Null),
                    Value::Bool(b) => Value::Int(if b { 1 } else { 0 }),
                    _ => Value::Null,
                },
                CastTarget::Float => match val {
                    Value::Float(f) => Value::Float(f),
                    Value::Int(n) => Value::Float(n as f32),
                    Value::String(s) => s.parse::<f32>().map(Value::Float).unwrap_or(Value::Null),
                    _ => Value::Null,
                },
                CastTarget::Text => Value::String(match val {
                    Value::String(s) => s,
                    Value::Int(n) => n.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => return Value::Null,
                }),
                CastTarget::Bool => match val {
                    Value::Bool(b) => Value::Bool(b),
                    Value::Int(n) => Value::Bool(n != 0),
                    Value::String(s) => Value::Bool(!s.is_empty()),
                    _ => Value::Null,
                },
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
                    Some(Value::Vector(v)) => {
                        Value::Float(v.iter().map(|x| x * x).sum::<f32>().sqrt())
                    }
                    _ => Value::Null,
                },
                VectorFnKind::CosineSim => match (vals.first(), vals.get(1)) {
                    (Some(Value::Vector(a)), Some(Value::Vector(b))) => {
                        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if na == 0.0 || nb == 0.0 {
                            Value::Float(0.0)
                        } else {
                            Value::Float(dot / (na * nb))
                        }
                    }
                    _ => Value::Null,
                },
                VectorFnKind::Dot => match (vals.first(), vals.get(1)) {
                    (Some(Value::Vector(a)), Some(Value::Vector(b))) => {
                        Value::Float(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>())
                    }
                    _ => Value::Null,
                },
                VectorFnKind::VecAdd => match (vals.first(), vals.get(1)) {
                    (Some(Value::Vector(a)), Some(Value::Vector(b))) => {
                        Value::Vector(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
                    }
                    _ => Value::Null,
                },
                VectorFnKind::VecScale => match (vals.first(), vals.get(1)) {
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
            }
        }
        _ => Value::Null,
    }
}

/// Nested-loop join executor — INNER or LEFT join on an equi-condition.
#[derive(Debug)]
pub struct NestedLoopJoinExec {
    pub left: Box<dyn PhysicalPlan>,
    pub right: Box<dyn PhysicalPlan>,
    pub left_col: String,
    pub right_col: String,
    pub join_type: crate::query::logical::JoinType,
    pub output_schema: Arc<Schema>,
}

impl PhysicalPlan for NestedLoopJoinExec {
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

        let mut output = Vec::new();

        match self.join_type {
            JoinType::Inner | JoinType::Left | JoinType::Full => {
                // Build hash map: right key → indices into right_rows
                let mut right_map: HashMap<String, Vec<usize>> = HashMap::new();
                for (i, row) in right_rows.iter().enumerate() {
                    let key = format!("{:?}", row.values[right_col_idx]);
                    right_map.entry(key).or_default().push(i);
                }

                let mut matched_right: HashSet<usize> = HashSet::new();

                for left_row in &left_rows {
                    let key = format!("{:?}", left_row.values[left_col_idx]);
                    match right_map.get(&key) {
                        Some(right_indices) => {
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
                        None if matches!(self.join_type, JoinType::Left | JoinType::Full) => {
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

                // FULL: emit unmatched right rows with NULL left values
                if self.join_type == JoinType::Full {
                    for (i, right_row) in right_rows.iter().enumerate() {
                        if !matched_right.contains(&i) {
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
                // Build hash map: left key → indices into left_rows
                let mut left_map: HashMap<String, Vec<usize>> = HashMap::new();
                for (i, row) in left_rows.iter().enumerate() {
                    let key = format!("{:?}", row.values[left_col_idx]);
                    left_map.entry(key).or_default().push(i);
                }

                for right_row in &right_rows {
                    let key = format!("{:?}", right_row.values[right_col_idx]);
                    match left_map.get(&key) {
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
                            // Unmatched right row: emit NULL left + right values
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
