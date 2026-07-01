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
}

impl PhysicalPlan for LimitExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let input_rows = self.input.execute(db)?;
        Ok(input_rows.into_iter().take(self.n).collect())
    }
}

/// Sort Executor
#[derive(Debug)]
pub struct SortExec {
    pub input: Box<dyn PhysicalPlan>,
    pub column: String,
    pub ascending: bool,
}

impl PhysicalPlan for SortExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let rows = self.input.execute(db)?;
        let schema = self.schema();
        let col_idx = schema.get_field_index(&self.column).ok_or_else(|| {
            EngineError::InvalidOp(format!("Column not found for sorting: {}", self.column))
        })?;

        let mut sorted_rows = rows;
        sorted_rows.sort_by(|a, b| {
            let val_a = &a.values[col_idx];
            let val_b = &b.values[col_idx];
            let cmp = val_a.compare(val_b).unwrap_or(std::cmp::Ordering::Equal);
            if self.ascending {
                cmp
            } else {
                cmp.reverse()
            }
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
                                    avg_accumulators.push((Value::Null, 0)); // Placeholder
                                }
                                crate::query::logical::AggregateFunction::Sum => {
                                    let val = evaluate_expression(inner, &row);
                                    if let Value::Vector(v) = val {
                                        regular_accs.push(Value::Vector(vec![0.0; v.len()]));
                                    } else if let Value::Matrix(m) = val {
                                        // Zero matrix
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
                                    avg_accumulators.push((Value::Null, 0)); // Placeholder
                                }
                                crate::query::logical::AggregateFunction::Min => {
                                    regular_accs.push(Value::Null);
                                    avg_accumulators.push((Value::Null, 0)); // Placeholder
                                }
                                crate::query::logical::AggregateFunction::Max => {
                                    regular_accs.push(Value::Null);
                                    avg_accumulators.push((Value::Null, 0)); // Placeholder
                                }
                                crate::query::logical::AggregateFunction::Avg => {
                                    // For AVG, initialize sum based on first value type
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
                                    regular_accs.push(Value::Null); // Placeholder, will be replaced with computed avg
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
                        crate::query::logical::AggregateFunction::Sum => {
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
                        crate::query::logical::AggregateFunction::Avg => {
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
                    if matches!(func, crate::query::logical::AggregateFunction::Avg) {
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
        _ => Value::Null,
    }
}
