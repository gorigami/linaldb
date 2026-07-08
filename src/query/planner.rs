use crate::core::tuple::Schema;
use crate::engine::{EngineError, TensorDb};
use crate::query::logical::{Expr, LogicalPlan};
use crate::query::physical::{
    AggregateExec, FilterExec, IndexScanExec, LimitExec, NestedLoopJoinExec, PhysicalPlan,
    ProjectionExec, SeqScanExec, SortExec, VectorSearchExec,
};
use std::sync::Arc;

pub struct Planner<'a> {
    db: &'a TensorDb,
}

impl<'a> Planner<'a> {
    pub fn new(db: &'a TensorDb) -> Self {
        Self { db }
    }

    pub fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<Box<dyn PhysicalPlan>, EngineError> {
        match logical_plan {
            LogicalPlan::Scan {
                dataset_name,
                schema,
            } => Ok(Box::new(SeqScanExec {
                dataset_name: dataset_name.clone(),
                schema: schema.clone(),
            })),
            LogicalPlan::Filter { input, predicate } => {
                let input_plan = self.create_physical_plan(input)?;

                // OPTIMIZATION: Check if we can use an Index
                if let LogicalPlan::Scan {
                    dataset_name,
                    schema,
                } = input.as_ref()
                {
                    if let Some(index_plan) =
                        self.try_optimize_filter(dataset_name, schema, predicate)
                    {
                        return Ok(index_plan);
                    }
                }

                // Default: Filter Scan
                // We need to convert logical Expr to a physical predicate closure
                // This is tricky because closures need to be generic or boxed.
                // For MVP, we'll implement a simple interpreter for Expr inside predicate.
                let predicate_clone = predicate.clone();
                let predicate_fn = Box::new(move |row: &crate::core::tuple::Tuple| {
                    evaluate_expr(&predicate_clone, row)
                });

                Ok(Box::new(FilterExec {
                    input: input_plan,
                    predicate: predicate_fn,
                }))
            }
            LogicalPlan::Project { input, columns } => {
                let input_plan = self.create_physical_plan(input)?;
                let input_schema = input_plan.schema();

                let column_indices: Vec<usize> = columns
                    .iter()
                    .map(|name| {
                        input_schema.get_field_index(name).ok_or_else(|| {
                            EngineError::InvalidOp(format!("Column not found: {}", name))
                        })
                    })
                    .collect::<Result<_, _>>()?;

                let output_fields = column_indices
                    .iter()
                    .map(|&idx| input_schema.fields[idx].clone())
                    .collect();
                let output_schema = Arc::new(Schema::new(output_fields));

                Ok(Box::new(ProjectionExec {
                    input: input_plan,
                    output_schema,
                    column_indices,
                }))
            }
            LogicalPlan::VectorSearch {
                input: _, // Vector Search usually is a leaf for now, or replaces Scan
                column,
                query,
                k,
            } => {
                // Vector Search replaces the Scan entirely if we are searching on a dataset
                // But wait, LogicalPlan::VectorSearch takes input.
                // Usually VectorSearch IS the access method.
                // Let's assume input is Scan.
                // If input is not Scan, we might need to materialize input first?
                // For MVP: assume input is Scan(dataset).

                match logical_plan {
                    LogicalPlan::VectorSearch {
                        input,
                        column: _,
                        query: _,
                        k: _,
                    } => {
                        if let LogicalPlan::Scan {
                            dataset_name,
                            schema,
                        } = input.as_ref()
                        {
                            Ok(Box::new(VectorSearchExec {
                                dataset_name: dataset_name.clone(),
                                schema: schema.clone(),
                                column: column.clone(),
                                query: query.clone(),
                                k: *k,
                            }))
                        } else {
                            Err(EngineError::InvalidOp(
                                "VectorSearch input must be a Scan for now".into(),
                            ))
                        }
                    }
                    _ => unreachable!(),
                }
            }
            LogicalPlan::Limit { input, n, offset } => {
                let input_plan = self.create_physical_plan(input)?;
                Ok(Box::new(LimitExec {
                    input: input_plan,
                    n: *n,
                    offset: *offset,
                }))
            }
            LogicalPlan::Sort { input, columns } => {
                let input_plan = self.create_physical_plan(input)?;
                Ok(Box::new(SortExec {
                    input: input_plan,
                    columns: columns.clone(),
                }))
            }
            LogicalPlan::Aggregate {
                input,
                group_expr,
                aggr_expr,
            } => {
                let input_plan = self.create_physical_plan(input)?;
                let schema = logical_plan.schema();
                Ok(Box::new(AggregateExec {
                    input: input_plan,
                    group_expr: group_expr.clone(),
                    aggr_expr: aggr_expr.clone(),
                    schema,
                }))
            }
            LogicalPlan::Join {
                left,
                right,
                left_col,
                right_col,
                join_type,
            } => {
                let left_plan = self.create_physical_plan(left)?;
                let right_plan = self.create_physical_plan(right)?;
                let output_schema = logical_plan.schema();
                Ok(Box::new(NestedLoopJoinExec {
                    left: left_plan,
                    right: right_plan,
                    left_col: left_col.clone(),
                    right_col: right_col.clone(),
                    join_type: *join_type,
                    output_schema,
                }))
            }
        }
    }

    fn try_optimize_filter(
        &self,
        dataset_name: &str,
        schema: &Schema,
        predicate: &Expr,
    ) -> Option<Box<dyn PhysicalPlan>> {
        // Look for: Col = Literal
        if let Expr::BinaryExpr { left, op, right } = predicate {
            if op == "=" {
                if let (Expr::Column(col_name), Expr::Literal(val)) =
                    (left.as_ref(), right.as_ref())
                {
                    // Check if index exists
                    if let Ok(dataset) = self.db.get_dataset(dataset_name) {
                        if let Some(index) = dataset.get_index(col_name) {
                            if index.index_type() == crate::core::index::IndexType::Hash {
                                // FOUND MATCH! Use IndexScan
                                return Some(Box::new(IndexScanExec {
                                    dataset_name: dataset_name.to_string(),
                                    schema: Arc::new(schema.clone()),
                                    column: col_name.clone(),
                                    value: val.clone(),
                                }));
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

/// Public entry point for evaluating a logical predicate against a row.
pub fn evaluate_predicate(expr: &Expr, row: &crate::core::tuple::Tuple) -> bool {
    evaluate_expr(expr, row)
}

fn evaluate_expr(expr: &Expr, row: &crate::core::tuple::Tuple) -> bool {
    match expr {
        Expr::And(left, right) => evaluate_expr(left, row) && evaluate_expr(right, row),
        Expr::Or(left, right) => evaluate_expr(left, row) || evaluate_expr(right, row),
        Expr::Not(inner) => !evaluate_expr(inner, row),
        Expr::IsNull(inner) => matches!(
            eval_value(inner, row),
            Some(crate::core::value::Value::Null) | None
        ),
        Expr::IsNotNull(inner) => !matches!(
            eval_value(inner, row),
            Some(crate::core::value::Value::Null) | None
        ),
        Expr::In { expr, list } => {
            if let Some(val) = eval_value(expr, row) {
                list.iter().any(|item| {
                    eval_value(item, row)
                        .map(|v| val.compare(&v) == Some(std::cmp::Ordering::Equal))
                        .unwrap_or(false)
                })
            } else {
                false
            }
        }
        Expr::Between { expr, low, high } => {
            let val = eval_value(expr, row);
            let lo = eval_value(low, row);
            let hi = eval_value(high, row);
            if let (Some(v), Some(l), Some(h)) = (val, lo, hi) {
                let ge = matches!(
                    v.compare(&l),
                    Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
                );
                let le = matches!(
                    v.compare(&h),
                    Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
                );
                ge && le
            } else {
                false
            }
        }
        Expr::BinaryExpr { left, op, right } => {
            let left_val = eval_value(left, row);
            let right_val = eval_value(right, row);

            if let (Some(l), Some(r)) = (left_val, right_val) {
                let ord = l.compare(&r);
                match op.as_str() {
                    "=" => ord == Some(std::cmp::Ordering::Equal),
                    "!=" => ord.is_some() && ord != Some(std::cmp::Ordering::Equal),
                    ">" => ord == Some(std::cmp::Ordering::Greater),
                    "<" => ord == Some(std::cmp::Ordering::Less),
                    ">=" => matches!(
                        ord,
                        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
                    ),
                    "<=" => matches!(
                        ord,
                        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
                    ),
                    _ => false,
                }
            } else {
                false
            }
        }
        _ => false,
    }
}

fn eval_value(expr: &Expr, row: &crate::core::tuple::Tuple) -> Option<crate::core::value::Value> {
    match expr {
        Expr::Column(name) => row.get(name).cloned(),
        Expr::Literal(val) => Some(val.clone()),
        _ => None,
    }
}
