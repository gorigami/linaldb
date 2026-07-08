use crate::core::dataset_legacy;
use crate::core::value::{Value, ValueType};
use crate::dsl::ast::*;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use crate::query::logical::{AggregateFunction, Expr as LogicalExpr, JoinType, LogicalPlan};
use crate::query::planner::Planner;

// ─── Dataset query execution ──────────────────────────────────────────────────

pub(super) fn execute_create_dataset_from(
    db: &mut TensorDb,
    name: String,
    clause: DatasetFromClause,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let source_ds = db
        .get_dataset(&clause.source)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    let source_schema = source_ds.schema.clone();

    let mut plan = LogicalPlan::Scan {
        dataset_name: clause.source.clone(),
        schema: source_schema,
    };

    if let Some(f) = clause.filter {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: dataset_filter_to_logical(&f),
        };
    }

    if !clause.group_by.is_empty() {
        let group_exprs: Vec<LogicalExpr> = clause
            .group_by
            .iter()
            .map(|c| LogicalExpr::Column(c.clone()))
            .collect();
        let aggr_exprs: Vec<LogicalExpr> = clause
            .select
            .as_ref()
            .map(|exprs| {
                exprs
                    .iter()
                    .filter_map(|e| match e {
                        SelectExpr::Aggregate { func, expr } => Some(LogicalExpr::AggregateExpr {
                            func: agg_func_to_logical(func),
                            expr: Box::new(dsl_expr_to_logical_expr(expr)),
                        }),
                        SelectExpr::Column(_) => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        plan = LogicalPlan::Aggregate {
            input: Box::new(plan),
            group_expr: group_exprs,
            aggr_expr: aggr_exprs,
        };
    } else if let Some(exprs) = clause.select {
        let has_aggr = exprs
            .iter()
            .any(|e| matches!(e, SelectExpr::Aggregate { .. }));
        if has_aggr {
            let aggr_exprs: Vec<LogicalExpr> = exprs
                .into_iter()
                .filter_map(|e| match e {
                    SelectExpr::Aggregate { func, expr } => Some(LogicalExpr::AggregateExpr {
                        func: agg_func_to_logical(&func),
                        expr: Box::new(dsl_expr_to_logical_expr(&expr)),
                    }),
                    SelectExpr::Column(_) => None,
                })
                .collect();
            plan = LogicalPlan::Aggregate {
                input: Box::new(plan),
                group_expr: vec![],
                aggr_expr: aggr_exprs,
            };
        } else {
            let cols: Vec<String> = exprs
                .into_iter()
                .filter_map(|e| match e {
                    SelectExpr::Column(c) => Some(c),
                    SelectExpr::Aggregate { .. } => None,
                })
                .collect();
            plan = LogicalPlan::Project {
                input: Box::new(plan),
                columns: cols,
            };
        }
    }

    if let Some(f) = clause.having {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: dataset_filter_to_logical(&f),
        };
    }

    if let Some(ord) = clause.order_by {
        plan = LogicalPlan::Sort {
            input: Box::new(plan),
            column: ord.column,
            ascending: ord.ascending,
        };
    }

    if let Some(n) = clause.limit {
        plan = LogicalPlan::Limit {
            input: Box::new(plan),
            n,
        };
    }

    let planner = Planner::new(db);
    let physical_plan = planner
        .create_physical_plan(&plan)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    let result_rows = physical_plan.execute(db).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let result_schema = physical_plan.schema();

    db.create_dataset(name.clone(), result_schema)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    let target_ds = db.get_dataset_mut(&name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    target_ds.rows = result_rows;
    target_ds
        .metadata
        .update_stats(&target_ds.schema, &target_ds.rows);
    Ok(DslOutput::None)
}

pub(super) fn execute_select(
    db: &mut TensorDb,
    s: SelectStmt,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let source_ds = db.get_dataset(&s.dataset).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let source_schema = source_ds.schema.clone();

    let mut plan = LogicalPlan::Scan {
        dataset_name: s.dataset.clone(),
        schema: source_schema.clone(),
    };

    // Build join nodes left-to-right
    for join in &s.joins {
        let right_ds = db
            .get_dataset(&join.dataset)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        let right_schema = right_ds.schema.clone();
        let right_plan = LogicalPlan::Scan {
            dataset_name: join.dataset.clone(),
            schema: right_schema,
        };
        let join_type = match join.kind {
            JoinKind::Inner => JoinType::Inner,
            JoinKind::Left => JoinType::Left,
        };
        plan = LogicalPlan::Join {
            left: Box::new(plan),
            right: Box::new(right_plan),
            left_col: join.left_col.clone(),
            right_col: join.right_col.clone(),
            join_type,
        };
    }

    if let Some(filter_expr) = &s.filter {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: dsl_expr_to_logical_expr(filter_expr),
        };
    }

    if !s.group_by.is_empty() {
        let group_exprs: Vec<LogicalExpr> = s
            .group_by
            .iter()
            .map(|c| LogicalExpr::Column(c.clone()))
            .collect();
        let aggr_exprs: Vec<LogicalExpr> = match &s.columns {
            SelectColumns::Named(exprs) => exprs
                .iter()
                .filter_map(|e| match e {
                    SelectExpr::Aggregate { func, expr } => Some(LogicalExpr::AggregateExpr {
                        func: agg_func_to_logical(func),
                        expr: Box::new(dsl_expr_to_logical_expr(expr)),
                    }),
                    SelectExpr::Column(_) => None,
                })
                .collect(),
            SelectColumns::All => vec![],
        };
        plan = LogicalPlan::Aggregate {
            input: Box::new(plan),
            group_expr: group_exprs,
            aggr_expr: aggr_exprs,
        };
        if let Some(having_expr) = &s.having {
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: dsl_expr_to_logical_expr(having_expr),
            };
        }
    } else {
        if let Some(having_expr) = &s.having {
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: dsl_expr_to_logical_expr(having_expr),
            };
        }
        if let Some(ord) = &s.order_by {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                column: ord.column.clone(),
                ascending: ord.ascending,
            };
        }
        if let Some(n) = s.limit {
            plan = LogicalPlan::Limit {
                input: Box::new(plan),
                n,
            };
        }
        let effective_schema = plan.schema();
        let cols: Vec<String> = match s.columns {
            SelectColumns::All => effective_schema
                .fields
                .iter()
                .map(|f| f.name.clone())
                .collect(),
            SelectColumns::Named(exprs) => exprs
                .into_iter()
                .filter_map(|e| match e {
                    SelectExpr::Column(name) => Some(name),
                    SelectExpr::Aggregate { .. } => None,
                })
                .collect(),
        };
        plan = LogicalPlan::Project {
            input: Box::new(plan),
            columns: cols,
        };
    }

    let planner = Planner::new(db);
    let physical_plan = planner
        .create_physical_plan(&plan)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    let result_rows = physical_plan.execute(db).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let result_schema = physical_plan.schema();
    let ds = dataset_legacy::Dataset::with_rows(
        dataset_legacy::DatasetId(0),
        result_schema,
        result_rows,
        Some("Query Result".into()),
    )
    .map_err(|e| DslError::Parse {
        line: line_no,
        msg: e,
    })?;
    Ok(DslOutput::Table(ds))
}

pub(super) fn execute_add_computed_column(
    db: &mut TensorDb,
    dataset: &str,
    col_name: &str,
    expr: &Expr,
    lazy: bool,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let ds = db.get_dataset(dataset).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;

    if ds.schema.fields.iter().any(|f| f.name == col_name) {
        return Err(DslError::Parse {
            line: line_no,
            msg: format!(
                "Column '{}' already exists in dataset '{}'",
                col_name, dataset
            ),
        });
    }

    let logical_expr = dsl_expr_to_logical_expr(expr);

    if lazy {
        let first_row = ds.rows.first().ok_or_else(|| DslError::Parse {
            line: line_no,
            msg: format!(
                "Cannot infer type for computed column '{}' from empty dataset",
                col_name
            ),
        })?;
        let field_names: Vec<String> = ds.schema.fields.iter().map(|f| f.name.clone()).collect();
        let env: std::collections::HashMap<&str, &Value> = field_names
            .iter()
            .zip(first_row.values.iter())
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        let vtype = match eval_row_expr(expr, &env) {
            Value::Int(_) => ValueType::Int,
            Value::Float(_) => ValueType::Float,
            Value::String(_) => ValueType::String,
            Value::Bool(_) => ValueType::Bool,
            Value::Vector(v) => ValueType::Vector(v.len()),
            Value::Matrix(m) => {
                let r = m.len();
                let c = m.first().map_or(0, |row| row.len());
                ValueType::Matrix(r, c)
            }
            Value::Null => ValueType::Float,
        };

        db.alter_dataset_add_computed_column(
            dataset,
            col_name.to_string(),
            vtype,
            vec![Value::Null; ds.rows.len()],
            logical_expr,
            true,
        )
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    } else {
        if ds.rows.is_empty() {
            return Err(DslError::Parse {
                line: line_no,
                msg: format!(
                    "Cannot infer type for computed column '{}' from empty dataset",
                    col_name
                ),
            });
        }

        let field_names: Vec<String> = ds.schema.fields.iter().map(|f| f.name.clone()).collect();
        let computed: Vec<Value> = ds
            .rows
            .iter()
            .map(|row| {
                let env: std::collections::HashMap<&str, &Value> = field_names
                    .iter()
                    .zip(row.values.iter())
                    .map(|(k, v)| (k.as_str(), v))
                    .collect();
                eval_row_expr(expr, &env)
            })
            .collect();

        let vtype = match &computed[0] {
            Value::Int(_) => ValueType::Int,
            Value::Float(_) => ValueType::Float,
            Value::String(_) => ValueType::String,
            Value::Bool(_) => ValueType::Bool,
            Value::Vector(v) => ValueType::Vector(v.len()),
            Value::Matrix(m) => ValueType::Matrix(m.len(), m.first().map_or(0, |r| r.len())),
            Value::Null => ValueType::Null,
        };

        db.alter_dataset_add_computed_column(
            dataset,
            col_name.to_string(),
            vtype,
            computed,
            logical_expr,
            false,
        )
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    }

    Ok(DslOutput::Message(format!(
        "Added computed column '{}' to dataset '{}'",
        col_name, dataset
    )))
}

// ─── Shared logical plan helpers ──────────────────────────────────────────────

pub(super) fn agg_func_to_logical(f: &AggFuncAst) -> AggregateFunction {
    match f {
        AggFuncAst::Sum => AggregateFunction::Sum,
        AggFuncAst::Avg => AggregateFunction::Avg,
        AggFuncAst::Count => AggregateFunction::Count,
        AggFuncAst::Min => AggregateFunction::Min,
        AggFuncAst::Max => AggregateFunction::Max,
    }
}

pub(super) fn dataset_filter_to_logical(f: &DatasetFilter) -> LogicalExpr {
    let op = match f.op {
        CmpOp::Eq => "=",
        CmpOp::NotEq => "!=",
        CmpOp::Gt => ">",
        CmpOp::GtEq => ">=",
        CmpOp::Lt => "<",
        CmpOp::LtEq => "<=",
    };
    let val = match &f.value {
        FilterValue::Int(n) => Value::Int(*n),
        FilterValue::Float(f) => Value::Float(*f as f32),
        FilterValue::Str(s) => Value::String(s.clone()),
        FilterValue::Bool(b) => Value::Bool(*b),
    };
    LogicalExpr::BinaryExpr {
        left: Box::new(LogicalExpr::Column(f.column.clone())),
        op: op.to_string(),
        right: Box::new(LogicalExpr::Literal(val)),
    }
}

pub(super) fn dsl_expr_to_logical_expr(e: &Expr) -> LogicalExpr {
    match e {
        Expr::Ref(name) => LogicalExpr::Column(name.clone()),
        Expr::Int(n) => LogicalExpr::Literal(Value::Int(*n)),
        Expr::Scalar(f) => LogicalExpr::Literal(Value::Float(*f as f32)),
        Expr::StringLit(s) => LogicalExpr::Literal(Value::String(s.clone())),
        Expr::Infix { op, lhs, rhs } => {
            let sym = match op {
                InfixOp::Add => "+",
                InfixOp::Subtract => "-",
                InfixOp::Multiply => "*",
                InfixOp::Divide => "/",
                InfixOp::Eq => "=",
                InfixOp::NotEq => "!=",
                InfixOp::Gt => ">",
                InfixOp::Lt => "<",
                InfixOp::GtEq => ">=",
                InfixOp::LtEq => "<=",
            };
            LogicalExpr::BinaryExpr {
                left: Box::new(dsl_expr_to_logical_expr(lhs)),
                op: sym.to_string(),
                right: Box::new(dsl_expr_to_logical_expr(rhs)),
            }
        }
        Expr::And(lhs, rhs) => LogicalExpr::And(
            Box::new(dsl_expr_to_logical_expr(lhs)),
            Box::new(dsl_expr_to_logical_expr(rhs)),
        ),
        Expr::Or(lhs, rhs) => LogicalExpr::Or(
            Box::new(dsl_expr_to_logical_expr(lhs)),
            Box::new(dsl_expr_to_logical_expr(rhs)),
        ),
        Expr::Not(inner) => LogicalExpr::Not(Box::new(dsl_expr_to_logical_expr(inner))),
        Expr::IsNull(inner) => LogicalExpr::IsNull(Box::new(dsl_expr_to_logical_expr(inner))),
        Expr::IsNotNull(inner) => LogicalExpr::IsNotNull(Box::new(dsl_expr_to_logical_expr(inner))),
        _ => LogicalExpr::Literal(Value::Null),
    }
}

// ─── UPDATE ───────────────────────────────────────────────────────────────────

pub(super) fn execute_update(
    db: &mut TensorDb,
    s: UpdateStmt,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    // Build a filter predicate (if any) using the same physical evaluator
    let predicate: Option<Box<dyn Fn(&crate::core::tuple::Tuple) -> bool>> =
        s.filter
            .as_ref()
            .map(|f| -> Box<dyn Fn(&crate::core::tuple::Tuple) -> bool> {
                let logical = dsl_expr_to_logical_expr(f);
                Box::new(move |row| {
                    use crate::query::planner::evaluate_predicate;
                    evaluate_predicate(&logical, row)
                })
            });

    let ds = db
        .get_dataset_mut(&s.dataset)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    let field_names: Vec<String> = ds.schema.fields.iter().map(|f| f.name.clone()).collect();
    let mut updated = 0usize;

    for row in ds.rows.iter_mut() {
        if let Some(ref pred) = predicate {
            if !pred(row) {
                continue;
            }
        }
        for (col_name, expr) in &s.assignments {
            let env: std::collections::HashMap<&str, &Value> = field_names
                .iter()
                .zip(row.values.iter())
                .map(|(k, v)| (k.as_str(), v))
                .collect();
            let new_val = eval_row_expr(expr, &env);
            if let Some(idx) = field_names.iter().position(|n| n == col_name) {
                row.values[idx] = new_val;
            }
        }
        updated += 1;
    }

    Ok(DslOutput::Message(format!(
        "Updated {} row(s) in '{}'",
        updated, s.dataset
    )))
}

// ─── DELETE ───────────────────────────────────────────────────────────────────

pub(super) fn execute_delete(
    db: &mut TensorDb,
    s: DeleteStmt,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let predicate: Option<Box<dyn Fn(&crate::core::tuple::Tuple) -> bool>> =
        s.filter
            .as_ref()
            .map(|f| -> Box<dyn Fn(&crate::core::tuple::Tuple) -> bool> {
                let logical = dsl_expr_to_logical_expr(f);
                Box::new(move |row| {
                    use crate::query::planner::evaluate_predicate;
                    evaluate_predicate(&logical, row)
                })
            });

    let ds = db
        .get_dataset_mut(&s.dataset)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    let before = ds.rows.len();
    match predicate {
        Some(pred) => ds.rows.retain(|row| !pred(row)),
        None => ds.rows.clear(),
    }
    let deleted = before - ds.rows.len();

    Ok(DslOutput::Message(format!(
        "Deleted {} row(s) from '{}'",
        deleted, s.dataset
    )))
}

// ─── Row-level expression evaluation (for computed columns) ───────────────────

fn eval_row_expr(expr: &Expr, env: &std::collections::HashMap<&str, &Value>) -> Value {
    match expr {
        Expr::Ref(name) => env.get(name.as_str()).map_or(Value::Null, |v| (*v).clone()),
        Expr::Int(n) => Value::Int(*n),
        Expr::Scalar(f) => Value::Float(*f as f32),
        Expr::StringLit(s) => Value::String(s.clone()),
        Expr::Infix { op, lhs, rhs } => {
            let l = eval_row_expr(lhs, env);
            let r = eval_row_expr(rhs, env);
            match (op, l, r) {
                (InfixOp::Add, Value::Int(a), Value::Int(b)) => Value::Int(a + b),
                (InfixOp::Add, Value::Float(a), Value::Float(b)) => Value::Float(a + b),
                (InfixOp::Add, Value::Int(a), Value::Float(b)) => Value::Float(a as f32 + b),
                (InfixOp::Add, Value::Float(a), Value::Int(b)) => Value::Float(a + b as f32),
                (InfixOp::Subtract, Value::Int(a), Value::Int(b)) => Value::Int(a - b),
                (InfixOp::Subtract, Value::Float(a), Value::Float(b)) => Value::Float(a - b),
                (InfixOp::Subtract, Value::Int(a), Value::Float(b)) => Value::Float(a as f32 - b),
                (InfixOp::Subtract, Value::Float(a), Value::Int(b)) => Value::Float(a - b as f32),
                (InfixOp::Multiply, Value::Int(a), Value::Int(b)) => Value::Int(a * b),
                (InfixOp::Multiply, Value::Float(a), Value::Float(b)) => Value::Float(a * b),
                (InfixOp::Multiply, Value::Int(a), Value::Float(b)) => Value::Float(a as f32 * b),
                (InfixOp::Multiply, Value::Float(a), Value::Int(b)) => Value::Float(a * b as f32),
                (InfixOp::Divide, Value::Int(a), Value::Int(b)) if b != 0 => Value::Int(a / b),
                (InfixOp::Divide, Value::Float(a), Value::Float(b)) => Value::Float(a / b),
                (InfixOp::Divide, Value::Int(a), Value::Float(b)) => Value::Float(a as f32 / b),
                (InfixOp::Divide, Value::Float(a), Value::Int(b)) => Value::Float(a / b as f32),
                _ => Value::Null,
            }
        }
        _ => Value::Null,
    }
}
