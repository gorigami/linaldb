use crate::core::dataset_legacy;
use crate::core::tuple::Tuple;
use crate::core::value::{Value, ValueType};
use crate::dsl::ast::*;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use crate::query::logical::{AggregateFunction, Expr as LogicalExpr, JoinType, LogicalPlan};
use crate::query::planner::Planner;

type RowPredicate = Box<dyn Fn(&Tuple) -> bool>;

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
            predicate: dsl_expr_to_logical_expr(&f),
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
                        SelectExpr::Aggregate { func, expr, alias } => {
                            Some(LogicalExpr::AggregateExpr {
                                func: agg_func_to_logical(func),
                                expr: Box::new(dsl_expr_to_logical_expr(expr)),
                                alias: alias.clone(),
                            })
                        }
                        SelectExpr::Column(_)
                        | SelectExpr::Window { .. }
                        | SelectExpr::Computed { .. } => None,
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
                    SelectExpr::Aggregate { func, expr, alias } => {
                        Some(LogicalExpr::AggregateExpr {
                            func: agg_func_to_logical(&func),
                            expr: Box::new(dsl_expr_to_logical_expr(&expr)),
                            alias,
                        })
                    }
                    SelectExpr::Column(_)
                    | SelectExpr::Window { .. }
                    | SelectExpr::Computed { .. } => None,
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
                    SelectExpr::Aggregate { .. }
                    | SelectExpr::Window { .. }
                    | SelectExpr::Computed { .. } => None,
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
            predicate: dsl_expr_to_logical_expr(&f),
        };
    }

    if let Some(ord) = clause.order_by {
        plan = LogicalPlan::Sort {
            input: Box::new(plan),
            columns: ord.columns,
        };
    }

    if let Some(n) = clause.limit {
        plan = LogicalPlan::Limit {
            input: Box::new(plan),
            n,
            offset: clause.offset.unwrap_or(0),
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
    // Materialize CTEs as temp datasets
    let mut cte_names: Vec<String> = vec![];
    for (cte_name, cte_query) in s.ctes {
        let cte_result = execute_select(db, cte_query, line_no)?;
        if let DslOutput::Table(cte_ds) = cte_result {
            let schema = cte_ds.schema.clone();
            let rows = cte_ds.rows;
            db.create_dataset(cte_name.clone(), schema)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            let ds = db
                .get_dataset_mut(&cte_name)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            ds.rows = rows;
            cte_names.push(cte_name);
        }
    }

    // Resolve the FROM source — either a named dataset or an executed subquery.
    let mut plan = match s.source {
        DatasetSource::Named(ref name) => {
            let source_ds = db.get_dataset(name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let schema = source_ds.schema.clone();
            LogicalPlan::Scan {
                dataset_name: name.clone(),
                schema,
            }
        }
        DatasetSource::Subquery { query, alias } => {
            let inner = execute_select(db, *query, line_no)?;
            if let DslOutput::Table(inner_ds) = inner {
                let schema = inner_ds.schema.clone();
                let rows = inner_ds.rows;
                db.create_dataset(alias.clone(), schema.clone())
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                let target = db.get_dataset_mut(&alias).map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
                target.rows = rows;
                LogicalPlan::Scan {
                    dataset_name: alias,
                    schema,
                }
            } else {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Subquery must produce a table result".into(),
                });
            }
        }
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
            JoinKind::Right => JoinType::Right,
            JoinKind::Full => JoinType::Full,
        };
        plan = LogicalPlan::Join {
            left: Box::new(plan),
            right: Box::new(right_plan),
            left_col: join.left_col.clone(),
            right_col: join.right_col.clone(),
            join_type,
            right_dataset_name: join.dataset.clone(),
            similarity_threshold: join.similarity_threshold,
        };
    }

    if let Some(filter_expr) = &s.filter {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: dsl_expr_to_logical_expr(filter_expr),
        };
    }

    // Collect window and computed column exprs for post-processing
    let window_exprs: Vec<SelectExpr> = match &s.columns {
        SelectColumns::Named(exprs) => exprs
            .iter()
            .filter(|e| matches!(e, SelectExpr::Window { .. } | SelectExpr::Computed { .. }))
            .cloned()
            .collect(),
        SelectColumns::All => vec![],
    };

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
                    SelectExpr::Aggregate { func, expr, alias } => {
                        Some(LogicalExpr::AggregateExpr {
                            func: agg_func_to_logical(func),
                            expr: Box::new(dsl_expr_to_logical_expr(expr)),
                            alias: alias.clone(),
                        })
                    }
                    SelectExpr::Column(_)
                    | SelectExpr::Window { .. }
                    | SelectExpr::Computed { .. } => None,
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
        if let Some(ord) = &s.order_by {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                columns: ord.columns.clone(),
            };
        }
        if let Some(n) = s.limit {
            plan = LogicalPlan::Limit {
                input: Box::new(plan),
                n,
                offset: s.offset.unwrap_or(0),
            };
        }
    } else {
        // A SELECT with no GROUP BY can still contain aggregate functions
        // (a "global" aggregate over the whole result set, e.g. `SELECT
        // SUM(price) FROM t`). Without this check, aggregate expressions
        // were silently dropped by the plain-column projection below and
        // the query returned the raw, unaggregated rows.
        let has_aggr = match &s.columns {
            SelectColumns::Named(exprs) => exprs
                .iter()
                .any(|e| matches!(e, SelectExpr::Aggregate { .. })),
            SelectColumns::All => false,
        };

        if has_aggr {
            let aggr_exprs: Vec<LogicalExpr> = match &s.columns {
                SelectColumns::Named(exprs) => exprs
                    .iter()
                    .filter_map(|e| match e {
                        SelectExpr::Aggregate { func, expr, alias } => {
                            Some(LogicalExpr::AggregateExpr {
                                func: agg_func_to_logical(func),
                                expr: Box::new(dsl_expr_to_logical_expr(expr)),
                                alias: alias.clone(),
                            })
                        }
                        SelectExpr::Column(_)
                        | SelectExpr::Window { .. }
                        | SelectExpr::Computed { .. } => None,
                    })
                    .collect(),
                SelectColumns::All => vec![],
            };
            plan = LogicalPlan::Aggregate {
                input: Box::new(plan),
                group_expr: vec![],
                aggr_expr: aggr_exprs,
            };
        }

        if let Some(having_expr) = &s.having {
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: dsl_expr_to_logical_expr(having_expr),
            };
        }
        if let Some(ord) = &s.order_by {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                columns: ord.columns.clone(),
            };
        }
        if let Some(n) = s.limit {
            plan = LogicalPlan::Limit {
                input: Box::new(plan),
                n,
                offset: s.offset.unwrap_or(0),
            };
        }
        // Only project base columns here (Window/Computed added post-execution).
        // Skip entirely for aggregate plans — AggregateExec's schema already
        // reflects the correct output columns.
        if !has_aggr && window_exprs.is_empty() {
            let effective_schema = plan.schema();
            let cols: Vec<String> = match &s.columns {
                SelectColumns::All => effective_schema
                    .fields
                    .iter()
                    .map(|f| f.name.clone())
                    .collect(),
                SelectColumns::Named(exprs) => exprs
                    .iter()
                    .filter_map(|e| match e {
                        SelectExpr::Column(name) => Some(name.clone()),
                        SelectExpr::Aggregate { .. }
                        | SelectExpr::Window { .. }
                        | SelectExpr::Computed { .. } => None,
                    })
                    .collect(),
            };
            if !cols.is_empty() {
                plan = LogicalPlan::Project {
                    input: Box::new(plan),
                    columns: cols,
                };
            }
        }
    }

    if s.distinct {
        plan = LogicalPlan::Distinct {
            input: Box::new(plan),
        };
    }

    let planner = Planner::new(db);
    let physical_plan = planner
        .create_physical_plan(&plan)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    let mut result_rows = physical_plan.execute(db).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let base_schema = physical_plan.schema();

    // Post-process window and computed columns
    let result_schema = if !window_exprs.is_empty() {
        result_rows =
            apply_window_and_computed_exprs(result_rows, &base_schema, &window_exprs, line_no)?;

        // Derive the extended schema from the first row (types are actual, not inferred)
        let extended_schema = if let Some(first_row) = result_rows.first() {
            first_row.schema.clone()
        } else {
            // Fallback: build schema from inference (no rows to peek at).
            // Unaliased Computed names must match apply_window_and_computed_exprs's
            // `__cmp_{idx}` scheme (idx counting only Computed entries, in the
            // same window_exprs order) for the lookup below to find them.
            let mut fields = base_schema.fields.clone();
            let mut computed_idx = 0usize;
            for we in &window_exprs {
                let (col_name, vtype) = match we {
                    SelectExpr::Window { alias, func, .. } => {
                        let vtype = match func {
                            WindowFunc::RowNumber
                            | WindowFunc::Rank
                            | WindowFunc::DenseRank
                            | WindowFunc::Count(_)
                            | WindowFunc::Lag { .. }
                            | WindowFunc::Lead { .. } => ValueType::Int,
                            WindowFunc::Avg(_) | WindowFunc::Sum(_) => ValueType::Float,
                            WindowFunc::Min(_) | WindowFunc::Max(_) => ValueType::Float,
                        };
                        (alias.clone(), vtype)
                    }
                    SelectExpr::Computed { alias, expr } => {
                        let name = alias
                            .clone()
                            .unwrap_or_else(|| format!("__cmp_{}", computed_idx));
                        computed_idx += 1;
                        let vtype = infer_expr_result_type(expr);
                        (name, vtype)
                    }
                    _ => unreachable!(),
                };
                fields.push(crate::core::tuple::Field::new(&col_name, vtype));
            }
            std::sync::Arc::new(crate::core::tuple::Schema::new(fields))
        };

        // Now project to match the SELECT column order. Unaliased Computed
        // exprs must be named identically to how apply_window_and_computed_exprs
        // named them when appending the column (`__cmp_{idx}`, idx counting
        // only Computed entries in order) — a mismatch here means the
        // lookup below silently drops the column from the output entirely.
        let ordered_cols: Vec<String> = match &s.columns {
            SelectColumns::All => extended_schema
                .fields
                .iter()
                .map(|f| f.name.clone())
                .collect(),
            SelectColumns::Named(exprs) => {
                let mut computed_idx = 0usize;
                // Aggregate fields land in base_schema in the same relative
                // order they appear here (this branch only runs without a
                // GROUP BY, so base_schema's fields are exactly the
                // aggregate outputs, one per Aggregate entry below) — look
                // up the real output name (honors G2's AS alias) instead of
                // a placeholder, or the column is silently dropped by the
                // get_field_index lookup a few lines down.
                let mut agg_idx = 0usize;
                exprs
                    .iter()
                    .map(|e| match e {
                        SelectExpr::Column(name) => name.clone(),
                        SelectExpr::Window { alias, .. } => alias.clone(),
                        SelectExpr::Computed { alias, .. } => {
                            let name = alias
                                .clone()
                                .unwrap_or_else(|| format!("__cmp_{}", computed_idx));
                            computed_idx += 1;
                            name
                        }
                        SelectExpr::Aggregate { .. } => {
                            let name = base_schema
                                .fields
                                .get(agg_idx)
                                .map(|f| f.name.clone())
                                .unwrap_or_else(|| "agg".to_string());
                            agg_idx += 1;
                            name
                        }
                    })
                    .collect()
            }
        };
        let col_indices: Vec<usize> = ordered_cols
            .iter()
            .filter_map(|name| extended_schema.get_field_index(name))
            .collect();
        result_rows = result_rows
            .into_iter()
            .map(|row| {
                let vals: Vec<Value> = col_indices.iter().map(|&i| row.values[i].clone()).collect();
                let sel_fields: Vec<crate::core::tuple::Field> = col_indices
                    .iter()
                    .map(|&i| extended_schema.fields[i].clone())
                    .collect();
                let sel_schema = std::sync::Arc::new(crate::core::tuple::Schema::new(sel_fields));
                Tuple::new(sel_schema, vals).unwrap_or(row)
            })
            .collect();
        let final_fields: Vec<crate::core::tuple::Field> = col_indices
            .iter()
            .map(|&i| extended_schema.fields[i].clone())
            .collect();
        std::sync::Arc::new(crate::core::tuple::Schema::new(final_fields))
    } else {
        base_schema
    };

    // Handle UNION
    let (result_rows, result_schema) = if let Some((kind, right_stmt)) = s.union {
        let right_result = execute_select(db, *right_stmt, line_no)?;
        if let DslOutput::Table(right_ds) = right_result {
            let mut combined = result_rows;
            combined.extend(right_ds.rows);
            let final_rows = if matches!(kind, SetOpKind::Union) {
                // Deduplicate
                let mut seen = std::collections::HashSet::new();
                combined
                    .into_iter()
                    .filter(|row| seen.insert(format!("{:?}", row.values)))
                    .collect()
            } else {
                combined
            };
            (final_rows, result_schema)
        } else {
            (result_rows, result_schema)
        }
    } else {
        (result_rows, result_schema)
    };

    // Clean up CTE temp datasets so they don't shadow real datasets in subsequent queries
    for cte_name in &cte_names {
        let _ = db.remove_dataset(cte_name);
    }

    // Normalize all rows to the canonical result_schema Arc so that Dataset::with_rows
    // schema equality check (structural, not pointer) passes even for rows from different
    // query results (UNION, window/computed column extensions, etc.)
    let result_rows: Vec<Tuple> = result_rows
        .into_iter()
        .map(|row| {
            if row.values.len() == result_schema.fields.len() {
                let vals = row.values.clone();
                Tuple::new(result_schema.clone(), vals).unwrap_or(row)
            } else {
                row
            }
        })
        .collect();

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

fn infer_expr_result_type(expr: &Expr) -> ValueType {
    match expr {
        Expr::Int(_) => ValueType::Int,
        Expr::Scalar(_) => ValueType::Float,
        Expr::StringLit(_) => ValueType::String,
        Expr::Bool(_) => ValueType::Bool,
        Expr::Ref(_) => ValueType::Float,
        Expr::Infix { op, lhs, rhs } => {
            let lt = infer_expr_result_type(lhs);
            let rt = infer_expr_result_type(rhs);
            match op {
                InfixOp::Add | InfixOp::Subtract | InfixOp::Multiply | InfixOp::Divide => {
                    match (lt, rt) {
                        (ValueType::Float, _) | (_, ValueType::Float) => ValueType::Float,
                        (ValueType::Int, ValueType::Int) => ValueType::Int,
                        _ => ValueType::Float,
                    }
                }
                _ => ValueType::Bool,
            }
        }
        Expr::ScalarFn {
            func: ScalarFnKind::Length,
            ..
        } => ValueType::Int,
        Expr::ScalarFn { .. } => ValueType::String,
        Expr::Cast { to, .. } => match to {
            CastTarget::Int => ValueType::Int,
            CastTarget::Float => ValueType::Float,
            CastTarget::Text => ValueType::String,
            CastTarget::Bool => ValueType::Bool,
            CastTarget::Vector(n) => ValueType::Vector(*n),
            CastTarget::Matrix(r, c) => ValueType::Matrix(*r, *c),
        },
        Expr::VecLiteral(v) => ValueType::Vector(v.len()),
        Expr::MatLiteral(_) => ValueType::Matrix(0, 0),
        Expr::VectorFn { func, .. } => match func {
            VectorFnKind::Normalize
            | VectorFnKind::VecAdd
            | VectorFnKind::VecScale
            | VectorFnKind::Flatten => ValueType::Vector(0),
            VectorFnKind::L2Norm | VectorFnKind::CosineSim | VectorFnKind::Dot => ValueType::Float,
            VectorFnKind::Matmul | VectorFnKind::Transpose => ValueType::Matrix(0, 0),
            VectorFnKind::MatShape => ValueType::String,
        },
        _ => ValueType::Float,
    }
}

fn apply_window_and_computed_exprs(
    mut rows: Vec<Tuple>,
    _base_schema: &std::sync::Arc<crate::core::tuple::Schema>,
    window_exprs: &[SelectExpr],
    line_no: usize,
) -> Result<Vec<Tuple>, DslError> {
    use crate::query::physical::evaluate_expression;

    let mut computed_idx = 0usize;
    for we in window_exprs {
        match we {
            SelectExpr::Computed { expr, alias } => {
                let temp_name = alias
                    .clone()
                    .unwrap_or_else(|| format!("__cmp_{}", computed_idx));
                computed_idx += 1;
                let logical_expr = dsl_expr_to_logical_expr(expr);
                let fallback_vtype = infer_expr_result_type(expr);
                rows = rows
                    .into_iter()
                    .map(|row| {
                        let val = evaluate_expression(&logical_expr, &row);
                        let actual_vtype = match val.value_type() {
                            ValueType::Null => fallback_vtype.clone(),
                            t => t,
                        };
                        let mut vals = row.values.clone();
                        vals.push(val);
                        let ext_schema = std::sync::Arc::new(crate::core::tuple::Schema::new(
                            row.schema
                                .fields
                                .iter()
                                .cloned()
                                .chain(std::iter::once(
                                    crate::core::tuple::Field::new(&temp_name, actual_vtype)
                                        .nullable(),
                                ))
                                .collect(),
                        ));
                        Tuple::new(ext_schema, vals).unwrap_or(row)
                    })
                    .collect();
            }
            SelectExpr::Window { func, spec, alias } => {
                rows = apply_window_func(rows, func, spec, alias, line_no)?;
            }
            _ => {}
        }
    }
    Ok(rows)
}

fn apply_window_func(
    rows: Vec<Tuple>,
    func: &WindowFunc,
    spec: &WindowSpec,
    alias: &str,
    line_no: usize,
) -> Result<Vec<Tuple>, DslError> {
    use crate::query::physical::evaluate_expression;

    if let Some(row) = rows.first() {
        for (col, _) in &spec.order_by {
            if let Some(field) = row.schema.get_field(col) {
                if matches!(
                    field.value_type,
                    ValueType::Vector(_) | ValueType::Matrix(_, _)
                ) {
                    return Err(DslError::Parse {
                        line: line_no,
                        msg: format!(
                            "Cannot ORDER BY column '{}' in a window function: Vector and Matrix \
                             values have no defined ordering. Sort by a scalar expression instead \
                             (e.g. a similarity/distance function).",
                            col
                        ),
                    });
                }
            }
        }
    }

    let n = rows.len();
    let mut result_vals: Vec<Value> = vec![Value::Null; n];

    // Group rows by partition key
    let partition_keys: Vec<String> = rows
        .iter()
        .map(|row| {
            spec.partition_by
                .iter()
                .map(|col| format!("{:?}", row.get(col)))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();

    // Collect unique partitions preserving order
    let mut partitions: Vec<String> = vec![];
    let mut seen_parts = std::collections::HashSet::new();
    for k in &partition_keys {
        if seen_parts.insert(k.clone()) {
            partitions.push(k.clone());
        }
    }

    for part_key in &partitions {
        let indices: Vec<usize> = (0..n).filter(|&i| &partition_keys[i] == part_key).collect();

        // Sort within partition if ORDER BY is specified
        let sorted_indices = if !spec.order_by.is_empty() {
            let mut si = indices.clone();
            si.sort_by(|&a, &b| {
                for (col, asc) in &spec.order_by {
                    let va = rows[a].get(col).cloned().unwrap_or(Value::Null);
                    let vb = rows[b].get(col).cloned().unwrap_or(Value::Null);
                    let ord = va.compare(&vb).unwrap_or(std::cmp::Ordering::Equal);
                    let ord = if *asc { ord } else { ord.reverse() };
                    if ord != std::cmp::Ordering::Equal {
                        return ord;
                    }
                }
                std::cmp::Ordering::Equal
            });
            si
        } else {
            indices.clone()
        };

        for (rank_0, &orig_idx) in sorted_indices.iter().enumerate() {
            let rank = rank_0 + 1;
            let val = match func {
                WindowFunc::RowNumber => Value::Int(rank as i64),
                WindowFunc::Rank => {
                    // Rank: same value → same rank, gaps
                    if rank_0 == 0 {
                        Value::Int(1)
                    } else {
                        let prev_idx = sorted_indices[rank_0 - 1];
                        let same = spec
                            .order_by
                            .iter()
                            .all(|(col, _)| rows[orig_idx].get(col) == rows[prev_idx].get(col));
                        if same {
                            result_vals[prev_idx].clone()
                        } else {
                            Value::Int(rank as i64)
                        }
                    }
                }
                WindowFunc::DenseRank => {
                    if rank_0 == 0 {
                        Value::Int(1)
                    } else {
                        let prev_idx = sorted_indices[rank_0 - 1];
                        let same = spec
                            .order_by
                            .iter()
                            .all(|(col, _)| rows[orig_idx].get(col) == rows[prev_idx].get(col));
                        if same {
                            result_vals[prev_idx].clone()
                        } else {
                            // dense rank = previous dense rank + 1
                            if let Value::Int(prev_dr) = &result_vals[prev_idx] {
                                Value::Int(prev_dr + 1)
                            } else {
                                Value::Int(rank as i64)
                            }
                        }
                    }
                }
                WindowFunc::Lag { col, offset } => {
                    if rank_0 < *offset {
                        Value::Null
                    } else {
                        let lag_idx = sorted_indices[rank_0 - offset];
                        rows[lag_idx].get(col).cloned().unwrap_or(Value::Null)
                    }
                }
                WindowFunc::Lead { col, offset } => {
                    if rank_0 + offset >= sorted_indices.len() {
                        Value::Null
                    } else {
                        let lead_idx = sorted_indices[rank_0 + offset];
                        rows[lead_idx].get(col).cloned().unwrap_or(Value::Null)
                    }
                }
                WindowFunc::Sum(inner) => {
                    // SUM_VEC/AVG_VEC collapse into this same WindowFunc::Sum/Avg
                    // at parse time (parser/dataset.rs), so this must handle
                    // Vector/Matrix element-wise, not just Int/Float — otherwise
                    // vector window aggregates silently zero out.
                    let logical = dsl_expr_to_logical_expr(inner);
                    let vals: Vec<Value> = sorted_indices[..=rank_0]
                        .iter()
                        .map(|&i| evaluate_expression(&logical, &rows[i]))
                        .collect();
                    window_running_sum(&vals, line_no)?
                }
                WindowFunc::Avg(inner) => {
                    let logical = dsl_expr_to_logical_expr(inner);
                    let vals: Vec<Value> = sorted_indices[..=rank_0]
                        .iter()
                        .map(|&i| evaluate_expression(&logical, &rows[i]))
                        .collect();
                    let count = vals.len().max(1) as f32;
                    match window_running_sum(&vals, line_no)? {
                        Value::Float(s) => Value::Float(s / count),
                        Value::Vector(v) => Value::Vector(v.iter().map(|x| x / count).collect()),
                        Value::Matrix(m) => Value::Matrix(
                            m.iter()
                                .map(|row| row.iter().map(|x| x / count).collect())
                                .collect(),
                        ),
                        other => other,
                    }
                }
                WindowFunc::Count(inner) => {
                    let logical = dsl_expr_to_logical_expr(inner);
                    let cnt = sorted_indices[..=rank_0]
                        .iter()
                        .filter(|&&i| {
                            !matches!(evaluate_expression(&logical, &rows[i]), Value::Null)
                        })
                        .count();
                    Value::Int(cnt as i64)
                }
                WindowFunc::Min(inner) => {
                    let logical = dsl_expr_to_logical_expr(inner);
                    sorted_indices[..=rank_0]
                        .iter()
                        .map(|&i| evaluate_expression(&logical, &rows[i]))
                        .filter(|v| !matches!(v, Value::Null))
                        .min_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(Value::Null)
                }
                WindowFunc::Max(inner) => {
                    let logical = dsl_expr_to_logical_expr(inner);
                    sorted_indices[..=rank_0]
                        .iter()
                        .map(|&i| evaluate_expression(&logical, &rows[i]))
                        .filter(|v| !matches!(v, Value::Null))
                        .max_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(Value::Null)
                }
            };
            result_vals[orig_idx] = val;
        }
    }

    // Infer value type from computed results (Value::value_type() already covers
    // Vector/Matrix correctly, unlike the old hand-rolled match here which
    // defaulted anything non-scalar to Int).
    let vtype = result_vals
        .iter()
        .find(|v| !matches!(v, Value::Null))
        .map(|v| v.value_type())
        .unwrap_or(ValueType::Int);

    // Append window result to each row using the alias as the column name
    rows.into_iter()
        .enumerate()
        .map(|(i, row)| {
            let mut vals = row.values.clone();
            vals.push(result_vals[i].clone());
            let new_fields: Vec<crate::core::tuple::Field> = row
                .schema
                .fields
                .iter()
                .cloned()
                .chain(std::iter::once(
                    crate::core::tuple::Field::new(alias, vtype.clone()).nullable(),
                ))
                .collect();
            let new_schema = std::sync::Arc::new(crate::core::tuple::Schema::new(new_fields));
            Tuple::new(new_schema, vals).map_err(|e| DslError::Parse {
                line: line_no,
                msg: format!("Failed to append window column '{}': {}", alias, e),
            })
        })
        .collect()
}

/// Element-wise running SUM over a window slice. Handles Int/Float/Vector/Matrix,
/// mirroring the vector-aware accumulation `AggregateExec` uses for grouped SUM/AVG
/// (`src/query/physical.rs`) — errors on dimension/shape mismatch instead of
/// silently dropping to `0.0`.
fn window_running_sum(vals: &[Value], line_no: usize) -> Result<Value, DslError> {
    let mut acc: Option<Value> = None;
    for v in vals.iter().cloned() {
        acc = Some(match (acc, v) {
            (None, Value::Int(n)) => Value::Float(n as f32),
            (None, Value::Float(f)) => Value::Float(f),
            (None, Value::Vector(vec)) => Value::Vector(vec),
            (None, Value::Matrix(m)) => Value::Matrix(m),
            (None, _) => Value::Float(0.0),
            (Some(Value::Float(s)), Value::Int(n)) => Value::Float(s + n as f32),
            (Some(Value::Float(s)), Value::Float(f)) => Value::Float(s + f),
            (Some(Value::Vector(mut sum)), Value::Vector(v2)) => {
                if sum.len() != v2.len() {
                    return Err(DslError::Parse {
                        line: line_no,
                        msg: format!(
                            "Window SUM/AVG: vector dimension mismatch — expected {}, got {}",
                            sum.len(),
                            v2.len()
                        ),
                    });
                }
                for (s, x) in sum.iter_mut().zip(v2.iter()) {
                    *s += x;
                }
                Value::Vector(sum)
            }
            (Some(Value::Matrix(mut sum)), Value::Matrix(m2)) => {
                let expected = (sum.len(), sum.first().map_or(0, |r| r.len()));
                let actual = (m2.len(), m2.first().map_or(0, |r| r.len()));
                if expected != actual {
                    return Err(DslError::Parse {
                        line: line_no,
                        msg: format!(
                            "Window SUM/AVG: matrix shape mismatch — expected {:?}, got {:?}",
                            expected, actual
                        ),
                    });
                }
                for i in 0..sum.len() {
                    for j in 0..sum[i].len() {
                        sum[i][j] += m2[i][j];
                    }
                }
                Value::Matrix(sum)
            }
            (Some(other), _) => other,
        });
    }
    Ok(acc.unwrap_or(Value::Float(0.0)))
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
        AggFuncAst::AvgVec => AggregateFunction::AvgVec,
        AggFuncAst::SumVec => AggregateFunction::SumVec,
    }
}

pub(super) fn dsl_expr_to_logical_expr(e: &Expr) -> LogicalExpr {
    match e {
        Expr::Ref(name) => LogicalExpr::Column(name.clone()),
        // `table.col` — the table qualifier is only meaningful for JOIN's
        // ON clause (which also strips it, see parse_join_col_ref); a
        // single row here is already the merged output of the JOIN, so
        // resolve by the bare column name, same as an unqualified Ref.
        Expr::Field { field, .. } => LogicalExpr::Column(field.clone()),
        Expr::Int(n) => LogicalExpr::Literal(Value::Int(*n)),
        Expr::Scalar(f) => LogicalExpr::Literal(Value::Float(*f as f32)),
        Expr::StringLit(s) => LogicalExpr::Literal(Value::String(s.clone())),
        Expr::Bool(b) => LogicalExpr::Literal(Value::Bool(*b)),
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
        Expr::In { expr, list } => LogicalExpr::In {
            expr: Box::new(dsl_expr_to_logical_expr(expr)),
            list: list.iter().map(dsl_expr_to_logical_expr).collect(),
        },
        Expr::Between { expr, low, high } => LogicalExpr::Between {
            expr: Box::new(dsl_expr_to_logical_expr(expr)),
            low: Box::new(dsl_expr_to_logical_expr(low)),
            high: Box::new(dsl_expr_to_logical_expr(high)),
        },
        Expr::Case {
            operand,
            branches,
            else_expr,
        } => LogicalExpr::Case {
            operand: operand
                .as_ref()
                .map(|e| Box::new(dsl_expr_to_logical_expr(e))),
            branches: branches
                .iter()
                .map(|(c, r)| (dsl_expr_to_logical_expr(c), dsl_expr_to_logical_expr(r)))
                .collect(),
            else_expr: else_expr
                .as_ref()
                .map(|e| Box::new(dsl_expr_to_logical_expr(e))),
        },
        Expr::Coalesce(args) => {
            LogicalExpr::Coalesce(args.iter().map(dsl_expr_to_logical_expr).collect())
        }
        Expr::Nullif(a, b) => LogicalExpr::Nullif(
            Box::new(dsl_expr_to_logical_expr(a)),
            Box::new(dsl_expr_to_logical_expr(b)),
        ),
        Expr::ScalarFn { func, args } => {
            use crate::query::logical::ScalarFnKind as LFnKind;
            let lfunc = match func {
                ScalarFnKind::Upper => LFnKind::Upper,
                ScalarFnKind::Lower => LFnKind::Lower,
                ScalarFnKind::Length => LFnKind::Length,
                ScalarFnKind::Trim => LFnKind::Trim,
                ScalarFnKind::Concat => LFnKind::Concat,
                ScalarFnKind::Substr => LFnKind::Substr,
            };
            LogicalExpr::ScalarFn {
                func: lfunc,
                args: args.iter().map(dsl_expr_to_logical_expr).collect(),
            }
        }
        Expr::Cast { expr, to } => {
            use crate::query::logical::CastTarget as LCast;
            let lto = match to {
                CastTarget::Int => LCast::Int,
                CastTarget::Float => LCast::Float,
                CastTarget::Text => LCast::Text,
                CastTarget::Bool => LCast::Bool,
                CastTarget::Vector(n) => LCast::Vector(*n),
                CastTarget::Matrix(r, c) => LCast::Matrix(*r, *c),
            };
            LogicalExpr::Cast {
                expr: Box::new(dsl_expr_to_logical_expr(expr)),
                to: lto,
            }
        }
        Expr::VecLiteral(vals) => {
            LogicalExpr::Literal(Value::Vector(vals.iter().map(|&v| v as f32).collect()))
        }
        Expr::MatLiteral(rows) => LogicalExpr::Literal(Value::Matrix(
            rows.iter()
                .map(|r| r.iter().map(|&v| v as f32).collect())
                .collect(),
        )),
        Expr::VectorFn { func, args } => {
            use crate::query::logical::VectorFnKind as LVk;
            let lfunc = match func {
                VectorFnKind::Normalize => LVk::Normalize,
                VectorFnKind::L2Norm => LVk::L2Norm,
                VectorFnKind::CosineSim => LVk::CosineSim,
                VectorFnKind::Dot => LVk::Dot,
                VectorFnKind::VecAdd => LVk::VecAdd,
                VectorFnKind::VecScale => LVk::VecScale,
                VectorFnKind::Matmul => LVk::Matmul,
                VectorFnKind::Transpose => LVk::Transpose,
                VectorFnKind::MatShape => LVk::MatShape,
                VectorFnKind::Flatten => LVk::Flatten,
            };
            LogicalExpr::VectorFn {
                func: lfunc,
                args: args.iter().map(dsl_expr_to_logical_expr).collect(),
            }
        }
        _ => LogicalExpr::Literal(Value::Null),
    }
}

// ─── TRANSFORM ────────────────────────────────────────────────────────────────

pub(super) fn execute_transform(
    db: &mut TensorDb,
    s: TransformStmt,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let select_stmt = SelectStmt {
        ctes: vec![],
        distinct: false,
        source: DatasetSource::Named(s.source.clone()),
        joins: vec![],
        columns: s.columns,
        filter: s.filter,
        group_by: vec![],
        having: None,
        order_by: None,
        limit: None,
        offset: None,
        union: None,
    };

    let result = execute_select(db, select_stmt, line_no)?;
    let DslOutput::Table(result_ds) = result else {
        return Err(DslError::Parse {
            line: line_no,
            msg: "TRANSFORM did not produce a table result".into(),
        });
    };

    let target_name = s.target.unwrap_or(s.source);
    let schema = result_ds.schema.clone();
    let rows = result_ds.rows;

    if db.get_dataset(&target_name).is_ok() {
        let ds = db
            .get_dataset_mut(&target_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        ds.rows = rows;
        ds.metadata.update_stats(&ds.schema, &ds.rows);
    } else {
        db.create_dataset(target_name.clone(), schema)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        let ds = db
            .get_dataset_mut(&target_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        ds.rows = rows;
        ds.metadata.update_stats(&ds.schema, &ds.rows);
    }

    Ok(DslOutput::Message(format!(
        "Transformed dataset '{}'.",
        target_name
    )))
}

// ─── UPDATE ───────────────────────────────────────────────────────────────────

pub(super) fn execute_update(
    db: &mut TensorDb,
    s: UpdateStmt,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    // Build a filter predicate (if any) using the same physical evaluator
    let predicate: Option<RowPredicate> = s.filter.as_ref().map(|f| -> RowPredicate {
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
    let predicate: Option<RowPredicate> = s.filter.as_ref().map(|f| -> RowPredicate {
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
        Expr::Bool(b) => Value::Bool(*b),
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
