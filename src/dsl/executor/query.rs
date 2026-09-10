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
    // Delegate to `execute_select` instead of re-deriving a LogicalPlan by
    // hand: the hand-rolled version here used to build its own Project/
    // Aggregate plan directly from `clause.select` and only ever kept
    // `SelectExpr::Column`/`Aggregate` entries, silently dropping any
    // `SelectExpr::Computed` (CASE WHEN, arithmetic, CAST, ...) or
    // `SelectExpr::Window` column from the materialized dataset with no
    // error at all -- e.g. `DATASET d FROM t SELECT a, CASE WHEN ... END AS
    // b` materialized `d` with only column `a`, `b` entirely missing.
    // `execute_select` already has the correct, tested post-processing for
    // both (`apply_window_and_computed_exprs`), so reuse it verbatim.
    let select_stmt = SelectStmt {
        ctes: vec![],
        distinct: false,
        source: DatasetSource::Named(clause.source),
        joins: vec![],
        columns: match clause.select {
            Some(exprs) => SelectColumns::Named(exprs),
            None => SelectColumns::All,
        },
        filter: clause.filter,
        group_by: clause.group_by,
        having: clause.having,
        order_by: clause.order_by,
        limit: clause.limit,
        offset: clause.offset,
        union: None,
    };

    let output = execute_select(db, select_stmt, line_no)?;
    let (result_schema, result_rows) = match output {
        DslOutput::Table(ds) => (ds.schema, ds.rows),
        _ => {
            return Err(DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::InvalidOp(
                    "DATASET ... FROM: inner SELECT did not produce a table".to_string(),
                ),
            })
        }
    };

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

    // `ORDER BY <alias>` where `<alias>` is a Computed/Window column (e.g.
    // `SELECT L2_NORM(v) AS energy FROM t ORDER BY energy`, or any `CASE
    // WHEN ... END AS x ... ORDER BY x`) can't be baked into the LogicalPlan
    // as a `Sort` here: that alias doesn't exist in any physical schema
    // until `apply_window_and_computed_exprs` appends it further down, so
    // `SortExec` failed with "Column not found for sorting" on every such
    // query -- previously the ONLY way to order by a derived column at all.
    // When `ORDER BY` names a column absent from the plan's own schema at
    // this point, defer both it and LIMIT/OFFSET to run on the final rows
    // after post-processing instead (see the bottom of this function).
    let mut deferred_order: Option<OrderByClause> = None;
    let mut deferred_limit: Option<(usize, usize)> = None;

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
            aggr_expr: aggr_exprs.clone(),
        };
        if let Some(having_expr) = &s.having {
            let schema_now = plan.schema();
            let predicate = resolve_having(having_expr, &aggr_exprs, &schema_now, line_no)?;
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate,
            };
        }
        if let Some(ord) = &s.order_by {
            let schema_now = plan.schema();
            if ord
                .columns
                .iter()
                .all(|(c, _)| schema_now.get_field_index(c).is_some())
            {
                plan = LogicalPlan::Sort {
                    input: Box::new(plan),
                    columns: ord.columns.clone(),
                };
            } else {
                deferred_order = Some(ord.clone());
            }
        }
        if deferred_order.is_none() {
            if let Some(n) = s.limit {
                plan = LogicalPlan::Limit {
                    input: Box::new(plan),
                    n,
                    offset: s.offset.unwrap_or(0),
                };
            }
        } else {
            deferred_limit = s.limit.map(|n| (n, s.offset.unwrap_or(0)));
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

        // Computed unconditionally (empty when `!has_aggr`) so a HAVING
        // clause on a plain, aggregate-free SELECT is still resolved and
        // validated against the real schema below, not just the has_aggr
        // case -- the same silent-failure gap applied there too before this
        // fix, since `resolve_having` handles an empty `aggr_exprs` fine.
        let aggr_exprs: Vec<LogicalExpr> = if has_aggr {
            match &s.columns {
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
            }
        } else {
            vec![]
        };
        if has_aggr {
            plan = LogicalPlan::Aggregate {
                input: Box::new(plan),
                group_expr: vec![],
                aggr_expr: aggr_exprs.clone(),
            };
        }

        if let Some(having_expr) = &s.having {
            let schema_now = plan.schema();
            let predicate = resolve_having(having_expr, &aggr_exprs, &schema_now, line_no)?;
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate,
            };
        }
        if let Some(ord) = &s.order_by {
            let schema_now = plan.schema();
            if ord
                .columns
                .iter()
                .all(|(c, _)| schema_now.get_field_index(c).is_some())
            {
                plan = LogicalPlan::Sort {
                    input: Box::new(plan),
                    columns: ord.columns.clone(),
                };
            } else {
                deferred_order = Some(ord.clone());
            }
        }
        if deferred_order.is_none() {
            if let Some(n) = s.limit {
                plan = LogicalPlan::Limit {
                    input: Box::new(plan),
                    n,
                    offset: s.offset.unwrap_or(0),
                };
            }
        } else {
            deferred_limit = s.limit.map(|n| (n, s.offset.unwrap_or(0)));
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
                // Aggregate fields land in base_schema after the GROUP BY key
                // fields (LogicalPlan::Aggregate::schema() always puts group
                // keys first), in the same relative order the Aggregate
                // SelectExpr entries appear here — look up the real output
                // name (honors G2's AS alias) instead of a placeholder, or
                // the column is silently dropped by the get_field_index
                // lookup a few lines down. This path isn't exclusive to the
                // no-GROUP-BY case: a plain qualified column with an alias
                // (e.g. `t.col AS col`) parses as SelectExpr::Computed, which
                // also makes window_exprs non-empty even when GROUP BY is
                // present — so the offset must account for group keys
                // whenever they exist, not just when this branch "normally"
                // runs. Previously started at 0 unconditionally, which
                // returned wrong/misaligned names (or silently dropped
                // columns) for any GROUP BY query that also had a Computed
                // item in its SELECT list.
                let mut agg_idx = s.group_by.len();
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

    // Apply the ORDER BY / LIMIT that had to be deferred past post-processing
    // because they target a Computed/Window alias (see where `deferred_order`
    // is set, above) -- now that `result_schema` includes that column.
    if let Some(ord) = &deferred_order {
        result_rows =
            crate::query::physical::sort_tuples(result_rows, &result_schema, &ord.columns)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
    }
    if let Some((n, offset)) = deferred_limit {
        result_rows = result_rows.into_iter().skip(offset).take(n).collect();
    }

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
                        // Float64 always wins the promotion, even against a
                        // plain Float, mirroring the runtime arithmetic rule.
                        (ValueType::Float64, _) | (_, ValueType::Float64) => ValueType::Float64,
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
            CastTarget::Double => ValueType::Float64,
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
            VectorFnKind::L2Norm
            | VectorFnKind::CosineSim
            | VectorFnKind::Dot
            | VectorFnKind::Distance => ValueType::Float,
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

                // Evaluate every row first so the whole column gets ONE
                // consistent declared type, chosen from the first non-null
                // actual value (falling back to the static guess only if
                // every row is null) -- mirrors the window-function path
                // below. Deciding this per-row instead (as a prior version
                // did) let, e.g., a NULL-producing row keep the naive
                // fallback type while other rows in the same column got
                // their real type, silently building rows with different
                // schemas for the same logical column and later failing
                // Dataset::with_rows's structural schema-equality check.
                let vals: Vec<Value> = rows
                    .iter()
                    .map(|row| evaluate_expression(&logical_expr, row))
                    .collect();
                let vtype = vals
                    .iter()
                    .find(|v| !matches!(v, Value::Null))
                    .map(|v| v.value_type())
                    .unwrap_or(fallback_vtype);

                rows = rows
                    .into_iter()
                    .zip(vals)
                    .map(|(row, val)| {
                        let mut new_vals = row.values.clone();
                        new_vals.push(val);
                        let ext_schema = std::sync::Arc::new(crate::core::tuple::Schema::new(
                            row.schema
                                .fields
                                .iter()
                                .cloned()
                                .chain(std::iter::once(
                                    crate::core::tuple::Field::new(&temp_name, vtype.clone())
                                        .nullable(),
                                ))
                                .collect(),
                        ));
                        Tuple::new(ext_schema, new_vals).unwrap_or(row)
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
                        Value::Float64(s) => Value::Float64(s / count as f64),
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
            (None, Value::Float64(f)) => Value::Float64(f),
            (None, Value::Vector(vec)) => Value::Vector(vec),
            (None, Value::Matrix(m)) => Value::Matrix(m),
            (None, _) => Value::Float(0.0),
            (Some(Value::Float(s)), Value::Int(n)) => Value::Float(s + n as f32),
            (Some(Value::Float(s)), Value::Float(f)) => Value::Float(s + f),
            // Once a Float64 has been seen (either as the running accumulator
            // or the incoming value), the accumulator promotes to Float64 and
            // never demotes back to f32.
            (Some(Value::Float64(s)), Value::Float64(f)) => Value::Float64(s + f),
            (Some(Value::Float64(s)), Value::Float(f)) => Value::Float64(s + f as f64),
            (Some(Value::Float64(s)), Value::Int(n)) => Value::Float64(s + n as f64),
            (Some(Value::Float(s)), Value::Float64(f)) => Value::Float64(s as f64 + f),
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
            Value::Float64(_) => ValueType::Float64,
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
            Value::Float64(_) => ValueType::Float64,
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

/// Recursively rewrites every `Expr::Ref(name)` in `expr` whose `name` is a
/// key in `rename`, replacing it with the mapped value; every other node is
/// cloned as-is. Used to resolve a `HAVING` clause's bare aggregate-call
/// references (e.g. `Expr::Ref("AVG(score)")`, produced by the parser's
/// aggregate-call special-case, see `dsl/parser/expr.rs`) against the SELECT
/// list's real (possibly aliased) output column name before lowering.
///
/// `Call`/`Index` (tensor-DSL constructs like `ADD a b`/`t[0,1]`) can't
/// plausibly appear inside a dataset-query HAVING boolean/comparison
/// predicate, so they're passed through unrewritten rather than recursing
/// into `CallExpr`'s own variants for a shape this rewrite never needs to
/// reach.
fn rewrite_ref_names(expr: &Expr, rename: &std::collections::HashMap<String, String>) -> Expr {
    match expr {
        Expr::Ref(name) => Expr::Ref(rename.get(name).cloned().unwrap_or_else(|| name.clone())),
        Expr::Int(_)
        | Expr::Scalar(_)
        | Expr::StringLit(_)
        | Expr::Bool(_)
        | Expr::DatasetRef(_)
        | Expr::VecLiteral(_)
        | Expr::MatLiteral(_)
        | Expr::Call(_)
        | Expr::Index { .. } => expr.clone(),
        Expr::Infix { op, lhs, rhs } => Expr::Infix {
            op: *op,
            lhs: Box::new(rewrite_ref_names(lhs, rename)),
            rhs: Box::new(rewrite_ref_names(rhs, rename)),
        },
        Expr::And(l, r) => Expr::And(
            Box::new(rewrite_ref_names(l, rename)),
            Box::new(rewrite_ref_names(r, rename)),
        ),
        Expr::Or(l, r) => Expr::Or(
            Box::new(rewrite_ref_names(l, rename)),
            Box::new(rewrite_ref_names(r, rename)),
        ),
        Expr::Not(e) => Expr::Not(Box::new(rewrite_ref_names(e, rename))),
        Expr::IsNull(e) => Expr::IsNull(Box::new(rewrite_ref_names(e, rename))),
        Expr::IsNotNull(e) => Expr::IsNotNull(Box::new(rewrite_ref_names(e, rename))),
        Expr::In { expr, list } => Expr::In {
            expr: Box::new(rewrite_ref_names(expr, rename)),
            list: list.iter().map(|e| rewrite_ref_names(e, rename)).collect(),
        },
        Expr::Between { expr, low, high } => Expr::Between {
            expr: Box::new(rewrite_ref_names(expr, rename)),
            low: Box::new(rewrite_ref_names(low, rename)),
            high: Box::new(rewrite_ref_names(high, rename)),
        },
        Expr::Field { base, field } => Expr::Field {
            base: Box::new(rewrite_ref_names(base, rename)),
            field: field.clone(),
        },
        Expr::Case {
            operand,
            branches,
            else_expr,
        } => Expr::Case {
            operand: operand
                .as_ref()
                .map(|e| Box::new(rewrite_ref_names(e, rename))),
            branches: branches
                .iter()
                .map(|(c, r)| (rewrite_ref_names(c, rename), rewrite_ref_names(r, rename)))
                .collect(),
            else_expr: else_expr
                .as_ref()
                .map(|e| Box::new(rewrite_ref_names(e, rename))),
        },
        Expr::Coalesce(args) => {
            Expr::Coalesce(args.iter().map(|e| rewrite_ref_names(e, rename)).collect())
        }
        Expr::Nullif(a, b) => Expr::Nullif(
            Box::new(rewrite_ref_names(a, rename)),
            Box::new(rewrite_ref_names(b, rename)),
        ),
        Expr::ScalarFn { func, args } => Expr::ScalarFn {
            func: *func,
            args: args.iter().map(|e| rewrite_ref_names(e, rename)).collect(),
        },
        Expr::Cast { expr, to } => Expr::Cast {
            expr: Box::new(rewrite_ref_names(expr, rename)),
            to: *to,
        },
        Expr::VectorFn { func, args } => Expr::VectorFn {
            func: *func,
            args: args.iter().map(|e| rewrite_ref_names(e, rename)).collect(),
        },
    }
}

/// Collects every column name a lowered `LogicalExpr` predicate actually
/// references, so callers can validate them against a real output schema
/// (see `resolve_having` below) instead of letting an unresolvable
/// reference silently evaluate to `false` at row-filtering time.
fn collect_referenced_columns(expr: &LogicalExpr, out: &mut Vec<String>) {
    match expr {
        LogicalExpr::Column(name) => out.push(name.clone()),
        LogicalExpr::Literal(_) | LogicalExpr::VecLiteral(_) | LogicalExpr::MatLiteral(_) => {}
        LogicalExpr::BinaryExpr { left, right, .. } => {
            collect_referenced_columns(left, out);
            collect_referenced_columns(right, out);
        }
        LogicalExpr::And(l, r) | LogicalExpr::Or(l, r) => {
            collect_referenced_columns(l, out);
            collect_referenced_columns(r, out);
        }
        LogicalExpr::Not(e) | LogicalExpr::IsNull(e) | LogicalExpr::IsNotNull(e) => {
            collect_referenced_columns(e, out)
        }
        LogicalExpr::In { expr, list } => {
            collect_referenced_columns(expr, out);
            list.iter().for_each(|e| collect_referenced_columns(e, out));
        }
        LogicalExpr::Between { expr, low, high } => {
            collect_referenced_columns(expr, out);
            collect_referenced_columns(low, out);
            collect_referenced_columns(high, out);
        }
        LogicalExpr::AggregateExpr { expr, .. } => collect_referenced_columns(expr, out),
        LogicalExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            if let Some(o) = operand {
                collect_referenced_columns(o, out);
            }
            for (c, r) in branches {
                collect_referenced_columns(c, out);
                collect_referenced_columns(r, out);
            }
            if let Some(e) = else_expr {
                collect_referenced_columns(e, out);
            }
        }
        LogicalExpr::Coalesce(args) | LogicalExpr::ScalarFn { args, .. } => {
            args.iter().for_each(|e| collect_referenced_columns(e, out))
        }
        LogicalExpr::VectorFn { args, .. } => {
            args.iter().for_each(|e| collect_referenced_columns(e, out))
        }
        LogicalExpr::Nullif(a, b) => {
            collect_referenced_columns(a, out);
            collect_referenced_columns(b, out);
        }
        LogicalExpr::Cast { expr, .. } => collect_referenced_columns(expr, out),
    }
}

/// Resolves and lowers a `HAVING` clause's AST expression into a
/// `LogicalExpr` predicate, fixing two related silent-correctness gaps:
///
/// 1. The parser lowers a bare aggregate call (`AVG(score)`) anywhere in an
///    expression -- including HAVING -- to `Expr::Ref("AVG(score)")`, a
///    literal column-name lookup (`dsl/parser/expr.rs`). That string only
///    matches the aggregate's real output column when the SELECT list left
///    it unaliased; `AVG(score) AS avg_score` renames the real column to
///    `avg_score`, so `HAVING AVG(score) > 0.5` would silently reference a
///    column that no longer exists. This rewrites any such reference to the
///    aggregate's actual output name first.
/// 2. Once rewritten, every column the predicate still references is
///    checked against `schema_now` (the Aggregate plan's real output
///    schema) -- a genuinely unknown column now raises a clear error
///    instead of silently building a predicate that evaluates to `false`
///    for every row (`query::planner::eval_value` returns `None` for an
///    unresolvable column, and a `None` operand in a comparison silently
///    becomes `false`, not an error).
fn resolve_having(
    having_expr: &Expr,
    aggr_exprs: &[LogicalExpr],
    schema_now: &crate::core::tuple::Schema,
    line: usize,
) -> Result<LogicalExpr, DslError> {
    let mut rename = std::collections::HashMap::new();
    for a in aggr_exprs {
        if let LogicalExpr::AggregateExpr {
            func,
            expr: inner,
            alias: Some(alias),
        } = a
        {
            let default_name = crate::query::logical::aggregate_default_name(func, inner);
            if &default_name != alias {
                rename.insert(default_name, alias.clone());
            }
        }
    }

    let rewritten = rewrite_ref_names(having_expr, &rename);
    let predicate = dsl_expr_to_logical_expr(&rewritten);

    let mut referenced = Vec::new();
    collect_referenced_columns(&predicate, &mut referenced);
    for name in &referenced {
        if schema_now.get_field_index(name).is_none() {
            let available: Vec<&str> = schema_now.fields.iter().map(|f| f.name.as_str()).collect();
            return Err(DslError::Engine {
                line,
                source: crate::engine::EngineError::InvalidOp(format!(
                    "HAVING references unknown column '{}' -- available: {}",
                    name,
                    available.join(", ")
                )),
            });
        }
    }

    Ok(predicate)
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
            // A bare numeric literal cast straight to DOUBLE must skip the
            // generic recursive lowering below: `Expr::Scalar`'s own arm
            // (above) always narrows to `Value::Float(f32)`, so by the time
            // a Cast-to-Double evaluator would widen it back to f64, the
            // literal's real precision is already gone. Same "check the
            // target type before narrowing" idiom already used by
            // INSERT/ALTER...DEFAULT (dsl/executor/mod.rs).
            if matches!(to, CastTarget::Double) {
                if let Expr::Scalar(f) = expr.as_ref() {
                    return LogicalExpr::Literal(Value::Float64(*f));
                }
            }
            let lto = match to {
                CastTarget::Int => LCast::Int,
                CastTarget::Float => LCast::Float,
                CastTarget::Double => LCast::Double,
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
                VectorFnKind::Distance => LVk::Distance,
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
            // Any pairing touching Float64 promotes to Float64 (widening the
            // other side), checked before the plain-f32/Int arms below so it
            // always takes priority over them.
            if matches!(l, Value::Float64(_)) || matches!(r, Value::Float64(_)) {
                let (Some(a), Some(b)) = (l.as_float64(), r.as_float64()) else {
                    return Value::Null;
                };
                return match op {
                    InfixOp::Add => Value::Float64(a + b),
                    InfixOp::Subtract => Value::Float64(a - b),
                    InfixOp::Multiply => Value::Float64(a * b),
                    InfixOp::Divide => Value::Float64(a / b),
                    _ => Value::Null,
                };
            }
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
        // TODO(known gap, flagged not fixed): this wildcard also swallows
        // `Expr::Cast` — a computed/LAZY column (`ADD COLUMN x = CAST(...)
        // [LAZY]`) silently evaluates to Value::Null for *any* CAST target,
        // not just DOUBLE. Different root cause from the Cast-arm precision
        // fix in `dsl_expr_to_logical_expr` above (missing feature here, not
        // a narrowing bug) and a materially larger fix (needs a full
        // CastTarget match mirroring `query::physical::evaluate_expression`).
        // See CHANGELOG.md's "Flagged, not fixed" note.
        _ => Value::Null,
    }
}
