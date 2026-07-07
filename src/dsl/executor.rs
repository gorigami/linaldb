use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::tensor::Shape;
use crate::core::tuple::{Field, Schema, Tuple};
use crate::core::value::{Value, ValueType};
use crate::dsl::ast::*;
use crate::dsl::{DslError, DslOutput};
use crate::engine::context::ExecutionContext;
use crate::engine::{BinaryOp, TensorDb, TensorKind, UnaryOp};
use crate::query::logical::{Expr as LogicalExpr, LogicalPlan};
use crate::query::planner::Planner;
use std::sync::Arc;

// ─── Entry point ──────────────────────────────────────────────────────────────

/// Execute a fully-parsed AST statement against the database.
///
/// This is the typed dispatch layer that replaces the `if/else if` string chain
/// in `execute_line_with_context`. Each variant calls the engine API directly or
/// delegates to an existing handler via a reconstructed canonical string for
/// statements whose handler logic is not yet ported.
pub fn execute_statement(
    db: &mut TensorDb,
    stmt: Statement,
    line_no: usize,
    ctx: Option<&mut ExecutionContext>,
) -> Result<DslOutput, DslError> {
    use crate::dsl::handlers;

    let mut local_ctx;
    let ctx: &mut ExecutionContext = match ctx {
        Some(c) => c,
        None => {
            local_ctx = ExecutionContext::new();
            &mut local_ctx
        }
    };

    match stmt {
        // ── Tensor construction ─────────────────────────────────────────────
        Statement::DefineTensor(s) => {
            let shape = Shape::new(s.shape);
            let values: Vec<f32> = s.values.iter().map(|&v| v as f32).collect();
            let kind = to_engine_kind(s.kind);
            db.insert_named_with_kind(s.name.clone(), shape, values, kind)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!("Defined tensor: {}", s.name)))
        }

        Statement::Vector(s) => {
            let n = s.values.len();
            let shape = Shape::new(vec![n]);
            let values: Vec<f32> = s.values.iter().map(|&v| v as f32).collect();
            db.insert_named_with_kind(s.name.clone(), shape, values, TensorKind::Normal)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!("Defined vector: {}", s.name)))
        }

        Statement::Matrix(s) => {
            let shape = Shape::new(vec![s.rows, s.cols]);
            let values: Vec<f32> = s.values.iter().map(|&v| v as f32).collect();
            db.insert_named_with_kind(s.name.clone(), shape, values, TensorKind::Normal)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!("Defined matrix: {}", s.name)))
        }

        // ── Assignment / computation ────────────────────────────────────────
        Statement::Let(s) => eval_let(db, ctx, &s.name, s.lazy, &s.expr, line_no),

        Statement::Derive(s) => eval_let(db, ctx, &s.name, false, &s.source_expr, line_no),

        // ── Zero-copy semantics ─────────────────────────────────────────────
        Statement::Bind(s) => {
            db.active_instance_mut()
                .bind_resource(&s.alias, &s.target)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!(
                "Bound alias '{}' to resource '{}'",
                s.alias, s.target
            )))
        }

        Statement::Attach(s) => {
            db.active_instance_mut()
                .add_column_to_tensor_dataset(&s.dataset, &s.column, &s.tensor)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!(
                "Attached tensor '{}' as column '{}' in dataset '{}'",
                s.tensor, s.column, s.dataset
            )))
        }

        // ── Database management ─────────────────────────────────────────────
        Statement::CreateDatabase(s) => {
            if s.if_not_exists && db.list_databases().contains(&s.name) {
                return Ok(DslOutput::Message(format!(
                    "Database '{}' already exists (skipped)",
                    s.name
                )));
            }
            db.create_database(s.name.clone())
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!("Created database: {}", s.name)))
        }

        Statement::DropDatabase(s) => {
            if s.if_exists && !db.list_databases().contains(&s.name) {
                return Ok(DslOutput::Message(format!(
                    "Database '{}' not found (skipped)",
                    s.name
                )));
            }
            db.drop_database(&s.name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            Ok(DslOutput::Message(format!("Dropped database: {}", s.name)))
        }

        Statement::UseDatabase(s) => {
            db.use_database(&s.name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            Ok(DslOutput::Message(format!(
                "Switched to database '{}'",
                s.name
            )))
        }

        // ── Introspection ───────────────────────────────────────────────────
        Statement::Show(s) => execute_show(db, s.target, line_no),

        Statement::Explain(s) => execute_explain(db as &TensorDb, s.target, line_no),

        Statement::Audit(s) => {
            let issues = db
                .verify_tensor_dataset(&s.target)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            if issues.is_empty() {
                Ok(DslOutput::Message(format!(
                    "Audit PASSED for dataset '{}'. All column references are valid.",
                    s.target
                )))
            } else {
                Ok(DslOutput::Message(format!(
                    "Audit FAILED for dataset '{}'. The following columns point to missing or invalid tensors: {:?}",
                    s.target, issues
                )))
            }
        }

        // ── Dataset operations ──────────────────────────────────────────────
        Statement::CreateDataset(s) => {
            if let Some(clause) = s.from {
                // DATASET name FROM source [clauses] — build LogicalPlan directly
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
                                    SelectExpr::Aggregate { func, expr } => {
                                        Some(LogicalExpr::AggregateExpr {
                                            func: agg_func_to_logical(func),
                                            expr: Box::new(dsl_expr_to_logical_expr(expr)),
                                        })
                                    }
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
                                SelectExpr::Aggregate { func, expr } => {
                                    Some(LogicalExpr::AggregateExpr {
                                        func: agg_func_to_logical(&func),
                                        expr: Box::new(dsl_expr_to_logical_expr(&expr)),
                                    })
                                }
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
                let physical_plan =
                    planner
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

                db.create_dataset(s.name.clone(), result_schema)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                let target_ds = db.get_dataset_mut(&s.name).map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
                target_ds.rows = result_rows;
                target_ds
                    .metadata
                    .update_stats(&target_ds.schema, &target_ds.rows);
                Ok(DslOutput::None)
            } else {
                let fields: Vec<Field> = s
                    .columns
                    .iter()
                    .map(|c| Field::new(&c.name, col_type_to_value_type(&c.col_type)))
                    .collect();
                let schema = Arc::new(Schema::new(fields));
                db.create_dataset(s.name.clone(), schema)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                Ok(DslOutput::Message(format!("Created dataset: {}", s.name)))
            }
        }

        Statement::AlterDataset(s) => match s.operation {
            AlterOp::AddColumn(col_def) => {
                let vtype = col_type_to_value_type(&col_def.col_type);
                let default_value = match col_def.default_val {
                    Some(FilterValue::Int(n)) => Value::Int(n),
                    Some(FilterValue::Float(f)) => Value::Float(f as f32),
                    Some(FilterValue::Str(s)) => Value::String(s),
                    Some(FilterValue::Bool(b)) => Value::Bool(b),
                    None if col_def.nullable => Value::Null,
                    None => match &vtype {
                        ValueType::Int => Value::Int(0),
                        ValueType::Float => Value::Float(0.0),
                        ValueType::String => Value::String(String::new()),
                        ValueType::Bool => Value::Bool(false),
                        ValueType::Vector(dim) => Value::Vector(vec![0.0; *dim]),
                        ValueType::Matrix(r, c) => Value::Matrix(vec![vec![0.0; *c]; *r]),
                        ValueType::Null => Value::Null,
                    },
                };
                db.alter_dataset_add_column(
                    &s.dataset,
                    col_def.name.clone(),
                    vtype,
                    default_value,
                    col_def.nullable,
                )
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
                Ok(DslOutput::Message(format!(
                    "Added column '{}' to dataset '{}'",
                    col_def.name, s.dataset
                )))
            }
            AlterOp::AddComputedColumn { name, expr, lazy } => {
                execute_add_computed_column(db, &s.dataset, &name, &expr, lazy, line_no)
            }
        },

        Statement::InsertInto(s) => {
            let schema = db
                .get_dataset(&s.dataset)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?
                .schema
                .clone();
            let values: Vec<Value> = match s.row {
                InsertRow::Named(named) => {
                    let col_map: std::collections::HashMap<&str, &InsertValue> =
                        named.iter().map(|(k, v)| (k.as_str(), v)).collect();
                    schema
                        .fields
                        .iter()
                        .map(|f| match col_map.get(f.name.as_str()) {
                            Some(InsertValue::Scalar(n)) => match f.value_type {
                                ValueType::Int => Value::Int(*n as i64),
                                _ => Value::Float(*n as f32),
                            },
                            Some(InsertValue::Text(t)) => Value::String(t.clone()),
                            Some(InsertValue::Vector(v)) => {
                                Value::Vector(v.iter().map(|&x| x as f32).collect())
                            }
                            Some(InsertValue::Matrix(m)) => Value::Matrix(
                                m.iter()
                                    .map(|row| row.iter().map(|&x| x as f32).collect())
                                    .collect(),
                            ),
                            Some(InsertValue::Bool(b)) => Value::Bool(*b),
                            Some(InsertValue::TensorRef(_)) | Some(InsertValue::Null) | None => {
                                Value::Null
                            }
                        })
                        .collect()
                }
                InsertRow::Positional(vals) => vals
                    .into_iter()
                    .zip(schema.fields.iter())
                    .map(|(v, f)| match v {
                        InsertValue::Scalar(n) => match f.value_type {
                            ValueType::Int => Value::Int(n as i64),
                            _ => Value::Float(n as f32),
                        },
                        InsertValue::Text(t) => Value::String(t),
                        InsertValue::Vector(v) => {
                            Value::Vector(v.into_iter().map(|x| x as f32).collect())
                        }
                        InsertValue::Matrix(m) => Value::Matrix(
                            m.into_iter()
                                .map(|row| row.into_iter().map(|x| x as f32).collect())
                                .collect(),
                        ),
                        InsertValue::Bool(b) => Value::Bool(b),
                        InsertValue::TensorRef(_) | InsertValue::Null => Value::Null,
                    })
                    .collect(),
            };
            let tuple = Tuple::new(schema.clone(), values).map_err(|e| DslError::Parse {
                line: line_no,
                msg: e,
            })?;
            db.insert_row(&s.dataset, tuple)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::None)
        }

        Statement::Select(s) => {
            let source_ds = db.get_dataset(&s.dataset).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let source_schema = source_ds.schema.clone();

            let mut plan = LogicalPlan::Scan {
                dataset_name: s.dataset.clone(),
                schema: source_schema.clone(),
            };

            if let Some(filter_expr) = &s.filter {
                plan = LogicalPlan::Filter {
                    input: Box::new(plan),
                    predicate: dsl_expr_to_logical_expr(filter_expr),
                };
            }

            if !s.group_by.is_empty() {
                // Build aggregate plan directly from typed AST
                let group_exprs: Vec<LogicalExpr> = s
                    .group_by
                    .iter()
                    .map(|c| LogicalExpr::Column(c.clone()))
                    .collect();
                let aggr_exprs: Vec<LogicalExpr> = match &s.columns {
                    SelectColumns::Named(exprs) => exprs
                        .iter()
                        .filter_map(|e| match e {
                            SelectExpr::Aggregate { func, expr } => {
                                Some(LogicalExpr::AggregateExpr {
                                    func: agg_func_to_logical(func),
                                    expr: Box::new(dsl_expr_to_logical_expr(expr)),
                                })
                            }
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
                let cols: Vec<String> = match s.columns {
                    SelectColumns::All => source_schema
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
            let physical_plan =
                planner
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
            let ds = crate::core::dataset_legacy::Dataset::with_rows(
                crate::core::dataset_legacy::DatasetId(0),
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

        Statement::Materialize(s) => {
            let dataset_name = if let Some(dot) = s.target.find('.') {
                s.target[..dot].to_string()
            } else {
                s.target.clone()
            };
            db.materialize_lazy_columns(&dataset_name)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!(
                "Materialized lazy columns in dataset '{}'",
                dataset_name
            )))
        }

        Statement::Deliver(s) => Ok(DslOutput::Message(format!(
            "Delivery Projection for '{}' created. (Phase 1 Read-Only View)",
            s.dataset
        ))),

        // ── Persistence ─────────────────────────────────────────────────────
        Statement::Save(s) => {
            handlers::persistence::save_typed(db, s.kind, &s.name, s.path.as_deref(), line_no)
        }

        Statement::Load(s) => {
            handlers::persistence::load_typed(db, s.kind, &s.name, s.path.as_deref(), line_no)
        }

        Statement::List(s) => handlers::persistence::list_typed(db, &s.target, line_no),

        Statement::Import(s) => handlers::persistence::import_typed(
            db,
            s.ephemeral,
            &s.path,
            s.name.as_deref(),
            line_no,
        ),

        Statement::ImportCsv(s) => {
            handlers::persistence::import_csv_typed(db, &s.path, s.name.as_deref(), line_no)
        }

        Statement::Export(s) => handlers::persistence::export_typed(db, &s.name, &s.path, line_no),

        // ── Index ───────────────────────────────────────────────────────────
        Statement::CreateIndex(s) => match s.kind {
            IndexKindAst::Default | IndexKindAst::Hash | IndexKindAst::BTree => {
                db.create_index(&s.dataset, &s.column)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                Ok(DslOutput::Message(format!(
                    "Created index on {}({})",
                    s.dataset, s.column
                )))
            }
            IndexKindAst::Vector => {
                db.create_vector_index(&s.dataset, &s.column)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                Ok(DslOutput::Message(format!(
                    "Created VECTOR index on {}({})",
                    s.dataset, s.column
                )))
            }
        },

        // ── Metadata ────────────────────────────────────────────────────────
        Statement::SetMetadata(s) => {
            handlers::metadata::set_metadata_typed(db, &s.dataset, &s.key, &s.value, line_no)
        }

        // ── Search ──────────────────────────────────────────────────────────
        Statement::Search(s) => {
            let source_ds = db.get_dataset(&s.dataset).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let schema = source_ds.schema.clone();
            let query_tensor = match s.query {
                SearchQuery::TensorRef(ref name) => db
                    .get(name)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?
                    .clone(),
                SearchQuery::Inline(ref values) => {
                    use crate::core::tensor::{TensorId, TensorMetadata};
                    let vals_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                    let n = vals_f32.len();
                    let id = TensorId::new();
                    let meta = TensorMetadata::new(id, None);
                    crate::core::tensor::Tensor::new(
                        id,
                        crate::core::tensor::Shape::new(vec![n]),
                        vals_f32,
                        meta,
                    )
                    .map_err(|e| DslError::Parse {
                        line: line_no,
                        msg: e,
                    })?
                }
            };
            let plan = LogicalPlan::VectorSearch {
                input: Box::new(LogicalPlan::Scan {
                    dataset_name: s.dataset.clone(),
                    schema,
                }),
                column: s.column.clone(),
                query: query_tensor,
                k: s.top_k,
            };
            let planner = Planner::new(db);
            let physical_plan =
                planner
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
            let row_count = result_rows.len();
            let target = s.target.unwrap_or_else(|| "search_results".to_string());
            if let Ok(ds) = db.get_dataset_mut(&target) {
                ds.rows = result_rows;
                ds.metadata.update_stats(&ds.schema, &ds.rows);
            } else {
                db.create_dataset(target.clone(), result_schema.clone())
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                let ds = db.get_dataset_mut(&target).map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
                ds.rows = result_rows;
                ds.metadata.update_stats(&ds.schema, &ds.rows);
            }
            Ok(DslOutput::Message(format!(
                "Search completed. Found {} results in '{}'.",
                row_count, target
            )))
        }

        // ── Session ─────────────────────────────────────────────────────────
        Statement::Reset => {
            db.reset_session();
            Ok(DslOutput::Message(
                "Session reset complete. All in-memory data has been cleared from the active database.".to_string(),
            ))
        }
    }
}

// ─── Let / expression evaluation ──────────────────────────────────────────────

/// Evaluate a `LET` or `DERIVE` statement: compute `expr` and store the result
/// under `output_name`.
fn eval_let(
    db: &mut TensorDb,
    ctx: &mut ExecutionContext,
    output_name: &str,
    lazy: bool,
    expr: &Expr,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let result = eval_expr_to_name(db, ctx, output_name, expr, lazy, line_no)?;
    Ok(DslOutput::Message(if lazy {
        format!("Defined lazy variable: {}", result)
    } else {
        format!("Defined variable: {}", result)
    }))
}

/// Recursively evaluate `expr`, placing the result into `desired_name` when a
/// new tensor must be created.  Returns the name of the tensor holding the
/// result (which equals `desired_name` for computed exprs, or the referenced
/// name for bare `Ref` nodes).
fn eval_expr_to_name(
    db: &mut TensorDb,
    ctx: &mut ExecutionContext,
    desired_name: &str,
    expr: &Expr,
    lazy: bool,
    line_no: usize,
) -> Result<String, DslError> {
    let eng = |e| DslError::Engine {
        line: line_no,
        source: e,
    };

    match expr {
        // A bare name — no computation needed; just return the reference
        Expr::Ref(name) => Ok(name.clone()),

        // Numeric literal — materialize as a scalar tensor
        Expr::Int(n) => {
            db.insert_named(desired_name, Shape::new(vec![]), vec![*n as f32])
                .map_err(eng)?;
            Ok(desired_name.to_string())
        }
        Expr::Scalar(n) => {
            db.insert_named(desired_name, Shape::new(vec![]), vec![*n as f32])
                .map_err(eng)?;
            Ok(desired_name.to_string())
        }

        // String literals are not valid tensor expressions
        Expr::StringLit(_) => Err(DslError::Parse {
            line: line_no,
            msg: "string literal is not a valid tensor expression".into(),
        }),

        // Infix arithmetic: evaluate both sides into temps, then apply op
        Expr::Infix { op, lhs, rhs } => {
            let l_tmp = fresh_temp("l");
            let r_tmp = fresh_temp("r");
            let l = eval_expr_to_name(db, ctx, &l_tmp, lhs, false, line_no)?;
            let r = eval_expr_to_name(db, ctx, &r_tmp, rhs, false, line_no)?;
            let bin_op = infix_to_binary_op(*op);
            if lazy {
                db.eval_lazy_binary(ctx, desired_name, &l, &r, bin_op)
            } else {
                db.eval_binary(ctx, desired_name, &l, &r, bin_op)
            }
            .map_err(eng)?;
            Ok(desired_name.to_string())
        }

        // Named operation call
        Expr::Call(call) => {
            eval_call(db, ctx, desired_name, call, lazy, line_no)?;
            Ok(desired_name.to_string())
        }

        // Subscript: t[i, j]  /  t[0:5, *]
        Expr::Index { base, indices } => {
            let base_tmp = fresh_temp("base");
            let base_name = eval_expr_to_name(db, ctx, &base_tmp, base, false, line_no)?;
            apply_index(db, ctx, desired_name, &base_name, indices, line_no)?;
            Ok(desired_name.to_string())
        }

        // Field / column access: dataset.column
        Expr::Field { base, field } => {
            let base_tmp = fresh_temp("base");
            let base_name = eval_expr_to_name(db, ctx, &base_tmp, base, false, line_no)?;
            if db.get_dataset(&base_name).is_ok() || db.get_tensor_dataset(&base_name).is_some() {
                db.eval_column_access(ctx, desired_name, &base_name, field)
            } else {
                db.eval_field_access(ctx, desired_name, &base_name, field)
            }
            .map_err(eng)?;
            Ok(desired_name.to_string())
        }

        // dataset("name") constructor
        Expr::DatasetRef(name) => {
            let ds = crate::core::dataset::Dataset::new(name);
            db.register_tensor_dataset(ds.clone());
            db.register_dataset_var(desired_name.to_string(), name.clone());
            Ok(desired_name.to_string())
        }
    }
}

/// Evaluate a `CallExpr` (named prefix operation) and store the result under
/// `output_name`.
fn eval_call(
    db: &mut TensorDb,
    ctx: &mut ExecutionContext,
    output: &str,
    call: &CallExpr,
    lazy: bool,
    line_no: usize,
) -> Result<(), DslError> {
    let eng = |e| DslError::Engine {
        line: line_no,
        source: e,
    };

    // Resolve a sub-expression operand to a tensor name, creating a temp if needed
    macro_rules! operand {
        ($expr:expr, $hint:expr) => {{
            let tmp = fresh_temp($hint);
            eval_expr_to_name(db, ctx, &tmp, $expr, false, line_no)?
        }};
    }

    match call {
        // ── Two-operand ops ─────────────────────────────────────────────────
        CallExpr::Add(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_binary(ctx, output, &a, &b, BinaryOp::Add)
            } else {
                db.eval_binary(ctx, output, &a, &b, BinaryOp::Add)
            }
            .map_err(eng)
        }
        CallExpr::Subtract(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_binary(ctx, output, &a, &b, BinaryOp::Subtract)
            } else {
                db.eval_binary(ctx, output, &a, &b, BinaryOp::Subtract)
            }
            .map_err(eng)
        }
        CallExpr::Multiply(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_binary(ctx, output, &a, &b, BinaryOp::Multiply)
            } else {
                db.eval_binary(ctx, output, &a, &b, BinaryOp::Multiply)
            }
            .map_err(eng)
        }
        CallExpr::Divide(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_binary(ctx, output, &a, &b, BinaryOp::Divide)
            } else {
                db.eval_binary(ctx, output, &a, &b, BinaryOp::Divide)
            }
            .map_err(eng)
        }
        CallExpr::Correlate(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            db.eval_binary(ctx, output, &a, &b, BinaryOp::Correlate)
                .map_err(eng)
        }
        CallExpr::Similarity(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            db.eval_binary(ctx, output, &a, &b, BinaryOp::Similarity)
                .map_err(eng)
        }
        CallExpr::Distance(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            db.eval_binary(ctx, output, &a, &b, BinaryOp::Distance)
                .map_err(eng)
        }
        CallExpr::Matmul(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_matmul(ctx, output, &a, &b)
            } else {
                db.eval_matmul(ctx, output, &a, &b)
            }
            .map_err(eng)
        }

        // ── Single-operand ops ──────────────────────────────────────────────
        CallExpr::Normalize(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Normalize)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Normalize)
            }
            .map_err(eng)
        }
        CallExpr::Transpose(a) => {
            let a = operand!(a, "a");
            db.eval_unary(ctx, output, &a, UnaryOp::Transpose)
                .map_err(eng)
        }
        CallExpr::Flatten(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Flatten)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Flatten)
            }
            .map_err(eng)
        }
        CallExpr::Sum(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Sum)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Sum)
            }
            .map_err(eng)
        }
        CallExpr::Mean(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Mean)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Mean)
            }
            .map_err(eng)
        }
        CallExpr::Stdev(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Stdev)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Stdev)
            }
            .map_err(eng)
        }
        CallExpr::Scale { input, factor } => {
            let a = operand!(input, "a");
            let op = UnaryOp::Scale(*factor as f32);
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, op)
            } else {
                db.eval_unary(ctx, output, &a, op)
            }
            .map_err(eng)
        }
        CallExpr::Reshape { input, shape } => {
            let a = operand!(input, "a");
            let new_shape = Shape::new(shape.clone());
            db.eval_reshape(ctx, output, &a, new_shape).map_err(eng)
        }

        // ── N-ary ────────────────────────────────────────────────────────────
        CallExpr::Stack(operands) => {
            let names: Vec<String> = operands
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let tmp = fresh_temp(&format!("s{}", i));
                    eval_expr_to_name(db, ctx, &tmp, e, false, line_no)
                })
                .collect::<Result<_, _>>()?;
            let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
            db.eval_stack(ctx, output, name_refs, 0).map_err(eng)
        }
    }
}

/// Apply index or slice specs to `base_name`, storing result in `output`.
fn apply_index(
    db: &mut TensorDb,
    ctx: &mut ExecutionContext,
    output: &str,
    base_name: &str,
    indices: &[IndexSpec],
    line_no: usize,
) -> Result<(), DslError> {
    use crate::engine::kernels::SliceSpec;
    let eng = |e| DslError::Engine {
        line: line_no,
        source: e,
    };

    let specs: Vec<SliceSpec> = indices
        .iter()
        .map(|i| match i {
            IndexSpec::All => SliceSpec::All,
            IndexSpec::Index(n) => SliceSpec::Index(*n),
            IndexSpec::Range(s, e) => SliceSpec::Range(*s, *e),
        })
        .collect();

    let all_single = specs.iter().all(|s| matches!(s, SliceSpec::Index(_)));
    if all_single {
        let idx: Vec<usize> = specs
            .iter()
            .filter_map(|s| {
                if let SliceSpec::Index(n) = s {
                    Some(*n)
                } else {
                    None
                }
            })
            .collect();
        db.eval_index(ctx, output, base_name, idx).map_err(eng)
    } else {
        db.eval_slice(ctx, output, base_name, specs).map_err(eng)
    }
}

// ─── Conversion helpers ───────────────────────────────────────────────────────

fn to_engine_kind(k: TensorKindAst) -> TensorKind {
    match k {
        TensorKindAst::Normal => TensorKind::Normal,
        TensorKindAst::Strict => TensorKind::Strict,
        TensorKindAst::Lazy => TensorKind::Lazy,
    }
}

fn infix_to_binary_op(op: InfixOp) -> BinaryOp {
    match op {
        InfixOp::Add => BinaryOp::Add,
        InfixOp::Subtract => BinaryOp::Subtract,
        InfixOp::Multiply => BinaryOp::Multiply,
        InfixOp::Divide => BinaryOp::Divide,
        _ => unreachable!("comparison operators cannot appear in tensor expressions"),
    }
}

// ─── Temp name generator ──────────────────────────────────────────────────────

fn fresh_temp(hint: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("_t_{}_{}", hint, n)
}

// ─── Show — typed dispatch ─────────────────────────────────────────────────────

fn execute_show(
    db: &mut TensorDb,
    target: ShowTarget,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    match target {
        ShowTarget::All => {
            let mut names = db.list_names();
            names.sort();
            let mut output = String::from("--- ALL TENSORS ---\n");
            for name in names {
                if let Ok(t) = db.get(&name) {
                    output.push_str(&format!(
                        "{}: shape {:?}, len {}, data = {:?}\n",
                        name,
                        t.shape.dims,
                        t.data.len(),
                        t.data
                    ));
                }
            }
            output.push_str("-------------------");
            Ok(DslOutput::Message(output))
        }

        ShowTarget::AllDatasets => {
            let mut names = db.list_dataset_names();
            names.sort();
            let mut output = String::from("--- ALL DATASETS ---\n");
            for name in names {
                if let Ok(dataset) = db.get_dataset(&name) {
                    output.push_str(&format!(
                        "Dataset: {} (rows: {}, columns: {})\n",
                        name,
                        dataset.len(),
                        dataset.schema.len()
                    ));
                    for field in &dataset.schema.fields {
                        output.push_str(&format!("  - {}: {}\n", field.name, field.value_type));
                    }
                }
            }
            output.push_str("--------------------");
            Ok(DslOutput::Message(output))
        }

        ShowTarget::AllDatabases => {
            let mut names = db.list_databases();
            names.sort();
            let mut output = String::from("--- ALL DATABASES ---\n");
            for name in names {
                output.push_str(&format!("  - {}\n", name));
            }
            output.push_str("---------------------");
            Ok(DslOutput::Message(output))
        }

        ShowTarget::Indexes(filter) => {
            let indices = db.list_indices();
            let mut output = if let Some(ref ds_name) = filter {
                format!("--- INDICES FOR {} ---\n", ds_name)
            } else {
                String::from("--- ALL INDICES ---\n")
            };
            output.push_str(&format!(
                "{:<20} {:<20} {:<10}\n",
                "Dataset", "Column", "Type"
            ));
            output.push_str(&format!("{:-<52}\n", ""));
            let mut count = 0;
            for (ds, col, type_str) in indices {
                if let Some(ref ds_filter) = filter {
                    if &ds != ds_filter {
                        continue;
                    }
                }
                output.push_str(&format!("{:<20} {:<20} {:<10}\n", ds, col, type_str));
                count += 1;
            }
            output.push_str("-------------------");
            if count == 0 {
                if let Some(ref ds_name) = filter {
                    if db.get_dataset(ds_name).is_err() {
                        return Err(DslError::Engine {
                            line: line_no,
                            source: crate::engine::EngineError::NameNotFound(ds_name.clone()),
                        });
                    }
                }
            }
            Ok(DslOutput::Message(output))
        }

        ShowTarget::Lineage(name) => {
            let tree = db.get_lineage_tree(&name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let mut output = format!("Lineage for tensor '{}':\n", name);
            output.push_str(&format_lineage_tree(&tree, 0));
            Ok(DslOutput::Message(output))
        }

        ShowTarget::DatasetMetadata(dataset_name) => {
            if let Ok(dataset) = db.get_dataset(&dataset_name) {
                let metadata = &dataset.metadata;
                let mut output = format!(
                    "=== Dataset Metadata: {} (In-Memory/Legacy) ===\n",
                    dataset_name
                );
                output.push_str(&format!("Version: {}\n", metadata.version));
                output.push_str("Origin: Created\n");
                output.push_str(&format!("Created: {:?}\n", metadata.created_at));
                output.push_str(&format!("Updated: {:?}\n", metadata.updated_at));
                output.push_str(&format!("Rows: {}\n", metadata.row_count));
                if !metadata.extra.is_empty() {
                    output.push_str("\nExtra Metadata:\n");
                    for (k, v) in &metadata.extra {
                        output.push_str(&format!("  {}: {}\n", k, v));
                    }
                }
                output.push_str("================================");
                return Ok(DslOutput::Message(output));
            }
            if let Some(dataset) = db.get_tensor_dataset(&dataset_name) {
                if let Some(metadata) = &dataset.metadata {
                    let mut output = format!(
                        "=== Dataset Metadata: {} (In-Memory/Tensor) ===\n",
                        dataset_name
                    );
                    output.push_str(&format!("Version: {}\n", metadata.version));
                    output.push_str(&format!("Hash: {}\n", metadata.hash));
                    output.push_str(&format!("Origin: {:?}\n", metadata.origin));
                    if let Some(author) = &metadata.author {
                        output.push_str(&format!("Author: {}\n", author));
                    }
                    if let Some(desc) = &metadata.description {
                        output.push_str(&format!("Description: {}\n", desc));
                    }
                    if !metadata.tags.is_empty() {
                        output.push_str(&format!("Tags: {}\n", metadata.tags.join(", ")));
                    }
                    output.push_str(&format!("Created: {:?}\n", metadata.created_at));
                    output.push_str(&format!("Updated: {:?}\n", metadata.updated_at));
                    output.push_str("================================");
                    return Ok(DslOutput::Message(output));
                }
            }
            let path = format!(
                "{}/{}",
                db.config.storage.data_dir.to_string_lossy(),
                db.active_instance().name
            );
            let storage = crate::core::storage::ParquetStorage::new(&path);
            if !storage.metadata_exists(&dataset_name) {
                return Ok(DslOutput::Message(format!(
                    "No metadata found for dataset '{}'",
                    dataset_name
                )));
            }
            let metadata =
                storage
                    .load_dataset_metadata(&dataset_name)
                    .map_err(|e| DslError::Parse {
                        line: line_no,
                        msg: format!("Failed to load metadata: {}", e),
                    })?;
            let mut output = format!("=== Dataset Metadata: {} ===\n", metadata.name);
            output.push_str(&format!("Version: {}\n", metadata.version));
            output.push_str(&format!("Schema Version: {}\n", metadata.schema_version));
            output.push_str(&format!("Hash: {}\n", metadata.hash));
            output.push_str(&format!("Origin: {:?}\n", metadata.origin));
            if let Some(author) = &metadata.author {
                output.push_str(&format!("Author: {}\n", author));
            }
            if let Some(desc) = &metadata.description {
                output.push_str(&format!("Description: {}\n", desc));
            }
            if !metadata.tags.is_empty() {
                output.push_str(&format!("Tags: {}\n", metadata.tags.join(", ")));
            }
            output.push_str(&format!("Created: {:?}\n", metadata.created_at));
            output.push_str(&format!("Updated: {:?}\n", metadata.updated_at));
            output.push_str("================================");
            Ok(DslOutput::Message(output))
        }

        ShowTarget::DatasetVersions(dataset_name) => {
            let path = format!(
                "{}/{}",
                db.config.storage.data_dir.to_string_lossy(),
                db.active_instance().name
            );
            let storage = crate::core::storage::ParquetStorage::new(&path);
            if !storage.metadata_exists(&dataset_name) {
                return Ok(DslOutput::Message(format!(
                    "No metadata found for dataset '{}'",
                    dataset_name
                )));
            }
            let metadata =
                storage
                    .load_dataset_metadata(&dataset_name)
                    .map_err(|e| DslError::Parse {
                        line: line_no,
                        msg: format!("Failed to load metadata: {}", e),
                    })?;
            let mut output = format!("=== Version History for Dataset: {} ===\n", dataset_name);
            output.push_str(&format!("Current Version: {}\n", metadata.version));
            output.push_str(&format!(
                "Current Schema Version: {}\n",
                metadata.schema_version
            ));
            output.push_str("\nSchema History:\n");
            if metadata.schema_history.is_empty() {
                output.push_str("  (Initial schema only)\n");
            } else {
                for v in &metadata.schema_history {
                    output.push_str(&format!(
                        "  - v{}: {} columns, migration: {:?}\n",
                        v.version,
                        v.schema.columns.len(),
                        v.migration
                    ));
                }
            }
            output.push_str("================================");
            Ok(DslOutput::Message(output))
        }

        ShowTarget::Shape(name) => {
            let t = db.get(&name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            Ok(DslOutput::Message(format!(
                "SHAPE {}: {:?}\n",
                name, t.shape.dims
            )))
        }

        ShowTarget::Schema(name) => {
            if let Ok(dataset) = db.get_dataset(&name) {
                let mut output = format!("Schema for dataset '{}' (Legacy):\n", name);
                output.push_str(&format!(
                    "{:<20} {:<20} {:<10}\n",
                    "Field", "Type", "Nullable"
                ));
                output.push_str(&format!("{:-<52}\n", ""));
                for field in &dataset.schema.fields {
                    output.push_str(&format!(
                        "{:<20} {:<20} {:<10}\n",
                        field.name,
                        format!("{:?}", field.value_type),
                        field.nullable
                    ));
                }
                return Ok(DslOutput::Message(output));
            }
            if let Some(ds) = db.get_tensor_dataset(&name) {
                let mut output = format!("Schema for dataset '{}' (Tensor-First):\n", name);
                output.push_str(&format!(
                    "{:<20} {:<20} {:<10} {:<10}\n",
                    "Column", "Type", "Role", "Nullable"
                ));
                output.push_str(&format!("{:-<62}\n", ""));
                for col in &ds.schema.columns {
                    output.push_str(&format!(
                        "{:<20} {:<20} {:<10} {:<10}\n",
                        col.name,
                        format!("{}", col.value_type),
                        format!("{:?}", col.role),
                        col.nullable
                    ));
                }
                return Ok(DslOutput::Message(output));
            }
            Err(DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::DatasetNotFound(name),
            })
        }

        ShowTarget::StringLiteral(s) => Ok(DslOutput::Message(s)),

        ShowTarget::Named(name) => {
            let _ = db.evaluate(&name);
            if let Ok(t) = db.get(&name) {
                return Ok(DslOutput::Tensor(t.clone()));
            }
            if let Ok(dataset) = db.get_dataset(&name) {
                return Ok(DslOutput::Table(dataset.clone()));
            }
            if let Some(ds) = db.get_tensor_dataset(&name) {
                let health_info = db.verify_tensor_dataset(&name).unwrap_or_default();
                return Ok(DslOutput::TensorTable(ds.clone(), health_info));
            }
            Err(DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::NameNotFound(name),
            })
        }
    }
}

// ─── Explain — typed dispatch ──────────────────────────────────────────────────

pub fn execute_explain(
    db: &TensorDb,
    target: ExplainTarget,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let logical_plan = match target {
        ExplainTarget::Dataset(name) => {
            let source_ds = db.get_dataset(&name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let schema = source_ds.schema.clone();
            LogicalPlan::Scan {
                dataset_name: name,
                schema,
            }
        }

        ExplainTarget::DatasetQuery { name: _, from } => {
            let source_ds = db.get_dataset(&from.source).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let source_schema = source_ds.schema.clone();
            let mut plan = LogicalPlan::Scan {
                dataset_name: from.source.clone(),
                schema: source_schema,
            };
            if let Some(f) = from.filter {
                plan = LogicalPlan::Filter {
                    input: Box::new(plan),
                    predicate: dataset_filter_to_logical(&f),
                };
            }
            if !from.group_by.is_empty() {
                let group_exprs: Vec<LogicalExpr> = from
                    .group_by
                    .iter()
                    .map(|c| LogicalExpr::Column(c.clone()))
                    .collect();
                let aggr_exprs: Vec<LogicalExpr> = from
                    .select
                    .as_ref()
                    .map(|exprs| {
                        exprs
                            .iter()
                            .filter_map(|e| match e {
                                SelectExpr::Aggregate { func, expr } => {
                                    Some(LogicalExpr::AggregateExpr {
                                        func: agg_func_to_logical(func),
                                        expr: Box::new(dsl_expr_to_logical_expr(expr)),
                                    })
                                }
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
            } else if let Some(exprs) = from.select {
                let cols: Vec<String> = exprs
                    .into_iter()
                    .filter_map(|e| match e {
                        SelectExpr::Column(c) => Some(c),
                        SelectExpr::Aggregate { .. } => None,
                    })
                    .collect();
                if !cols.is_empty() {
                    plan = LogicalPlan::Project {
                        input: Box::new(plan),
                        columns: cols,
                    };
                }
            }
            if let Some(f) = from.having {
                plan = LogicalPlan::Filter {
                    input: Box::new(plan),
                    predicate: dataset_filter_to_logical(&f),
                };
            }
            if let Some(ord) = from.order_by {
                plan = LogicalPlan::Sort {
                    input: Box::new(plan),
                    column: ord.column,
                    ascending: ord.ascending,
                };
            }
            if let Some(n) = from.limit {
                plan = LogicalPlan::Limit {
                    input: Box::new(plan),
                    n,
                };
            }
            plan
        }

        ExplainTarget::Search(s) => {
            let source_ds = db.get_dataset(&s.dataset).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let schema = source_ds.schema.clone();
            let query_tensor = match s.query {
                SearchQuery::TensorRef(ref name) => db
                    .get(name)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?
                    .clone(),
                SearchQuery::Inline(ref values) => {
                    use crate::core::tensor::{TensorId, TensorMetadata};
                    let vals_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                    let n = vals_f32.len();
                    let id = TensorId::new();
                    let meta = TensorMetadata::new(id, None);
                    crate::core::tensor::Tensor::new(
                        id,
                        crate::core::tensor::Shape::new(vec![n]),
                        vals_f32,
                        meta,
                    )
                    .map_err(|e| DslError::Parse {
                        line: line_no,
                        msg: e,
                    })?
                }
            };
            LogicalPlan::VectorSearch {
                input: Box::new(LogicalPlan::Scan {
                    dataset_name: s.dataset.clone(),
                    schema,
                }),
                column: s.column.clone(),
                query: query_tensor,
                k: s.top_k,
            }
        }

        ExplainTarget::Select(s) => {
            let source_ds = db.get_dataset(&s.dataset).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let source_schema = source_ds.schema.clone();

            let mut plan = LogicalPlan::Scan {
                dataset_name: s.dataset.clone(),
                schema: source_schema.clone(),
            };

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
                            SelectExpr::Aggregate { func, expr } => {
                                Some(LogicalExpr::AggregateExpr {
                                    func: agg_func_to_logical(func),
                                    expr: Box::new(dsl_expr_to_logical_expr(expr)),
                                })
                            }
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
                let cols: Vec<String> = match s.columns {
                    SelectColumns::All => source_schema
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

            plan
        }
    };

    let planner = Planner::new(db);
    let physical_plan =
        planner
            .create_physical_plan(&logical_plan)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    let output = format!(
        "--- Logical Plan ---\n{:#?}\n\n--- Physical Plan ---\n{:#?}",
        logical_plan, physical_plan
    );
    Ok(DslOutput::Message(output))
}

fn format_lineage_tree(node: &crate::engine::LineageNode, indent: usize) -> String {
    let mut out = String::new();
    let indent_str = "  ".repeat(indent);
    let name_part = if let Some(name) = &node.name {
        format!(" ({})", name)
    } else {
        String::new()
    };
    out.push_str(&format!(
        "{}{}{} [{}]\n",
        indent_str, node.operation, name_part, node.tensor_id.0
    ));
    for input in &node.inputs {
        out.push_str(&format_lineage_tree(input, indent + 1));
    }
    out
}

/// Map `AggFuncAst` to the query engine's `AggregateFunction`.
fn agg_func_to_logical(f: &AggFuncAst) -> crate::query::logical::AggregateFunction {
    use crate::query::logical::AggregateFunction;
    match f {
        AggFuncAst::Sum => AggregateFunction::Sum,
        AggFuncAst::Avg => AggregateFunction::Avg,
        AggFuncAst::Count => AggregateFunction::Count,
        AggFuncAst::Min => AggregateFunction::Min,
        AggFuncAst::Max => AggregateFunction::Max,
    }
}

/// Convert a DSL `ColType` to the engine's `ValueType` without a string round-trip.
fn execute_add_computed_column(
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

    // Reject duplicate column names
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
        // Infer type by evaluating on the first row — required even for lazy columns
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

        // Lazy: add placeholder NULL column; engine evaluates on access
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

        // Evaluate expression for every row
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

        // Infer type from first computed value
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

/// Evaluate a typed `Expr` against a row's field values.
/// Returns `Value::Null` for unrecognised expressions.
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

fn col_type_to_value_type(ct: &ColType) -> ValueType {
    match ct {
        ColType::Int => ValueType::Int,
        ColType::Float => ValueType::Float,
        ColType::String => ValueType::String,
        ColType::Bool => ValueType::Bool,
        ColType::Vector(n) => ValueType::Vector(*n),
        ColType::Matrix(r, c) => ValueType::Matrix(*r, *c),
        ColType::Tensor(_) => ValueType::Float,
    }
}

/// Convert a typed `DatasetFilter` to a `query::logical::Expr` predicate.
fn dataset_filter_to_logical(f: &DatasetFilter) -> LogicalExpr {
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

/// Convert a DSL `Expr` (used in WHERE / HAVING) to a `query::logical::Expr`.
/// Note: `InfixOp` only has arithmetic operators; comparison predicates in WHERE
/// are not reachable via the typed AST path and resolve to `Literal(Null)`.
fn dsl_expr_to_logical_expr(e: &Expr) -> LogicalExpr {
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
        _ => LogicalExpr::Literal(Value::Null),
    }
}

/// Serialize an `Expr` back to DSL text.  Used when delegating to a handler
/// that still does its own string parsing (e.g. SELECT WHERE clauses).
pub fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Ref(n) => n.clone(),
        Expr::Int(n) => format!("{}", n),
        Expr::Scalar(n) => format!("{}", n),
        Expr::StringLit(s) => format!("\"{}\"", s),
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
            format!("{} {} {}", expr_to_string(lhs), sym, expr_to_string(rhs))
        }
        Expr::Field { base, field } => format!("{}.{}", expr_to_string(base), field),
        Expr::Index { base, indices } => {
            let idx: Vec<String> = indices
                .iter()
                .map(|i| match i {
                    IndexSpec::All => "*".into(),
                    IndexSpec::Index(n) => n.to_string(),
                    IndexSpec::Range(s, e) => format!("{}:{}", s, e),
                })
                .collect();
            format!("{}[{}]", expr_to_string(base), idx.join(", "))
        }
        Expr::Call(c) => call_to_string(c),
        Expr::DatasetRef(name) => format!("dataset(\"{}\")", name),
    }
}

fn call_to_string(c: &CallExpr) -> String {
    match c {
        CallExpr::Add(a, b) => format!("ADD {} {}", expr_to_string(a), expr_to_string(b)),
        CallExpr::Subtract(a, b) => format!("SUBTRACT {} {}", expr_to_string(a), expr_to_string(b)),
        CallExpr::Multiply(a, b) => format!("MULTIPLY {} {}", expr_to_string(a), expr_to_string(b)),
        CallExpr::Divide(a, b) => format!("DIVIDE {} {}", expr_to_string(a), expr_to_string(b)),
        CallExpr::Correlate(a, b) => {
            format!("CORRELATE {} WITH {}", expr_to_string(a), expr_to_string(b))
        }
        CallExpr::Similarity(a, b) => format!(
            "SIMILARITY {} WITH {}",
            expr_to_string(a),
            expr_to_string(b)
        ),
        CallExpr::Distance(a, b) => {
            format!("DISTANCE {} TO {}", expr_to_string(a), expr_to_string(b))
        }
        CallExpr::Matmul(a, b) => format!("MATMUL {} {}", expr_to_string(a), expr_to_string(b)),
        CallExpr::Normalize(a) => format!("NORMALIZE {}", expr_to_string(a)),
        CallExpr::Transpose(a) => format!("TRANSPOSE {}", expr_to_string(a)),
        CallExpr::Flatten(a) => format!("FLATTEN {}", expr_to_string(a)),
        CallExpr::Sum(a) => format!("SUM {}", expr_to_string(a)),
        CallExpr::Mean(a) => format!("MEAN {}", expr_to_string(a)),
        CallExpr::Stdev(a) => format!("STDEV {}", expr_to_string(a)),
        CallExpr::Scale { input, factor } => {
            format!("SCALE {} BY {}", expr_to_string(input), factor)
        }
        CallExpr::Reshape { input, shape } => {
            let d: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
            format!("RESHAPE {} TO [{}]", expr_to_string(input), d.join(", "))
        }
        CallExpr::Stack(ops) => {
            let names: Vec<String> = ops.iter().map(expr_to_string).collect();
            format!("STACK {}", names.join(" "))
        }
    }
}
