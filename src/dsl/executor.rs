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
        Statement::CreateDatabase(s) => handlers::instance::handle_create_database(
            db,
            &format!("CREATE DATABASE {}", s.name),
            line_no,
        ),

        Statement::DropDatabase(s) => handlers::instance::handle_drop_database(
            db,
            &format!("DROP DATABASE {}", s.name),
            line_no,
        ),

        Statement::UseDatabase(s) => {
            handlers::instance::handle_use_database(db, &format!("USE {}", s.name), line_no)
        }

        // ── Introspection ───────────────────────────────────────────────────
        Statement::Show(s) => {
            handlers::introspection::handle_show(db, &show_to_string(&s), line_no)
        }

        Statement::Explain(s) => {
            handlers::explain::handle_explain(db, &format!("EXPLAIN {}", s.target), line_no)
        }

        Statement::Audit(s) => {
            handlers::audit::handle_audit(db, &format!("AUDIT {}", s.target), line_no)
        }

        // ── Dataset operations ──────────────────────────────────────────────
        Statement::CreateDataset(s) => {
            if let Some(src) = s.from {
                // DATASET name FROM source — query variant, still uses legacy handler
                handlers::dataset::handle_dataset(
                    db,
                    &format!("DATASET {} FROM \"{}\"", s.name, src),
                    line_no,
                )
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
                db.alter_dataset_add_column(
                    &s.dataset,
                    col_def.name.clone(),
                    col_type_to_value_type(&col_def.col_type),
                    Value::Null,
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
            let col_map: std::collections::HashMap<&str, &InsertValue> =
                s.row.iter().map(|(k, v)| (k.as_str(), v)).collect();
            let values: Vec<Value> = schema
                .fields
                .iter()
                .map(|f| match col_map.get(f.name.as_str()) {
                    Some(InsertValue::Scalar(n)) => match f.value_type {
                        ValueType::Int => Value::Int(*n as i64),
                        _ => Value::Float(*n as f32),
                    },
                    Some(InsertValue::Text(t)) => Value::String(t.clone()),
                    Some(InsertValue::TensorRef(_)) | Some(InsertValue::Null) | None => Value::Null,
                })
                .collect();
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
            // Fall back to string-reconstruction for GROUP BY / aggregate queries
            // since SelectColumns::Named only carries strings, not aggregate exprs.
            if !s.group_by.is_empty() {
                return handlers::dataset::handle_select(db, &select_to_string(&s), line_no);
            }
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
            let cols = match s.columns {
                SelectColumns::All => source_schema
                    .fields
                    .iter()
                    .map(|f| f.name.clone())
                    .collect(),
                SelectColumns::Named(cs) => cs,
            };
            plan = LogicalPlan::Project {
                input: Box::new(plan),
                columns: cols,
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

        Statement::Deliver(s) => {
            handlers::dataset::handle_deliver(db, &deliver_to_string(&s), line_no)
        }

        // ── Persistence ─────────────────────────────────────────────────────
        Statement::Save(s) => handlers::persistence::handle_save(db, &save_to_string(&s), line_no),

        Statement::Load(s) => handlers::persistence::handle_load(db, &load_to_string(&s), line_no),

        Statement::List(s) => {
            handlers::persistence::handle_list_datasets(db, &list_to_string(&s), line_no)
        }

        Statement::Import(s) => {
            if s.ephemeral {
                handlers::persistence::handle_use_dataset(
                    db,
                    &format!("USE DATASET FROM \"{}\"", s.path),
                    line_no,
                )
            } else {
                handlers::persistence::handle_import(
                    db,
                    &format!("IMPORT DATASET FROM \"{}\"", s.path),
                    line_no,
                )
            }
        }

        Statement::Export(s) => handlers::persistence::handle_export(
            db,
            &format!("EXPORT {} TO \"{}\"", s.name, s.path),
            line_no,
        ),

        // ── Index ───────────────────────────────────────────────────────────
        Statement::CreateIndex(s) => handlers::index::handle_create_index(
            db,
            &format!("CREATE INDEX idx ON {}({})", s.dataset, s.column),
            line_no,
        ),

        // ── Metadata ────────────────────────────────────────────────────────
        Statement::SetMetadata(s) => handlers::metadata::handle_set_metadata(
            db,
            &format!("SET DATASET {} {} = \"{}\"", s.dataset, s.key, s.value),
            line_no,
        ),

        // ── Search ──────────────────────────────────────────────────────────
        Statement::Search(s) => handlers::search::handle_search(db, &search_to_string(&s), line_no),

        // ── Session ─────────────────────────────────────────────────────────
        Statement::Reset => handlers::session::handle_session(db, "RESET", line_no),
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
    }
}

// ─── Temp name generator ──────────────────────────────────────────────────────

fn fresh_temp(hint: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("_t_{}_{}", hint, n)
}

// ─── String reconstruction for delegated handlers ─────────────────────────────

fn show_to_string(s: &ShowStmt) -> String {
    let target = match &s.target {
        ShowTarget::All => "ALL".to_string(),
        ShowTarget::AllDatasets => "ALL DATASETS".to_string(),
        ShowTarget::AllDatabases => "DATABASES".to_string(),
        ShowTarget::Schema(n) => format!("SCHEMA {}", n),
        ShowTarget::Shape(n) => format!("SHAPE {}", n),
        ShowTarget::Lineage(n) => format!("LINEAGE {}", n),
        ShowTarget::Indexes(None) => "INDEXES".to_string(),
        ShowTarget::Indexes(Some(n)) => format!("INDEXES {}", n),
        ShowTarget::DatasetMetadata(n) => format!("DATASET METADATA {}", n),
        ShowTarget::DatasetVersions(n) => format!("DATASET VERSIONS {}", n),
        ShowTarget::StringLiteral(s) => format!("\"{}\"", s),
        ShowTarget::Named(n) => n.clone(),
    };
    format!("SHOW {}", target)
}

/// Convert a DSL `ColType` to the engine's `ValueType` without a string round-trip.
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

/// Convert a DSL `Expr` (used in WHERE / HAVING) to a `query::logical::Expr`.
/// Note: `InfixOp` only has arithmetic operators; comparison predicates in WHERE
/// are not reachable via the typed AST path and resolve to `Literal(Null)`.
fn dsl_expr_to_logical_expr(e: &Expr) -> LogicalExpr {
    match e {
        Expr::Ref(name) => LogicalExpr::Column(name.clone()),
        Expr::Scalar(f) => LogicalExpr::Literal(Value::Float(*f as f32)),
        Expr::StringLit(s) => LogicalExpr::Literal(Value::String(s.clone())),
        Expr::Infix { op, lhs, rhs } => {
            let sym = match op {
                InfixOp::Add => "+",
                InfixOp::Subtract => "-",
                InfixOp::Multiply => "*",
                InfixOp::Divide => "/",
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

fn select_to_string(s: &SelectStmt) -> String {
    let cols = match &s.columns {
        SelectColumns::All => "*".to_string(),
        SelectColumns::Named(cs) => cs.join(", "),
    };
    let mut q = format!("SELECT {} FROM {}", cols, s.dataset);
    if let Some(f) = &s.filter {
        q.push_str(&format!(" WHERE {}", expr_to_string(f)));
    }
    if !s.group_by.is_empty() {
        q.push_str(&format!(" GROUP BY {}", s.group_by.join(", ")));
    }
    if let Some(h) = &s.having {
        q.push_str(&format!(" HAVING {}", expr_to_string(h)));
    }
    if let Some(ord) = &s.order_by {
        q.push_str(&format!(
            " ORDER BY {} {}",
            ord.column,
            if ord.ascending { "ASC" } else { "DESC" }
        ));
    }
    if let Some(n) = s.limit {
        q.push_str(&format!(" LIMIT {}", n));
    }
    q
}

fn deliver_to_string(s: &DeliverStmt) -> String {
    match &s.path {
        Some(p) => format!("DELIVER {} TO \"{}\"", s.dataset, p),
        None => format!("DELIVER {}", s.dataset),
    }
}

fn save_to_string(s: &SaveStmt) -> String {
    let kind = match s.kind {
        PersistKind::Tensor => "TENSOR",
        PersistKind::Dataset => "DATASET",
    };
    match &s.path {
        Some(p) => format!("SAVE {} {} TO \"{}\"", kind, s.name, p),
        None => format!("SAVE {} {}", kind, s.name),
    }
}

fn load_to_string(s: &LoadStmt) -> String {
    let kind = match s.kind {
        PersistKind::Tensor => "TENSOR",
        PersistKind::Dataset => "DATASET",
    };
    match &s.path {
        Some(p) => format!("LOAD {} {} FROM \"{}\"", kind, s.name, p),
        None => format!("LOAD {} {}", kind, s.name),
    }
}

fn list_to_string(s: &ListStmt) -> String {
    match &s.target {
        ListTarget::Tensors => "LIST TENSORS".into(),
        ListTarget::Datasets => "LIST DATASETS".into(),
        ListTarget::DatasetVersions(n) => format!("LIST DATASET VERSIONS {}", n),
        ListTarget::DatasetPackages => "LIST DATASET PACKAGES".into(),
    }
}

fn search_to_string(s: &SearchStmt) -> String {
    let mut q = format!("SEARCH {}", s.query_tensor);
    if let Some(ref ds) = s.dataset {
        q.push_str(&format!(" IN {}", ds));
    }
    if let Some(k) = s.top_k {
        q.push_str(&format!(" TOP {}", k));
    }
    q
}

/// Serialize an `Expr` back to DSL text.  Used when delegating to a handler
/// that still does its own string parsing (e.g. SELECT WHERE clauses).
pub fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Ref(n) => n.clone(),
        Expr::Scalar(n) => format!("{}", n),
        Expr::StringLit(s) => format!("\"{}\"", s),
        Expr::Infix { op, lhs, rhs } => {
            let sym = match op {
                InfixOp::Add => "+",
                InfixOp::Subtract => "-",
                InfixOp::Multiply => "*",
                InfixOp::Divide => "/",
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
