use crate::core::value::Value;
use crate::dsl::ast::*;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

use super::query::execute_select;

pub(super) fn execute_define_pipeline(
    db: &mut TensorDb,
    s: DefinePipelineStmt,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let name = s.name.clone();
    if s.steps.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Pipeline must have at least one step".into(),
        });
    }
    db.pipelines.insert(name.clone(), s.steps);
    Ok(DslOutput::Message(format!("Defined pipeline '{}'.", name)))
}

pub(super) fn execute_apply_pipeline(
    db: &mut TensorDb,
    s: ApplyPipelineStmt,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let steps = db
        .pipelines
        .get(&s.pipeline)
        .ok_or_else(|| DslError::Parse {
            line: line_no,
            msg: format!("Pipeline '{}' not found", s.pipeline),
        })?
        .clone();

    let mut current = s.source.clone();
    let step_count = steps.len();

    for (i, step) in steps.into_iter().enumerate() {
        let is_last = i == step_count - 1;
        let target = if is_last {
            s.into.clone().unwrap_or_else(|| s.source.clone())
        } else {
            format!("__pipeline_{}_step{}", s.pipeline, i)
        };

        match step {
            PipelineStep::NormalizeCol(col) => {
                apply_normalize_col(db, &current, &target, &col, line_no)?;
            }
            other => {
                let select_stmt = pipeline_step_to_select(other, current.clone());
                let result = execute_select(db, select_stmt, line_no)?;
                let DslOutput::Table(result_ds) = result else {
                    return Err(DslError::Parse {
                        line: line_no,
                        msg: format!("Pipeline step {} did not produce a table", i),
                    });
                };
                let schema = result_ds.schema.clone();
                let rows = result_ds.rows;
                upsert_dataset(db, &target, schema, rows, line_no)?;
            }
        }

        current = target;
    }

    let final_name = s.into.as_deref().unwrap_or(s.source.as_str());
    Ok(DslOutput::Message(format!(
        "Applied pipeline '{}' → '{}'.",
        s.pipeline, final_name
    )))
}

pub(super) fn execute_drop_pipeline(
    db: &mut TensorDb,
    name: String,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    if db.pipelines.remove(&name).is_none() {
        return Err(DslError::Parse {
            line: line_no,
            msg: format!("Pipeline '{}' not found", name),
        });
    }
    Ok(DslOutput::Message(format!("Dropped pipeline '{}'.", name)))
}

pub(super) fn execute_describe_pipeline(
    db: &mut TensorDb,
    name: String,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let steps = db.pipelines.get(&name).ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: format!("Pipeline '{}' not found", name),
    })?;

    let mut out = format!("Pipeline: {}\n", name);
    out.push_str("Steps:\n");
    for (i, step) in steps.iter().enumerate() {
        out.push_str(&format!("  {}. {}\n", i + 1, describe_step(step)));
    }
    Ok(DslOutput::Message(out.trim_end().to_string()))
}

pub(super) fn execute_show_pipelines(db: &TensorDb) -> Result<DslOutput, DslError> {
    if db.pipelines.is_empty() {
        return Ok(DslOutput::Message("No pipelines defined.".into()));
    }
    let mut names: Vec<&String> = db.pipelines.keys().collect();
    names.sort();
    let mut out = String::from("--- PIPELINES ---\n");
    for name in names {
        out.push_str(&format!(
            "  {} ({} step(s))\n",
            name,
            db.pipelines[name].len()
        ));
    }
    out.push_str("-----------------");
    Ok(DslOutput::Message(out))
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn pipeline_step_to_select(step: PipelineStep, source: String) -> SelectStmt {
    let base = SelectStmt {
        ctes: vec![],
        distinct: false,
        source: DatasetSource::Named(source),
        joins: vec![],
        columns: SelectColumns::All,
        filter: None,
        group_by: vec![],
        having: None,
        order_by: None,
        limit: None,
        offset: None,
        union: None,
    };

    match step {
        PipelineStep::Select(exprs) => SelectStmt {
            columns: SelectColumns::Named(exprs),
            ..base
        },
        PipelineStep::Filter(expr) => SelectStmt {
            filter: Some(expr),
            ..base
        },
        PipelineStep::OrderBy(cols) => SelectStmt {
            order_by: Some(OrderByClause { columns: cols }),
            ..base
        },
        PipelineStep::Limit(n) => SelectStmt {
            limit: Some(n),
            ..base
        },
        // NormalizeCol is handled separately in execute_apply_pipeline
        PipelineStep::NormalizeCol(_) => base,
    }
}

fn apply_normalize_col(
    db: &mut TensorDb,
    source: &str,
    target: &str,
    col: &str,
    line_no: usize,
) -> Result<(), DslError> {
    let source_ds = db.get_dataset(source).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let schema = source_ds.schema.clone();
    let col_idx = schema
        .fields
        .iter()
        .position(|f| f.name == col)
        .ok_or_else(|| DslError::Parse {
            line: line_no,
            msg: format!("Column '{}' not found in dataset '{}'", col, source),
        })?;

    let mut rows = source_ds.rows.clone();
    for row in &mut rows {
        if let Some(Value::Vector(v)) = row.values.get_mut(col_idx) {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    upsert_dataset(db, target, schema, rows, line_no)
}

fn upsert_dataset(
    db: &mut TensorDb,
    name: &str,
    schema: std::sync::Arc<crate::core::tuple::Schema>,
    rows: Vec<crate::core::tuple::Tuple>,
    line_no: usize,
) -> Result<(), DslError> {
    if db.get_dataset(name).is_ok() {
        let ds = db.get_dataset_mut(name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
        ds.rows = rows;
        ds.metadata.update_stats(&ds.schema, &ds.rows);
    } else {
        db.create_dataset(name.to_string(), schema)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        let ds = db.get_dataset_mut(name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
        ds.rows = rows;
        ds.metadata.update_stats(&ds.schema, &ds.rows);
    }
    Ok(())
}

fn describe_step(step: &PipelineStep) -> String {
    match step {
        PipelineStep::Select(exprs) => {
            let cols: Vec<String> = exprs.iter().map(describe_select_expr).collect();
            format!("SELECT {}", cols.join(", "))
        }
        PipelineStep::Filter(expr) => format!("WHERE {}", describe_expr(expr)),
        PipelineStep::OrderBy(cols) => {
            let parts: Vec<String> = cols
                .iter()
                .map(|(col, asc)| {
                    if *asc {
                        col.clone()
                    } else {
                        format!("{} DESC", col)
                    }
                })
                .collect();
            format!("ORDER BY {}", parts.join(", "))
        }
        PipelineStep::Limit(n) => format!("LIMIT {}", n),
        PipelineStep::NormalizeCol(col) => format!("NORMALIZE {}", col),
    }
}

fn describe_select_expr(expr: &SelectExpr) -> String {
    match expr {
        SelectExpr::Column(name) => name.clone(),
        SelectExpr::Aggregate { func, .. } => format!("{:?}(...)", func),
        SelectExpr::Window { alias, .. } => alias.clone(),
        SelectExpr::Computed { alias, .. } => alias.as_deref().unwrap_or("<expr>").to_string(),
    }
}

fn describe_expr(expr: &Expr) -> String {
    match expr {
        Expr::Ref(name) => name.clone(),
        Expr::Int(n) => n.to_string(),
        Expr::Scalar(f) => f.to_string(),
        Expr::StringLit(s) => format!("\"{}\"", s),
        Expr::Bool(b) => b.to_string(),
        Expr::Infix { op, lhs, rhs } => {
            format!(
                "{} {} {}",
                describe_expr(lhs),
                describe_infix_op(op),
                describe_expr(rhs)
            )
        }
        Expr::And(l, r) => format!("{} AND {}", describe_expr(l), describe_expr(r)),
        Expr::Or(l, r) => format!("{} OR {}", describe_expr(l), describe_expr(r)),
        _ => "<expr>".into(),
    }
}

fn describe_infix_op(op: &InfixOp) -> &'static str {
    match op {
        InfixOp::Eq => "=",
        InfixOp::NotEq => "!=",
        InfixOp::Gt => ">",
        InfixOp::Lt => "<",
        InfixOp::GtEq => ">=",
        InfixOp::LtEq => "<=",
        InfixOp::Add => "+",
        InfixOp::Subtract => "-",
        InfixOp::Multiply => "*",
        InfixOp::Divide => "/",
    }
}
