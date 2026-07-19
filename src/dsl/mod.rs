pub mod ast;
pub mod error;
pub mod executor;
pub mod lexer;
pub mod parser;
pub mod persistence;

pub use error::DslError;

use crate::core::dataset_legacy::Dataset;
use crate::core::tensor::Tensor;
use crate::core::value::Value;
use crate::engine::TensorDb;
use serde::Serialize;

/// Table-row rendering only: caps how many elements of a Vector/Matrix cell
/// are shown so a single wide column doesn't dwarf the rest of the row.
/// Deliberately separate from `Value`'s own `Display` (used elsewhere, e.g.
/// tensor SHOW / error messages) rather than changing that impl's output.
fn format_table_cell(v: &Value) -> String {
    const MAX_ELEMS: usize = 6;
    match v {
        Value::Vector(vec) if vec.len() > MAX_ELEMS => {
            // Value::Float(*x).to_string() (not the raw f32's own to_string())
            // to reuse the same scientific-notation formatting for extreme
            // magnitudes as a plain Float cell -- these are the exact
            // magnitudes (~1e-19 to ~1e-21) real GW strain data has.
            let head: Vec<String> = vec[..MAX_ELEMS]
                .iter()
                .map(|x| Value::Float(*x).to_string())
                .collect();
            format!("[{}, ... ({} total)]", head.join(", "), vec.len())
        }
        Value::Matrix(m) => format!(
            "Matrix[{}x{}]",
            m.len(),
            m.first().map(|r| r.len()).unwrap_or(0)
        ),
        other => other.to_string(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum DslOutput {
    None,
    Message(String),
    Table(Dataset),
    TensorTable(crate::core::dataset::Dataset, Vec<String>),
    Tensor(Tensor),
    LazyTensor(crate::core::tensor::LazyTensor),
}

use std::fmt;

impl fmt::Display for DslOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslOutput::None => Ok(()),
            DslOutput::Message(s) => write!(f, "{}", s),
            DslOutput::Table(ds) => {
                writeln!(
                    f,
                    "Dataset (Legacy): {} (rows: {}, columns: {})",
                    ds.metadata.name.as_deref().unwrap_or("?"),
                    ds.len(),
                    ds.schema.len()
                )?;
                for field in &ds.schema.fields {
                    writeln!(f, "  - {}: {}", field.name, field.value_type)?;
                }
                const MAX_ROWS: usize = 20;
                if !ds.rows.is_empty() {
                    writeln!(f, "  ---")?;
                }
                for row in ds.rows.iter().take(MAX_ROWS) {
                    let cells: Vec<String> = row.values.iter().map(format_table_cell).collect();
                    writeln!(f, "  ({})", cells.join(", "))?;
                }
                if ds.rows.len() > MAX_ROWS {
                    writeln!(f, "  ... ({} more rows)", ds.rows.len() - MAX_ROWS)?;
                }
                Ok(())
            }
            DslOutput::TensorTable(ds, missing_cols) => {
                writeln!(f, "Dataset (Tensor-First): {}", ds.name)?;
                if !missing_cols.is_empty() {
                    writeln!(
                        f,
                        "⚠️  HEALTH WARNING: {} columns missing data!",
                        missing_cols.len()
                    )?;
                    for col in missing_cols {
                        writeln!(
                            f,
                            "  [!] Column '{}' depends on a deleted or missing tensor",
                            col
                        )?;
                    }
                } else {
                    writeln!(f, "✅ Dataset verified (Zero-Copy)")?;
                }
                writeln!(f, "Columns: {}", ds.columns.len())?;
                for col in &ds.schema.columns {
                    writeln!(f, "  - {}: {}", col.name, col.value_type)?;
                }
                Ok(())
            }
            DslOutput::Tensor(t) => {
                writeln!(f, "Tensor ID: {}", t.id.0)?;
                writeln!(f, "Created: {}", t.metadata.created_at)?;
                if let Some(lineage) = &t.metadata.lineage {
                    writeln!(f, "Source Op: {}", lineage.operation)?;
                }
                writeln!(f, "Shape: {:?}", t.shape.dims)?;
                let data = t.to_logical_vec();
                if data.len() > 10 {
                    writeln!(f, "Data: {:?}... (total {})", &data[..10], data.len())?;
                } else {
                    writeln!(f, "Data: {:?}", data)?;
                }
                Ok(())
            }
            DslOutput::LazyTensor(lt) => {
                writeln!(f, "Lazy Tensor ID: {}", lt.id.0)?;
                writeln!(f, "Created: {}", lt.metadata.created_at)?;
                writeln!(f, "Expression: {:?}", lt.expr)?;
                writeln!(f, "Status: PENDING EVALUATION")?;
                Ok(())
            }
        }
    }
}

/// Ejecuta un script completo (varias líneas) sobre un TensorDb
pub fn execute_script(db: &mut TensorDb, script: &str) -> Result<(), DslError> {
    let mut current_cmd = String::new();
    let mut start_line = 0;
    let mut paren_balance = 0;

    for (idx, raw_line) in script.lines().enumerate() {
        let line = raw_line.trim();

        // Ignorar vacío y comentarios IF we are not inside a command
        if current_cmd.is_empty() {
            if line.is_empty()
                || line.starts_with('#')
                || line.starts_with("//")
                || line.starts_with("--")
            {
                continue;
            }
            start_line = idx + 1;
        }

        if !current_cmd.is_empty() {
            current_cmd.push(' ');
        }
        current_cmd.push_str(line);

        // Update balance
        for c in line.chars() {
            if c == '(' {
                paren_balance += 1;
            } else if c == ')' {
                paren_balance -= 1;
            }
        }

        // Check if command is complete
        // Heuristic: balance is 0.
        // Note: This might be fragile if strings contain parens, but MVP.
        if paren_balance == 0 {
            {
                let output = execute_line(db, &current_cmd, start_line)?;
                if !matches!(output, DslOutput::None) {
                    println!("{}", output);
                }
            }
            current_cmd.clear();
        }
    }

    // Check if there is leftover
    if !current_cmd.is_empty() {
        return Err(DslError::Parse {
            line: start_line,
            msg: "Unexpected end of script (unbalanced parentheses?)".into(),
        });
    }

    Ok(())
}

/// Execute a single DSL line
pub fn execute_line(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    execute_line_with_context(db, line, line_no, None)
}

/// Check if a command is read-only
pub fn is_read_only(line: &str) -> bool {
    crate::dsl::parser::parse(line)
        .map(|s| s.is_read_only())
        .unwrap_or(false)
}

/// Execute a single DSL line with an immutable reference to the DB (shared/read-lock path).
///
/// Only read-only statements (EXPLAIN, AUDIT, LIST, DELIVER, SHOW) are dispatched here.
pub fn execute_line_shared(
    db: &TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    match crate::dsl::parser::parse(line) {
        Ok(crate::dsl::ast::Statement::Explain(s)) => {
            executor::execute_explain(db, s.target, line_no)
        }
        Ok(crate::dsl::ast::Statement::Audit(s)) => {
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
        Ok(crate::dsl::ast::Statement::List(s)) => persistence::list_typed(db, &s.target, line_no),
        Ok(crate::dsl::ast::Statement::Deliver(s)) => Ok(DslOutput::Message(format!(
            "Delivery Projection for '{}' created. (Phase 1 Read-Only View)",
            s.dataset
        ))),
        _ => Err(DslError::Parse {
            line: line_no,
            msg: format!(
                "Command is not supported in shared execution mode: {}",
                line
            ),
        }),
    }
}

/// Execute a single DSL line with an optional execution context
pub fn execute_line_with_context(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
    ctx: Option<&mut crate::engine::context::ExecutionContext>,
) -> Result<DslOutput, DslError> {
    // Every statement goes through the typed parser — there is no string-matching
    // fallback chain. On parse failure, only blank/comment lines are tolerated;
    // anything else surfaces the parser's own structured error (byte offset +
    // expectation detail) instead of a generic "Unknown command" message.
    match crate::dsl::parser::parse(line) {
        Ok(stmt) => {
            // Attach the raw source line to DefinePipeline for serialization.
            let stmt = match stmt {
                crate::dsl::ast::Statement::DefinePipeline(mut s) => {
                    s.source = line.to_string();
                    crate::dsl::ast::Statement::DefinePipeline(s)
                }
                other => other,
            };
            executor::execute_statement(db, stmt, line_no, ctx)
        }
        Err(parse_err) => {
            let trimmed = line.trim();
            // Comment-only/blank lines legitimately fail to parse (the lexer
            // skips `--`/`#`/`//` comments entirely, leaving no tokens), and
            // aren't errors — every direct caller of execute_line/execute_line_
            // with_context (REPL, server /execute) must tolerate them the same
            // way execute_script's line-by-line pre-filter already does.
            if trimmed.is_empty()
                || trimmed.starts_with('#')
                || trimmed.starts_with("//")
                || trimmed.starts_with("--")
            {
                return Ok(DslOutput::None);
            }
            Err(parse_err.into_dsl_error(line_no))
        }
    }
}
