pub mod ast;
pub mod delivery_dsl;
pub mod error;
pub mod executor;
pub mod handlers;
pub mod lexer;
pub mod parser;

pub use error::DslError;

use crate::core::dataset_legacy::Dataset;
use crate::core::tensor::Tensor;
use crate::engine::TensorDb;
use serde::Serialize;

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
            match execute_line(db, &current_cmd, start_line) {
                Ok(output) => {
                    if !matches!(output, DslOutput::None) {
                        println!("{}", output);
                    }
                }
                Err(e) => return Err(e),
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
    let line = line.trim();
    line.starts_with("EXPLAIN ")
        || line.starts_with("AUDIT ")
        || line.starts_with("LIST ")
        || line.starts_with("DELIVER ")
}

/// Execute a single DSL line with an immutable reference to the DB (shared/read-lock path).
pub fn execute_line_shared(
    db: &TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    if line.starts_with("EXPLAIN ") {
        handlers::explain::handle_explain(db, line, line_no)
    } else if line.starts_with("AUDIT ") {
        let rest = line.strip_prefix("AUDIT ").unwrap().trim();
        let ds_name = if rest.starts_with("DATASET ") {
            rest.strip_prefix("DATASET ").unwrap().trim()
        } else {
            rest
        };
        let issues = db
            .verify_tensor_dataset(ds_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        if issues.is_empty() {
            Ok(DslOutput::Message(format!(
                "Audit PASSED for dataset '{}'. All column references are valid.",
                ds_name
            )))
        } else {
            Ok(DslOutput::Message(format!(
                "Audit FAILED for dataset '{}'. The following columns point to missing or invalid tensors: {:?}",
                ds_name, issues
            )))
        }
    } else if line.starts_with("LIST ") {
        handlers::persistence::handle_list_datasets(db, line, line_no)
    } else if line.starts_with("DELIVER ") {
        let dataset = line
            .strip_prefix("DELIVER ")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap_or("?");
        Ok(DslOutput::Message(format!(
            "Delivery Projection for '{}' created. (Phase 1 Read-Only View)",
            dataset
        )))
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: format!(
                "Command is not supported in shared execution mode: {}",
                line
            ),
        })
    }
}

/// Execute a single DSL line with an optional execution context
pub fn execute_line_with_context(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
    ctx: Option<&mut crate::engine::context::ExecutionContext>,
) -> Result<DslOutput, DslError> {
    // Try the new typed parser first; fall through to the legacy chain on parse error.
    if let Ok(stmt) = crate::dsl::parser::parse(line) {
        return executor::execute_statement(db, stmt, line_no, ctx);
    }

    if line.starts_with("SELECT ") {
        handlers::dataset::handle_select(db, line, line_no)
    } else if line.starts_with("DATASET ") {
        handlers::dataset::handle_dataset(db, line, line_no)
    } else if line.starts_with("INSERT INTO ") {
        handlers::dataset::handle_insert(db, line, line_no)
    } else if line.starts_with("SEARCH ") {
        handlers::search::handle_search(db, line, line_no)
    } else if line.starts_with("EXPLAIN ") {
        handlers::explain::handle_explain(db, line, line_no)
    } else if line.starts_with("MATERIALIZE ") {
        handlers::dataset::handle_materialize(db, line, line_no)
    } else if line.contains(".add_column(") {
        handlers::dataset::handle_add_tensor_column(db, line, line_no)
    } else if line.starts_with("ALTER ") {
        let rest = line.strip_prefix("ALTER ").unwrap();
        if rest.starts_with("DATASET ") {
            handlers::dataset::handle_dataset(db, rest, line_no)
        } else {
            Err(DslError::Parse {
                line: line_no,
                msg: format!("Unsupported ALTER command: {}", line),
            })
        }
    } else if line.starts_with("SET ") {
        if line.starts_with("SET DATASET ") {
            handlers::metadata::handle_set_metadata(db, line, line_no)
        } else {
            Err(DslError::Parse {
                line: line_no,
                msg: format!("Unsupported SET command: {}", line),
            })
        }
    } else if line.starts_with("SAVE ") {
        handlers::persistence::handle_save(db, line, line_no)
    } else if line.starts_with("LOAD ") {
        handlers::persistence::handle_load(db, line, line_no)
    } else if line.starts_with("LIST ") {
        handlers::persistence::handle_list_datasets(db, line, line_no)
    } else if line.starts_with("IMPORT ") {
        handlers::persistence::handle_import(db, line, line_no)
    } else if line.starts_with("EXPORT ") {
        handlers::persistence::handle_export(db, line, line_no)
    } else {
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            return Ok(DslOutput::None);
        }
        Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown command: {}", line),
        })
    }
}
