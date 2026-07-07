use crate::core::value::Value;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

use crate::query::logical::{Expr, LogicalPlan};
use crate::query::planner::Planner;

/// `ds.add_column("col", tensor_var)` — method-call syntax for tensor-first datasets.
///
/// This syntax does not map to any typed AST variant, so it lives here as a
/// legacy fallback that fires when the typed parser cannot handle the line.
pub fn handle_add_tensor_column(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let dot_idx = line.find('.').ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: "Expected '.' in method call".into(),
    })?;
    let ds_name = line[..dot_idx].trim();

    let paren_start = line.find('(').ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: "Expected '(' in method call".into(),
    })?;
    let paren_end = line.rfind(')').ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: "Expected ')' in method call".into(),
    })?;

    let args_str = &line[paren_start + 1..paren_end];
    let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

    if args.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 2 arguments: add_column(name, tensor_var)".into(),
        });
    }

    let col_name = args[0].trim_matches('"').trim_matches('\'');
    let tensor_var = args[1];

    db.add_column_to_tensor_dataset(ds_name, col_name, tensor_var)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    Ok(DslOutput::Message(format!(
        "Added column '{}' to dataset '{}'",
        col_name, ds_name
    )))
}

// ─── Plan-builder functions (used by explain.rs for legacy EXPLAIN forms) ────

pub fn build_select_query_plan(
    db: &TensorDb,
    line: &str,
    line_no: usize,
) -> Result<LogicalPlan, DslError> {
    let from_idx = line.find(" FROM ").ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: "Expected SELECT ... FROM source ...".into(),
    })?;

    let cols_part = line[..from_idx].trim();
    let rest_part = line[from_idx + 6..].trim();

    let parts: Vec<&str> = rest_part.splitn(2, ' ').collect();
    let source_name = parts[0];
    let clauses_str = if parts.len() > 1 { parts[1] } else { "" };

    let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let source_schema = source_ds.schema.clone();

    let mut working_plan = LogicalPlan::Scan {
        dataset_name: source_name.to_string(),
        schema: source_schema.clone(),
    };

    let mut pending_group_by: Option<Vec<Expr>> = None;
    let mut remaining_clauses = clauses_str.to_string();
    let keywords = ["FILTER", "WHERE", "ORDER BY", "LIMIT", "GROUP BY", "HAVING"];

    while !remaining_clauses.is_empty() {
        let clauses_trimmed = remaining_clauses.trim();
        if clauses_trimmed.is_empty() {
            break;
        }

        if clauses_trimmed.starts_with("FILTER ") || clauses_trimmed.starts_with("WHERE ") {
            let kw = if clauses_trimmed.starts_with("WHERE ") {
                "WHERE"
            } else {
                "FILTER"
            };
            let (cond_str, rem) = split_clause(clauses_trimmed, kw, &keywords);
            let cond_string = cond_str.to_string();
            remaining_clauses = rem.to_string();
            let (col, op, val) = parse_filter_condition(&cond_string, line_no)?;
            working_plan = LogicalPlan::Filter {
                input: Box::new(working_plan),
                predicate: Expr::BinaryExpr {
                    left: Box::new(Expr::Column(col)),
                    op,
                    right: Box::new(Expr::Literal(val)),
                },
            };
        } else if clauses_trimmed.starts_with("GROUP BY ") {
            let (group_str, rem) = split_clause(clauses_trimmed, "GROUP BY", &keywords);
            let group_string = group_str.to_string();
            remaining_clauses = rem.to_string();
            let cols: Vec<String> = group_string
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            let exprs: Vec<Expr> = cols.into_iter().map(Expr::Column).collect();
            pending_group_by = Some(exprs);
        } else if clauses_trimmed.starts_with("HAVING ") {
            let (cond_str, rem) = split_clause(clauses_trimmed, "HAVING", &keywords);
            let cond_string = cond_str.to_string();
            remaining_clauses = rem.to_string();
            let (col, op, val) = parse_filter_condition(&cond_string, line_no)?;
            working_plan = LogicalPlan::Filter {
                input: Box::new(working_plan),
                predicate: Expr::BinaryExpr {
                    left: Box::new(Expr::Column(col)),
                    op,
                    right: Box::new(Expr::Literal(val)),
                },
            };
        } else if clauses_trimmed.starts_with("limit ") || clauses_trimmed.starts_with("LIMIT ") {
            let (limit_str, rem) = split_clause(clauses_trimmed, "LIMIT", &keywords);
            let limit_string = limit_str.to_string();
            remaining_clauses = rem.to_string();
            let n: usize = limit_string.parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: "Invalid limit".into(),
            })?;
            working_plan = LogicalPlan::Limit {
                input: Box::new(working_plan),
                n,
            };
        } else if clauses_trimmed.starts_with("ORDER BY ") {
            let (order_str, rem) = split_clause(clauses_trimmed, "ORDER BY", &keywords);
            let order_string = order_str.to_string();
            remaining_clauses = rem.to_string();
            let parts: Vec<&str> = order_string.split_whitespace().collect();
            let col = parts[0].to_string();
            let desc = parts.len() > 1 && parts[1].eq_ignore_ascii_case("DESC");
            working_plan = LogicalPlan::Sort {
                input: Box::new(working_plan),
                column: col,
                ascending: !desc,
            };
        } else {
            return Err(DslError::Parse {
                line: line_no,
                msg: format!("Unknown clause in SELECT: {}", clauses_trimmed),
            });
        }
    }

    let select_exprs_str = cols_part.trim_start_matches("SELECT ").trim();
    let exprs = parse_select_items(select_exprs_str, line_no)?;

    let has_aggr = exprs
        .iter()
        .any(|e| matches!(e, Expr::AggregateExpr { .. }));

    if pending_group_by.is_some() || has_aggr {
        let group_expr = pending_group_by.unwrap_or_default();
        let actual_aggs: Vec<Expr> = exprs
            .into_iter()
            .filter(|e| matches!(e, Expr::AggregateExpr { .. }))
            .collect();
        working_plan = LogicalPlan::Aggregate {
            input: Box::new(working_plan),
            group_expr,
            aggr_expr: actual_aggs,
        };
    } else {
        let mut cols = Vec::new();
        for e in &exprs {
            if let Expr::Column(c) = e {
                if c == "*" {
                    for field in &source_schema.fields {
                        cols.push(field.name.clone());
                    }
                } else {
                    cols.push(c.clone());
                }
            } else {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Only columns or Aggregates supported".into(),
                });
            }
        }
        working_plan = LogicalPlan::Project {
            input: Box::new(working_plan),
            columns: cols,
        };
    }

    Ok(working_plan)
}

pub fn build_dataset_query_plan(
    db: &TensorDb,
    line: &str,
    line_no: usize,
) -> Result<(String, LogicalPlan), DslError> {
    let rest = line.trim_start_matches("DATASET").trim();

    let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: DATASET target FROM source ...".into(),
        });
    }

    let target_name = parts[0].trim().to_string();
    let query_part = parts[1].trim();

    let keywords = [
        "FILTER", "SELECT", "ORDER BY", "LIMIT", "GROUP BY", "HAVING",
    ];
    let mut first_keyword_idx = None;

    for &kw in &keywords {
        if let Some(idx) = query_part.find(kw) {
            if idx > 0 && !query_part[idx - 1..].starts_with(' ') {
                continue;
            }
            if first_keyword_idx.is_none_or(|curr| idx < curr) {
                first_keyword_idx = Some(idx);
            }
        }
    }

    let (source_name, mut clauses_str) = if let Some(idx) = first_keyword_idx {
        (query_part[..idx].trim(), &query_part[idx..])
    } else {
        (query_part.trim(), "")
    };

    let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let source_schema = source_ds.schema.clone();

    let mut current_plan = LogicalPlan::Scan {
        dataset_name: source_name.to_string(),
        schema: source_schema.clone(),
    };

    let mut pending_group_by: Option<Vec<Expr>> = None;
    while !clauses_str.is_empty() {
        let clauses_trimmed = clauses_str.trim();

        if clauses_trimmed.starts_with("FILTER ") {
            let (cond_str, remaining) = split_clause(clauses_trimmed, "FILTER", &keywords);
            clauses_str = remaining;
            let (col, op, val) = parse_filter_condition(cond_str, line_no)?;
            current_plan = LogicalPlan::Filter {
                input: Box::new(current_plan),
                predicate: Expr::BinaryExpr {
                    left: Box::new(Expr::Column(col)),
                    op,
                    right: Box::new(Expr::Literal(val)),
                },
            };
        } else if clauses_trimmed.starts_with("GROUP BY ") {
            let (group_str, remaining) = split_clause(clauses_trimmed, "GROUP BY", &keywords);
            clauses_str = remaining;
            let cols: Vec<String> = group_str.split(',').map(|s| s.trim().to_string()).collect();
            let exprs: Vec<Expr> = cols.into_iter().map(Expr::Column).collect();
            pending_group_by = Some(exprs);
        } else if clauses_trimmed.starts_with("SELECT ") {
            let (cols_str, remaining) = split_clause(clauses_trimmed, "SELECT", &keywords);
            clauses_str = remaining;
            let exprs = parse_select_items(cols_str, line_no)?;
            let has_aggr = exprs
                .iter()
                .any(|e| matches!(e, Expr::AggregateExpr { .. }));
            if pending_group_by.is_some() || has_aggr {
                let group_expr = pending_group_by.take().unwrap_or_default();
                let actual_aggs: Vec<Expr> = exprs
                    .into_iter()
                    .filter(|e| matches!(e, Expr::AggregateExpr { .. }))
                    .collect();
                current_plan = LogicalPlan::Aggregate {
                    input: Box::new(current_plan),
                    group_expr,
                    aggr_expr: actual_aggs,
                };
            } else {
                let cols: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        if let Expr::Column(c) = e {
                            Ok(c.clone())
                        } else {
                            Err(DslError::Parse {
                                line: line_no,
                                msg: "Only columns supported in simple SELECT (Project)".into(),
                            })
                        }
                    })
                    .collect::<Result<_, _>>()?;
                current_plan = LogicalPlan::Project {
                    input: Box::new(current_plan),
                    columns: cols,
                };
            }
        } else if clauses_trimmed.starts_with("HAVING ") {
            let (cond_str, remaining) = split_clause(clauses_trimmed, "HAVING", &keywords);
            clauses_str = remaining;
            let (col, op, val) = parse_filter_condition(cond_str, line_no)?;
            current_plan = LogicalPlan::Filter {
                input: Box::new(current_plan),
                predicate: Expr::BinaryExpr {
                    left: Box::new(Expr::Column(col)),
                    op,
                    right: Box::new(Expr::Literal(val)),
                },
            };
        } else if clauses_trimmed.starts_with("ORDER BY ") {
            let (order_str, remaining) = split_clause(clauses_trimmed, "ORDER BY", &keywords);
            clauses_str = remaining;
            let parts: Vec<&str> = order_str.split_whitespace().collect();
            if parts.is_empty() {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Empty ORDER BY clause".into(),
                });
            }
            let col_name = parts[0].to_string();
            let ascending = !(parts.len() > 1 && parts[1] == "DESC");
            current_plan = LogicalPlan::Sort {
                input: Box::new(current_plan),
                column: col_name,
                ascending,
            };
        } else if clauses_trimmed.starts_with("LIMIT ") {
            let (limit_str, remaining) = split_clause(clauses_trimmed, "LIMIT", &keywords);
            clauses_str = remaining;
            let n: usize = limit_str.trim().parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid LIMIT: {}", limit_str),
            })?;
            current_plan = LogicalPlan::Limit {
                input: Box::new(current_plan),
                n,
            };
        } else {
            return Err(DslError::Parse {
                line: line_no,
                msg: format!("Unexpected clause: {}", clauses_str),
            });
        }
    }

    Ok((target_name, current_plan))
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

fn split_clause<'a>(s: &'a str, current_kw: &str, all_kws: &[&str]) -> (&'a str, &'a str) {
    let content_start = current_kw.len();
    let remaining_s = &s[content_start..];

    let mut next_kw_idx = None;
    for &kw in all_kws {
        if let Some(idx) = remaining_s.find(kw) {
            if idx > 0
                && remaining_s.as_bytes()[idx - 1] == b' '
                && next_kw_idx.is_none_or(|curr| idx < curr)
            {
                next_kw_idx = Some(idx);
            }
        }
    }

    if let Some(idx) = next_kw_idx {
        (remaining_s[..idx].trim(), &remaining_s[idx..])
    } else {
        (remaining_s.trim(), "")
    }
}

fn parse_filter_condition(s: &str, line_no: usize) -> Result<(String, String, Value), DslError> {
    let ops = [">=", "<=", "!=", "=", ">", "<"];
    for op in ops {
        if let Some(idx) = s.find(op) {
            let col = s[..idx].trim().to_string();
            let val_str = s[idx + op.len()..].trim();
            let val = parse_single_value(val_str, line_no)?;
            return Ok((col, op.to_string(), val));
        }
    }
    Err(DslError::Parse {
        line: line_no,
        msg: format!("Invalid filter condition: {}", s),
    })
}

fn parse_select_items(s: &str, line_no: usize) -> Result<Vec<Expr>, DslError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Empty SELECT clause".into(),
        });
    }

    let parts = split_args(s);
    let mut exprs = Vec::new();

    use crate::query::logical::{AggregateFunction, Expr};

    for part in parts {
        let part = part.trim();
        if part == "*" {
            exprs.push(Expr::Column("*".to_string()));
            continue;
        }
        if let Some(idx) = part.find('(') {
            let possible_func = part[..idx].trim().to_uppercase();
            let func = match possible_func.as_str() {
                "SUM" => Some(AggregateFunction::Sum),
                "AVG" => Some(AggregateFunction::Avg),
                "COUNT" => Some(AggregateFunction::Count),
                "MIN" => Some(AggregateFunction::Min),
                "MAX" => Some(AggregateFunction::Max),
                _ => None,
            };
            if let Some(f) = func {
                if part.ends_with(')') {
                    let content = &part[idx + 1..part.len() - 1].trim();
                    let inner = if *content == "*" {
                        Expr::Literal(Value::Int(1))
                    } else {
                        parse_expression(content, line_no)?
                    };
                    exprs.push(Expr::AggregateExpr {
                        func: f,
                        expr: Box::new(inner),
                    });
                    continue;
                }
            }
        }
        exprs.push(parse_expression(part, line_no)?);
    }
    Ok(exprs)
}

fn parse_expression(s: &str, line_no: usize) -> Result<Expr, DslError> {
    parse_expr_add_sub(s, line_no)
}

fn parse_expr_add_sub(s: &str, line_no: usize) -> Result<Expr, DslError> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = chars.len();
    let mut depth = 0;
    let mut last_op_idx = None;
    let mut last_op = ' ';

    while i > 0 {
        i -= 1;
        let c = chars[i];
        if c == ')' {
            depth += 1;
        } else if c == '(' {
            depth -= 1;
        } else if depth == 0 && (c == '+' || c == '-') {
            last_op_idx = Some(i);
            last_op = c;
            break;
        }
    }

    if let Some(idx) = last_op_idx {
        let left_str = s[..idx].trim();
        let right_str = s[idx + 1..].trim();
        if left_str.is_empty() {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Unary operators not supported yet".into(),
            });
        }
        let left = parse_expr_add_sub(left_str, line_no)?;
        let right = parse_term_mul_div(right_str, line_no)?;
        return Ok(Expr::BinaryExpr {
            left: Box::new(left),
            op: last_op.to_string(),
            right: Box::new(right),
        });
    }

    parse_term_mul_div(s, line_no)
}

fn parse_term_mul_div(s: &str, line_no: usize) -> Result<Expr, DslError> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = chars.len();
    let mut depth = 0;
    let mut last_op_idx = None;
    let mut last_op = ' ';

    while i > 0 {
        i -= 1;
        let c = chars[i];
        if c == ')' {
            depth += 1;
        } else if c == '(' {
            depth -= 1;
        } else if depth == 0 && (c == '*' || c == '/') {
            last_op_idx = Some(i);
            last_op = c;
            break;
        }
    }

    if let Some(idx) = last_op_idx {
        let left_str = s[..idx].trim();
        let right_str = s[idx + 1..].trim();
        let left = parse_term_mul_div(left_str, line_no)?;
        let right = parse_factor(right_str, line_no)?;
        return Ok(Expr::BinaryExpr {
            left: Box::new(left),
            op: last_op.to_string(),
            right: Box::new(right),
        });
    }

    parse_factor(s, line_no)
}

fn parse_factor(s: &str, line_no: usize) -> Result<Expr, DslError> {
    let s = s.trim();
    if s.starts_with('(') && s.ends_with(')') {
        return parse_expression(&s[1..s.len() - 1], line_no);
    }
    if let Ok(val) = parse_single_value(s, line_no) {
        Ok(Expr::Literal(val))
    } else {
        Ok(Expr::Column(s.to_string()))
    }
}

pub fn split_args(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in s.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                args.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        args.push(current.trim().to_string());
    }
    args
}

pub fn parse_single_value(s: &str, line_no: usize) -> Result<Value, DslError> {
    let s = s.trim();

    if s.starts_with('"') && s.ends_with('"') {
        return Ok(Value::String(s[1..s.len() - 1].to_string()));
    }
    if s == "true" {
        return Ok(Value::Bool(true));
    }
    if s == "false" {
        return Ok(Value::Bool(false));
    }
    if s.contains('.') && !s.starts_with('[') {
        return s
            .parse::<f32>()
            .map(Value::Float)
            .map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid float: {}", s),
            });
    }
    if s.starts_with('[') && s.ends_with(']') {
        let content = &s[1..s.len() - 1];
        let parts = split_args(content);
        if !parts.is_empty() && parts[0].starts_with('[') {
            let mut matrix = Vec::new();
            for p in parts {
                if let Value::Vector(v) = parse_single_value(&p, line_no)? {
                    matrix.push(v);
                } else {
                    return Err(DslError::Parse {
                        line: line_no,
                        msg: format!("Matrix elements must be vectors. Got: {}", p),
                    });
                }
            }
            return Ok(Value::Matrix(matrix));
        }
        let mut floats = Vec::with_capacity(parts.len());
        for p in parts {
            if p.is_empty() {
                continue;
            }
            let f = p.parse::<f32>().map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid vector element: {}", p),
            })?;
            floats.push(f);
        }
        return Ok(Value::Vector(floats));
    }
    s.parse::<i64>()
        .map(Value::Int)
        .map_err(|_| DslError::Parse {
            line: line_no,
            msg: format!("Invalid value: {}", s),
        })
}

// Used by explain.rs via `build_dataset_query_plan` — executes the plan and creates the target dataset.
pub fn execute_dataset_query(
    db: &mut TensorDb,
    target_name: &str,
    plan: LogicalPlan,
    line_no: usize,
) -> Result<DslOutput, DslError> {
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

    db.create_dataset(target_name.to_string(), result_schema)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    let target_ds = db
        .get_dataset_mut(target_name)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    target_ds.rows = result_rows;
    target_ds
        .metadata
        .update_stats(&target_ds.schema, &target_ds.rows);
    Ok(DslOutput::None)
}
