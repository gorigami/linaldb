use crate::dsl::ast::*;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use crate::query::logical::{Expr as LogicalExpr, LogicalPlan};
use crate::query::planner::Planner;

use super::query::{agg_func_to_logical, dsl_expr_to_logical_expr};

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
                    predicate: dsl_expr_to_logical_expr(&f),
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
            } else if let Some(exprs) = from.select {
                let cols: Vec<String> = exprs
                    .into_iter()
                    .filter_map(|e| match e {
                        SelectExpr::Column(c) => Some(c),
                        SelectExpr::Aggregate { .. }
                        | SelectExpr::Window { .. }
                        | SelectExpr::Computed { .. } => None,
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
                    predicate: dsl_expr_to_logical_expr(&f),
                };
            }
            if let Some(ord) = from.order_by {
                plan = LogicalPlan::Sort {
                    input: Box::new(plan),
                    columns: ord.columns,
                };
            }
            if let Some(n) = from.limit {
                plan = LogicalPlan::Limit {
                    input: Box::new(plan),
                    n,
                    offset: from.offset.unwrap_or(0),
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
                    use crate::core::tensor::{Shape, TensorId, TensorMetadata};
                    let vals_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                    let n = vals_f32.len();
                    let id = TensorId::new();
                    let meta = TensorMetadata::new(id, None);
                    crate::core::tensor::Tensor::new(id, Shape::new(vec![n]), vals_f32, meta)
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
            // Resolve the FROM source for EXPLAIN
            let source_name = match &s.source {
                DatasetSource::Named(n) => n.clone(),
                DatasetSource::Subquery { alias, .. } => alias.clone(),
            };
            let source_ds = db.get_dataset(&source_name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let source_schema = source_ds.schema.clone();

            let mut plan = LogicalPlan::Scan {
                dataset_name: source_name,
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
                            SelectExpr::Aggregate { .. }
                            | SelectExpr::Window { .. }
                            | SelectExpr::Computed { .. } => None,
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
