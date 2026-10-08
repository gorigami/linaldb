use crate::core::dataset_legacy;
use crate::core::tuple::Tuple;
use crate::core::value::{Value, ValueType};
use crate::dsl::ast::*;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use crate::query::logical::{AggregateFunction, Expr as LogicalExpr, JoinType, LogicalPlan};
use crate::query::planner::Planner;
use std::sync::Arc;

type RowPredicate = Box<dyn Fn(&Tuple) -> bool>;

// ─── Dataset query execution ──────────────────────────────────────────────────

pub(super) fn execute_create_dataset_from(
    db: &mut TensorDb,
    name: String,
    clause: DatasetFromClause,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let source_name = clause.source.clone();
    let had_group_by = !clause.group_by.is_empty();
    // What lineage records about the transform: the select list and filter
    // as text, so e.g. `SPEC_CLEAN(spec, pm, 0.01, 50)` and its parameters
    // show up in `EXPLAIN LINEAGE`.
    let as_alias = |text: String, alias: &Option<String>| match alias {
        Some(a) => format!("{} AS {}", text, a),
        None => text,
    };
    let select_text: Option<Vec<String>> = clause.select.as_ref().map(|exprs| {
        exprs
            .iter()
            .map(|e| match e {
                SelectExpr::Column(c) => c.clone(),
                SelectExpr::Computed { expr, alias } => {
                    as_alias(super::eval::expr_to_string(expr), alias)
                }
                SelectExpr::Aggregate { func, expr, alias } => as_alias(
                    format!(
                        "{}({})",
                        format!("{:?}", func).to_uppercase(),
                        super::eval::expr_to_string(expr)
                    ),
                    alias,
                ),
                SelectExpr::Window { alias, .. } => alias.clone(),
            })
            .collect()
    });
    let filter_text = clause.filter.as_ref().map(super::eval::expr_to_string);
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
        source: Some(DatasetSource::Named(clause.source)),
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
    let output_hash = target_ds.content_hash();

    let operation = if had_group_by {
        "DATASET FROM (GROUP BY)"
    } else {
        "DATASET FROM"
    };
    let inputs = if let Ok(src_ds) = db.get_dataset(&source_name) {
        vec![crate::core::provenance::ProvenanceEntity::dataset(
            source_name.clone(),
            src_ds.content_hash(),
        )]
    } else if let Ok(src_tensor) = db.get(&source_name) {
        vec![crate::core::provenance::ProvenanceEntity::tensor(
            src_tensor.id,
            Some(source_name.clone()),
            src_tensor.data_hash().to_string(),
        )]
    } else {
        Vec::new()
    };
    let record = crate::core::provenance::ProvenanceRecord::new(
        operation,
        crate::core::tensor::ExecutionId::new(),
    )
    .with_param("source", source_name);
    let record = match select_text {
        Some(cols) => record.with_param("select", cols),
        None => record,
    };
    let record = match filter_text {
        Some(f) => record.with_param("filter", f),
        None => record,
    }
    .with_inputs(inputs)
    .with_outputs(vec![crate::core::provenance::ProvenanceEntity::dataset(
        name,
        output_hash,
    )]);
    db.active_instance_mut().record_provenance(record);

    Ok(DslOutput::None)
}

/// Runs a `SEARCH` and returns its result rows, without touching the
/// database. Shared by the executor (which then either returns the rows or
/// stores them `INTO` a dataset) and the server's read-lock path, which
/// only handles the no-`INTO` form (`dsl::can_execute_shared`).
pub(crate) fn run_search(
    db: &TensorDb,
    s: &SearchStmt,
    line_no: usize,
) -> Result<(Arc<crate::core::tuple::Schema>, Vec<Tuple>), DslError> {
    let (mut plan, filter_schema) = search_plan(db, s, line_no)?;
    if let Some(filter_expr) = &s.filter {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: dsl_expr_to_logical_expr(
                filter_expr,
                &filter_schema,
                &std::collections::HashSet::new(),
            ),
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
    Ok((physical_plan.schema(), result_rows))
}

/// Builds a `SEARCH` statement's logical plan, without its `FILTER` -- shared
/// by execution and `EXPLAIN`. Also returns the schema the `FILTER`
/// predicate is resolved against (the search's own output schema).
pub(super) fn search_plan(
    db: &TensorDb,
    s: &SearchStmt,
    line_no: usize,
) -> Result<(LogicalPlan, Arc<crate::core::tuple::Schema>), DslError> {
    let engine_err = |e| DslError::Engine {
        line: line_no,
        source: e,
    };
    let invalid = |msg: String| DslError::Engine {
        line: line_no,
        source: crate::engine::EngineError::InvalidOp(msg),
    };
    let source_ds = db.get_dataset(&s.dataset).map_err(engine_err)?;
    let schema = source_ds.schema.clone();
    let single_query = |db: &TensorDb| -> Result<crate::core::tensor::Tensor, DslError> {
        Ok(match s.query {
            SearchQuery::TensorRef(ref name) => db.get(name).map_err(engine_err)?.clone(),
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
            SearchQuery::Batch { .. } => unreachable!("handled by the batch path"),
        })
    };

    if s.prefilter.is_none()
        && matches!(
            schema.get_field(&s.column).map(|f| &f.value_type),
            Some(ValueType::SparseVector(_))
        )
    {
        return Err(invalid(format!(
            "SEARCH: column '{}' is a SparseVector, which vector indexes can't hold -- add PREFILTER <predicate> (e.g. PREFILTER true) for an exact search",
            s.column
        )));
    }
    let is_batch = matches!(s.query, SearchQuery::Batch { .. });
    // RETURN: the dataset columns each hit carries, in the order given.
    let projection: Option<Vec<usize>> = match &s.returning {
        None => None,
        Some(cols) if cols.is_empty() && !is_batch => {
            return Err(invalid(
                "SEARCH: RETURN NONE needs a batch (QUERIES ...): a single-query SEARCH returns the dataset's own rows, so name at least one column".to_string(),
            ))
        }
        Some(cols) => {
            let mut idx = Vec::with_capacity(cols.len());
            for c in cols {
                let i = schema.get_field_index(c).ok_or_else(|| {
                    invalid(format!(
                        "SEARCH ... RETURN: unknown column '{}' in dataset '{}'",
                        c, s.dataset
                    ))
                })?;
                if idx.contains(&i) {
                    return Err(invalid(format!(
                        "SEARCH ... RETURN: column '{}' is listed twice",
                        c
                    )));
                }
                idx.push(i);
            }
            Some(idx)
        }
    };
    if !is_batch && s.prefilter.is_none() {
        let plan = LogicalPlan::VectorSearch {
            input: Box::new(LogicalPlan::Scan {
                dataset_name: s.dataset.clone(),
                schema: schema.clone(),
            }),
            column: s.column.clone(),
            query: single_query(db)?,
            k: s.top_k,
        };
        return Ok(match (&s.returning, &projection) {
            (Some(cols), Some(idx)) => {
                let fields = idx.iter().map(|&i| schema.fields[i].clone()).collect();
                (
                    LogicalPlan::Project {
                        input: Box::new(plan),
                        columns: cols.clone(),
                    },
                    Arc::new(crate::core::tuple::Schema::new(fields)),
                )
            }
            _ => (plan, schema),
        });
    }

    // Batch queries, and any PREFILTER search (a single query runs as a
    // batch of one, projected back to the dataset's columns below).
    let target_dim = match schema.get_field(&s.column).map(|f| &f.value_type) {
        Some(ValueType::Vector(d)) => *d,
        Some(ValueType::SparseVector(d)) => *d,
        Some(ValueType::QVector(d, _)) => *d,
        Some(other) => {
            return Err(invalid(format!(
                "SEARCH: column '{}' is {:?}, not a Vector",
                s.column, other
            )))
        }
        None => {
            return Err(invalid(format!(
                "SEARCH: column '{}' not found in dataset '{}'",
                s.column, s.dataset
            )))
        }
    };
    let batch = match s.query {
        SearchQuery::Batch {
            ref source,
            ref column,
            ref key,
        } => {
            resolve_batch_queries(db, source, column.as_deref(), key.as_deref()).map_err(invalid)?
        }
        _ => BatchQueries {
            queries: vec![(Value::Int(0), single_query(db)?.to_logical_vec())],
            key_type: ValueType::Int,
            source: None,
        },
    };
    for (i, (_, v)) in batch.queries.iter().enumerate() {
        if target_dim != 0 && v.len() != target_dim {
            return Err(invalid(format!(
                "SEARCH: query {} has dimension {}, but '{}' is Vector({})",
                i,
                v.len(),
                s.column,
                target_dim
            )));
        }
    }
    // A single prefiltered query returns the same shape as plain SEARCH
    // (the dataset's own rows); a batch adds the per-hit columns in front.
    let mut fields = if is_batch {
        vec![
            crate::core::tuple::Field::new("query_id", batch.key_type.clone()),
            crate::core::tuple::Field::new("rank", ValueType::Int),
            crate::core::tuple::Field::new("score", ValueType::Float),
            crate::core::tuple::Field::new("row_id", ValueType::Int),
        ]
    } else {
        Vec::new()
    };
    let kept: Vec<usize> = match &projection {
        Some(idx) => idx.clone(),
        None => (0..schema.fields.len()).collect(),
    };
    for f in kept.iter().map(|&i| &schema.fields[i]) {
        if fields.iter().any(|g| g.name == f.name) {
            return Err(invalid(format!(
                "SEARCH QUERIES: dataset '{}' has a column named '{}', which collides with the batch result column of the same name -- rename it first",
                s.dataset, f.name
            )));
        }
        let mut field = f.clone();
        field.is_lazy = false;
        fields.push(field);
    }
    let out_schema = Arc::new(crate::core::tuple::Schema::new(fields));
    let prefilter = match &s.prefilter {
        Some(expr) => Some(Arc::new(
            build_prefilter(db, expr, &s.dataset, &schema, &batch, s.approx).map_err(invalid)?,
        )),
        None => None,
    };
    let plan = LogicalPlan::BatchVectorSearch {
        dataset_name: s.dataset.clone(),
        column: s.column.clone(),
        queries: Arc::new(crate::query::logical::QueryBatch(batch.queries)),
        k: s.top_k,
        schema: out_schema.clone(),
        prefilter,
        rows_only: !is_batch,
        projection,
    };
    Ok((plan, out_schema))
}

/// `SEARCH ... QUERIES`'s resolved queries, `(query_id, vector)`, the
/// `query_id` column's type, and -- for a dataset source -- its name,
/// schema and every query row's values (for `PREFILTER`).
struct BatchQueries {
    queries: Vec<(Value, Vec<f32>)>,
    key_type: ValueType,
    source: Option<(String, Arc<crate::core::tuple::Schema>, Vec<Vec<Value>>)>,
}

/// Resolves `SEARCH ... QUERIES`'s source: rows of a 2-D tensor (ids
/// `0..n`), or a dataset's vector column (ids from its `KEY` column, else
/// `0..n`).
fn resolve_batch_queries(
    db: &TensorDb,
    source: &str,
    column: Option<&str>,
    key: Option<&str>,
) -> Result<BatchQueries, String> {
    let Some(column) = column else {
        let t = db.get(source).map_err(|e| e.to_string())?;
        let dims = &t.shape.dims;
        if dims.len() != 2 {
            return Err(format!(
                "SEARCH QUERIES: tensor '{}' has shape {:?}; a batch of queries must be a 2-D matrix (one query per row), or use QUERIES <dataset>.<column>",
                source, dims
            ));
        }
        let data = t.to_logical_vec();
        let d = dims[1];
        let queries = (0..dims[0])
            .map(|i| (Value::Int(i as i64), data[i * d..(i + 1) * d].to_vec()))
            .collect();
        return Ok(BatchQueries {
            queries,
            key_type: ValueType::Int,
            source: None,
        });
    };
    let ds = db.get_dataset(source).map_err(|e| e.to_string())?;
    let col_idx = ds.schema.get_field_index(column).ok_or_else(|| {
        format!(
            "SEARCH QUERIES: column '{}' not found in dataset '{}'",
            column, source
        )
    })?;
    let key_idx = match key {
        Some(k) => Some(ds.schema.get_field_index(k).ok_or_else(|| {
            format!(
                "SEARCH QUERIES: KEY column '{}' not found in dataset '{}'",
                k, source
            )
        })?),
        None => None,
    };
    let key_type = match key_idx {
        Some(i) => ds.schema.fields[i].value_type.clone(),
        None => ValueType::Int,
    };
    let mut queries = Vec::with_capacity(ds.rows.len());
    let mut rows = Vec::with_capacity(ds.rows.len());
    for (i, row) in ds.rows.iter().enumerate() {
        let row = crate::query::physical::evaluate_lazy_columns_in_row(ds, row)
            .map_err(|e| e.to_string())?;
        let v = match &row.values[col_idx] {
            Value::Vector(v) => v.clone(),
            Value::SparseVector(sv) => sv.to_dense(),
            Value::QVector(q) => q.dequantize(),
            other => {
                return Err(format!(
                    "SEARCH QUERIES: row {} of '{}.{}' is {:?}, not a Vector",
                    i,
                    source,
                    column,
                    other.value_type()
                ))
            }
        };
        let id = match key_idx {
            Some(k) => row.values[k].clone(),
            None => Value::Int(i as i64),
        };
        queries.push((id, v));
        rows.push(row.values);
    }
    Ok(BatchQueries {
        queries,
        key_type,
        source: Some((source.to_string(), ds.schema.clone(), rows)),
    })
}

/// Lowers a `PREFILTER` predicate. `<query dataset>.<column>` becomes the
/// query column `QUERY_COLUMN_PREFIX + column`, evaluated per query; every
/// other name must be a column of the searched dataset.
fn build_prefilter(
    db: &TensorDb,
    expr: &Expr,
    dataset: &str,
    schema: &crate::core::tuple::Schema,
    batch: &BatchQueries,
    approximate: bool,
) -> Result<crate::query::logical::Prefilter, String> {
    use crate::query::logical::QUERY_COLUMN_PREFIX;

    let query_source = batch.source.as_ref();
    let unknown = std::cell::RefCell::new(Vec::new());
    let rewritten = map_expr(expr, &|e| match (e, query_source) {
        (Expr::Field { base, field }, Some((qname, qschema, _))) => match base.as_ref() {
            Expr::Ref(b) if b == qname => {
                if qschema.get_field_index(field).is_none() {
                    unknown.borrow_mut().push(format!("{}.{}", qname, field));
                }
                Some(Expr::Ref(format!("{}{}", QUERY_COLUMN_PREFIX, field)))
            }
            _ => None,
        },
        _ => None,
    });
    if let Some(name) = unknown.into_inner().first() {
        return Err(format!("PREFILTER: unknown query column '{}'", name));
    }

    let mut fields: Vec<crate::core::tuple::Field> = schema
        .fields
        .iter()
        .map(|f| {
            let mut f = f.clone();
            f.is_lazy = false;
            f
        })
        .collect();
    let mut query_values = Vec::new();
    if let Some((_, qschema, rows)) = query_source {
        for f in &qschema.fields {
            let mut f = f.clone();
            f.name = format!("{}{}", QUERY_COLUMN_PREFIX, f.name);
            f.is_lazy = false;
            fields.push(f.nullable());
        }
        query_values = rows.clone();
    }
    let combined_schema = Arc::new(crate::core::tuple::Schema::new(fields));
    let predicate = dsl_expr_to_logical_expr(
        &rewritten,
        &combined_schema,
        &std::collections::HashSet::new(),
    );
    let mut referenced = Vec::new();
    collect_referenced_columns(&predicate, &mut referenced);
    for name in &referenced {
        if combined_schema.get_field_index(name).is_none() {
            return Err(format!(
                "PREFILTER: unknown column '{}' in dataset '{}'",
                name, dataset
            ));
        }
    }

    let sorted_range = sorted_prefilter_range(db, dataset, &predicate);
    Ok(crate::query::logical::Prefilter {
        predicate,
        combined_schema,
        query_values,
        sorted_range,
        approximate,
    })
}

/// The first conjunct of a `PREFILTER` predicate that the SORTED index on
/// one of `dataset`'s columns can answer, with bounds that use only
/// constants and query columns (so they're known per query).
fn sorted_prefilter_range(
    db: &TensorDb,
    dataset: &str,
    predicate: &LogicalExpr,
) -> Option<(String, Vec<(String, LogicalExpr)>)> {
    use crate::query::logical::QUERY_COLUMN_PREFIX;

    let ds = db.get_dataset(dataset).ok()?;
    let has_sorted = |c: &str| {
        ds.get_index(c)
            .is_some_and(|i| i.index_type() == crate::core::index::IndexType::Sorted)
    };
    let query_only = |e: &LogicalExpr| {
        let mut cols = Vec::new();
        collect_referenced_columns(e, &mut cols);
        cols.iter().all(|c| c.starts_with(QUERY_COLUMN_PREFIX))
    };
    let flip = |op: &str| {
        match op {
            "<" => ">",
            "<=" => ">=",
            ">" => "<",
            ">=" => "<=",
            other => other,
        }
        .to_string()
    };

    let mut conjuncts = vec![predicate];
    let mut i = 0;
    while i < conjuncts.len() {
        if let LogicalExpr::And(l, r) = conjuncts[i] {
            conjuncts[i] = l;
            conjuncts.push(r);
        } else {
            i += 1;
        }
    }
    conjuncts.into_iter().find_map(|c| match c {
        LogicalExpr::Between { expr, low, high } => match expr.as_ref() {
            LogicalExpr::Column(col) if has_sorted(col) && query_only(low) && query_only(high) => {
                Some((
                    col.clone(),
                    vec![
                        (">=".to_string(), (**low).clone()),
                        ("<=".to_string(), (**high).clone()),
                    ],
                ))
            }
            _ => None,
        },
        LogicalExpr::BinaryExpr { left, op, right }
            if matches!(op.as_str(), "<" | "<=" | ">" | ">=" | "=") =>
        {
            match (left.as_ref(), right.as_ref()) {
                (LogicalExpr::Column(col), other) if has_sorted(col) && query_only(other) => {
                    Some((col.clone(), vec![(op.clone(), other.clone())]))
                }
                (other, LogicalExpr::Column(col)) if has_sorted(col) && query_only(other) => {
                    Some((col.clone(), vec![(flip(op), other.clone())]))
                }
                _ => None,
            }
        }
        _ => None,
    })
}

/// A `SEARCH` result returned inline (no `INTO`).
pub(crate) fn search_result_table(
    schema: Arc<crate::core::tuple::Schema>,
    rows: Vec<Tuple>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let ds = dataset_legacy::Dataset::with_rows(
        dataset_legacy::DatasetId(0),
        schema,
        rows,
        Some("Search Result".into()),
    )
    .map_err(|e| DslError::Parse {
        line: line_no,
        msg: e,
    })?;
    Ok(DslOutput::Table(ds))
}

/// Rows a query has already computed and can refer to by name: its CTEs
/// (`WITH name AS (...)`). They're scoped to the query -- visible to later
/// CTEs, nested subqueries and the right side of a `UNION` -- and never
/// registered in the database, which is what lets `execute_select` take
/// `&TensorDb`. A CTE shadows a real dataset with the same name for the
/// duration of the query, as in SQL.
#[derive(Clone, Default)]
struct QueryScope {
    temps: std::collections::HashMap<String, (Arc<crate::core::tuple::Schema>, Arc<Vec<Tuple>>)>,
}

impl QueryScope {
    /// A `LogicalPlan` reading `name`: the scoped rows if it's a CTE, else a
    /// scan of the database's dataset.
    fn source_plan(
        &self,
        db: &TensorDb,
        name: &str,
        line_no: usize,
    ) -> Result<LogicalPlan, DslError> {
        if let Some((schema, rows)) = self.temps.get(name) {
            return Ok(LogicalPlan::Values {
                name: name.to_string(),
                schema: schema.clone(),
                rows: rows.clone(),
            });
        }
        let ds = db.get_dataset(name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
        Ok(LogicalPlan::Scan {
            dataset_name: name.to_string(),
            schema: ds.schema.clone(),
        })
    }
}

/// Executes a `SELECT`. Takes `&TensorDb`: a query reads the database and
/// never writes to it -- intermediate results (CTEs, `FROM` subqueries) stay
/// in a per-query `QueryScope` -- so `linal serve` can run it under a read
/// lock (`dsl::can_execute_shared`), concurrently with other reads of the
/// same database.
pub(crate) fn execute_select(
    db: &TensorDb,
    s: SelectStmt,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    execute_select_in_scope(db, s, line_no, &QueryScope::default())
}

fn execute_select_in_scope(
    db: &TensorDb,
    s: SelectStmt,
    line_no: usize,
    outer: &QueryScope,
) -> Result<DslOutput, DslError> {
    let mut scope = outer.clone();
    for (cte_name, cte_query) in s.ctes {
        let cte_result = execute_select_in_scope(db, cte_query, line_no, &scope)?;
        if let DslOutput::Table(cte_ds) = cte_result {
            scope
                .temps
                .insert(cte_name, (cte_ds.schema.clone(), Arc::new(cte_ds.rows)));
        }
    }

    // A SELECT with no FROM clause at all: evaluate the SELECT list once
    // against a synthetic empty row and return immediately -- everything
    // else in this function (JOIN/WHERE/GROUP BY/ORDER BY/etc.) is
    // meaningless without a real data source and is simply not applied.
    // See `SelectStmt::source`'s doc comment for why this exists (a real
    // `DSL_REFERENCE.md` example, `SELECT L2_NORM([3.0, 4.0]) AS five`, has
    // no dataset to name).
    let Some(source) = s.source else {
        let SelectColumns::Named(exprs) = &s.columns else {
            return Err(DslError::Parse {
                line: line_no,
                msg: "SELECT * requires a FROM clause".to_string(),
            });
        };
        let empty_schema = std::sync::Arc::new(crate::core::tuple::Schema::new(vec![]));
        let empty_right_tables = std::collections::HashSet::new();
        let empty_row = Tuple::new(empty_schema.clone(), vec![]).map_err(|e| DslError::Parse {
            line: line_no,
            msg: e,
        })?;

        let mut fields = Vec::with_capacity(exprs.len());
        let mut values = Vec::with_capacity(exprs.len());
        let mut computed_idx = 0usize;
        for e in exprs {
            let SelectExpr::Computed { expr, alias } = e else {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "SELECT without FROM only supports literal/computed expressions \
                          -- no column, aggregate, or window reference (there is no \
                          dataset to resolve one against)"
                        .to_string(),
                });
            };
            let name = alias.clone().unwrap_or_else(|| {
                let n = format!("__cmp_{computed_idx}");
                computed_idx += 1;
                n
            });
            let logical = dsl_expr_to_logical_expr(expr, &empty_schema, &empty_right_tables);
            let val = crate::query::physical::evaluate_expression(&logical, &empty_row);
            fields.push(crate::core::tuple::Field::new(&name, val.value_type()));
            values.push(val);
        }
        let schema = std::sync::Arc::new(crate::core::tuple::Schema::new(fields));
        let row = Tuple::new(schema.clone(), values).map_err(|e| DslError::Parse {
            line: line_no,
            msg: e,
        })?;
        let ds = dataset_legacy::Dataset::with_rows(
            dataset_legacy::DatasetId(0),
            schema,
            vec![row],
            Some("Query Result".into()),
        )
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: e,
        })?;
        return Ok(DslOutput::Table(ds));
    };

    // Resolve the FROM source — either a named dataset or an executed subquery.
    let mut plan = match source {
        DatasetSource::Named(ref name) => scope.source_plan(db, name, line_no)?,
        // Previously registered as a real dataset named after the alias and
        // never removed: it leaked into the catalog, and running the same
        // query a second time failed with "Dataset name already exists".
        DatasetSource::Subquery { query, alias } => {
            let inner = execute_select_in_scope(db, *query, line_no, &scope)?;
            if let DslOutput::Table(inner_ds) = inner {
                LogicalPlan::Values {
                    name: alias,
                    schema: inner_ds.schema.clone(),
                    rows: Arc::new(inner_ds.rows),
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
        let right_plan = scope.source_plan(db, &join.dataset, line_no)?;
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

    // Names a qualifier can use to mean the right side of a JOIN -- the
    // literal dataset name and, if given, its `[AS] alias` (either is a
    // valid qualifier in SELECT/WHERE/aggregates, e.g. `JOIN users u ON
    // ... SELECT u.name`). Needed to resolve a qualified `table.col`
    // reference correctly wherever this query's expressions get lowered
    // below (see `dsl_expr_to_logical_expr`'s `Expr::Field` arm). Empty for
    // a query with no JOIN, which makes every qualifier resolution below a
    // no-op fallback to the bare column name, identical to previous
    // behavior.
    let right_table_names: std::collections::HashSet<String> = s
        .joins
        .iter()
        .flat_map(|j| std::iter::once(j.dataset.clone()).chain(j.alias.clone()))
        .collect();

    if let Some(filter_expr) = &s.filter {
        let schema_now = plan.schema();
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: dsl_expr_to_logical_expr(filter_expr, &schema_now, &right_table_names),
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
        let pre_aggr_schema = plan.schema();
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
                            func: agg_func_to_logical(func, |e| {
                                dsl_expr_to_logical_expr(e, &pre_aggr_schema, &right_table_names)
                            }),
                            expr: Box::new(dsl_expr_to_logical_expr(
                                expr,
                                &pre_aggr_schema,
                                &right_table_names,
                            )),
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
            let predicate = resolve_having(
                having_expr,
                &aggr_exprs,
                &schema_now,
                &right_table_names,
                line_no,
            )?;
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
            let pre_aggr_schema = plan.schema();
            match &s.columns {
                SelectColumns::Named(exprs) => exprs
                    .iter()
                    .filter_map(|e| match e {
                        SelectExpr::Aggregate { func, expr, alias } => {
                            Some(LogicalExpr::AggregateExpr {
                                func: agg_func_to_logical(func, |e| {
                                    dsl_expr_to_logical_expr(
                                        e,
                                        &pre_aggr_schema,
                                        &right_table_names,
                                    )
                                }),
                                expr: Box::new(dsl_expr_to_logical_expr(
                                    expr,
                                    &pre_aggr_schema,
                                    &right_table_names,
                                )),
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
            let predicate = resolve_having(
                having_expr,
                &aggr_exprs,
                &schema_now,
                &right_table_names,
                line_no,
            )?;
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
        result_rows = apply_window_and_computed_exprs(
            result_rows,
            &base_schema,
            &right_table_names,
            &window_exprs,
            line_no,
        )?;

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
        let right_result = execute_select_in_scope(db, *right_stmt, line_no, &scope)?;
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
            CastTarget::QVector(n, e) => ValueType::QVector(*n, *e),
            CastTarget::Matrix(r, c) => ValueType::Matrix(*r, *c),
            CastTarget::BitVector(n) => ValueType::BitVector(n.unwrap_or(0)),
            CastTarget::SparseVector(n) => ValueType::SparseVector(*n),
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
            VectorFnKind::Real
            | VectorFnKind::Imag
            | VectorFnKind::ComplexAbs
            | VectorFnKind::Phase => ValueType::Float64,
            VectorFnKind::Conj | VectorFnKind::ComplexNew => ValueType::Complex,
            VectorFnKind::Tanimoto | VectorFnKind::Jaccard => ValueType::Float64,
            VectorFnKind::Hamming | VectorFnKind::BitCount => ValueType::Int,
            VectorFnKind::SpecCosine | VectorFnKind::SpecCosineMod => ValueType::Float64,
            VectorFnKind::SpecMatches => ValueType::Int,
            VectorFnKind::SpecEntropy => ValueType::Float64,
            VectorFnKind::SpecClean => ValueType::Matrix(2, 0),
            VectorFnKind::SparseNew => ValueType::SparseVector(0),
        },
        _ => ValueType::Float,
    }
}

fn apply_window_and_computed_exprs(
    mut rows: Vec<Tuple>,
    base_schema: &std::sync::Arc<crate::core::tuple::Schema>,
    right_tables: &std::collections::HashSet<String>,
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
                // `base_schema` (the plan's schema before any computed
                // columns were appended, i.e. still reflecting a JOIN's
                // `r_`-collision-renaming if present) is correct for every
                // iteration here: a qualified `table.col` inside `expr`
                // always refers to a real source column, never to a
                // previously-appended computed one.
                let logical_expr = dsl_expr_to_logical_expr(expr, base_schema, right_tables);
                crate::query::typecheck::check_expr(&logical_expr, base_schema).map_err(|e| {
                    DslError::Engine {
                        line: line_no,
                        source: crate::engine::EngineError::InvalidOp(e),
                    }
                })?;
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
                crate::query::row_error::clear();
                let vals: Vec<Value> = rows
                    .iter()
                    .map(|row| evaluate_expression(&logical_expr, row))
                    .collect();
                if let Some(e) = crate::query::row_error::take() {
                    return Err(DslError::Engine {
                        line: line_no,
                        source: crate::engine::EngineError::InvalidOp(e),
                    });
                }
                let mut vtype = vals
                    .iter()
                    .find(|v| !matches!(v, Value::Null))
                    .map(|v| v.value_type())
                    .unwrap_or(fallback_vtype);
                // Rows of different widths (e.g. `SPEC_CLEAN` peak lists):
                // the column is `Matrix(r, *)` / `Vector(*)`. Typing it by
                // the first row made every other row fail validation and
                // silently lose the column.
                let widths_differ = |t: &ValueType| {
                    vals.iter()
                        .filter(|v| !v.is_null())
                        .any(|v| v.value_type() != *t)
                };
                match vtype {
                    ValueType::Matrix(r, c) if c != 0 && widths_differ(&vtype) => {
                        vtype = ValueType::Matrix(r, 0)
                    }
                    ValueType::Vector(d) if d != 0 && widths_differ(&vtype) => {
                        vtype = ValueType::Vector(0)
                    }
                    _ => {}
                }

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
                rows = apply_window_func(rows, func, spec, alias, right_tables, line_no)?;
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
    right_tables: &std::collections::HashSet<String>,
    line_no: usize,
) -> Result<Vec<Tuple>, DslError> {
    use crate::query::physical::evaluate_expression;

    // Every row shares one schema at this point; used to resolve any
    // qualified `table.col` reference inside a windowed aggregate's inner
    // expression the same way `dsl_expr_to_logical_expr` resolves one
    // anywhere else. Empty rows means nothing below evaluates it anyway.
    let window_schema: std::sync::Arc<crate::core::tuple::Schema> = rows
        .first()
        .map(|r| r.schema.clone())
        .unwrap_or_else(|| std::sync::Arc::new(crate::core::tuple::Schema::new(vec![])));

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
                    let logical = dsl_expr_to_logical_expr(inner, &window_schema, right_tables);
                    let vals: Vec<Value> = sorted_indices[..=rank_0]
                        .iter()
                        .map(|&i| evaluate_expression(&logical, &rows[i]))
                        .collect();
                    window_running_sum(&vals, line_no)?
                }
                WindowFunc::Avg(inner) => {
                    let logical = dsl_expr_to_logical_expr(inner, &window_schema, right_tables);
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
                        // Caught by Phase 3's wildcard-arm audit -- without
                        // this, window AVG(complex_col) would silently
                        // return the running *sum* (undivided), not the
                        // average.
                        Value::Complex(s) => Value::Complex(s / count as f64),
                        other => other,
                    }
                }
                WindowFunc::Count(inner) => {
                    let logical = dsl_expr_to_logical_expr(inner, &window_schema, right_tables);
                    let cnt = sorted_indices[..=rank_0]
                        .iter()
                        .filter(|&&i| {
                            !matches!(evaluate_expression(&logical, &rows[i]), Value::Null)
                        })
                        .count();
                    Value::Int(cnt as i64)
                }
                WindowFunc::Min(inner) => {
                    let logical = dsl_expr_to_logical_expr(inner, &window_schema, right_tables);
                    let window_vals: Vec<Value> = sorted_indices[..=rank_0]
                        .iter()
                        .map(|&i| evaluate_expression(&logical, &rows[i]))
                        .filter(|v| !matches!(v, Value::Null))
                        .collect();
                    // Complex has no total order -- see the matching
                    // AggregateExec comment (query/physical.rs) for why a
                    // compare()-based min_by/max_by would silently return
                    // an arbitrary value here instead of erroring. Caught
                    // by Phase 3's wildcard-arm audit.
                    if window_vals.iter().any(|v| matches!(v, Value::Complex(_))) {
                        return Err(DslError::Engine {
                            line: line_no,
                            source: crate::engine::EngineError::InvalidOp(
                                "Window MIN: Complex values have no defined ordering".to_string(),
                            ),
                        });
                    }
                    window_vals
                        .into_iter()
                        .min_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(Value::Null)
                }
                WindowFunc::Max(inner) => {
                    let logical = dsl_expr_to_logical_expr(inner, &window_schema, right_tables);
                    let window_vals: Vec<Value> = sorted_indices[..=rank_0]
                        .iter()
                        .map(|&i| evaluate_expression(&logical, &rows[i]))
                        .filter(|v| !matches!(v, Value::Null))
                        .collect();
                    if window_vals.iter().any(|v| matches!(v, Value::Complex(_))) {
                        return Err(DslError::Engine {
                            line: line_no,
                            source: crate::engine::EngineError::InvalidOp(
                                "Window MAX: Complex values have no defined ordering".to_string(),
                            ),
                        });
                    }
                    window_vals
                        .into_iter()
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
            // Caught by Phase 3's wildcard-arm audit -- without this arm
            // (and the Some(Complex) continuation arm below), the first
            // Complex value in a window would silently seed the running
            // sum at Float(0.0), and every later Complex value would fall
            // to the `(Some(other), _) => other` catch-all below, silently
            // dropped instead of accumulated.
            (None, Value::Complex(c)) => Value::Complex(c),
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
            (Some(Value::Complex(s)), Value::Complex(c)) => Value::Complex(s + c),
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

    let before_hash = ds.content_hash();
    // Single dataset, no JOIN -- an empty right-table set makes any
    // `Expr::Field` qualifier resolve to its bare name, as before.
    let logical_expr =
        dsl_expr_to_logical_expr(expr, &ds.schema, &std::collections::HashSet::new());

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
            Value::BitVector(b) => ValueType::BitVector(b.len()),
            Value::SparseVector(sv) => ValueType::SparseVector(sv.dim()),
            Value::QVector(q) => ValueType::QVector(q.len(), q.encoding()),
            Value::Matrix(m) => {
                let r = m.len();
                let c = m.first().map_or(0, |row| row.len());
                ValueType::Matrix(r, c)
            }
            Value::Complex(_) => ValueType::Complex,
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
            Value::BitVector(b) => ValueType::BitVector(b.len()),
            Value::SparseVector(sv) => ValueType::SparseVector(sv.dim()),
            Value::QVector(q) => ValueType::QVector(q.len(), q.encoding()),
            Value::Matrix(m) => ValueType::Matrix(m.len(), m.first().map_or(0, |r| r.len())),
            Value::Complex(_) => ValueType::Complex,
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

    let after_ds = db.get_dataset(dataset).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let after_hash = after_ds.content_hash();
    let record = crate::core::provenance::ProvenanceRecord::new(
        "ADD COMPUTED COLUMN",
        crate::core::tensor::ExecutionId::new(),
    )
    .with_param("column", col_name)
    .with_inputs(vec![crate::core::provenance::ProvenanceEntity::dataset(
        dataset.to_string(),
        before_hash,
    )])
    .with_outputs(vec![crate::core::provenance::ProvenanceEntity::dataset(
        dataset.to_string(),
        after_hash,
    )]);
    db.active_instance_mut().record_provenance(record);

    Ok(DslOutput::Message(format!(
        "Added computed column '{}' to dataset '{}'",
        col_name, dataset
    )))
}

// ─── Shared logical plan helpers ──────────────────────────────────────────────

/// Lowers an aggregate function name to its logical form. `ARG_MAX`/
/// `ARG_MIN` carry a second (`by`) expression, lowered with the same
/// `lower` the caller uses for the aggregated expression itself.
pub(super) fn agg_func_to_logical(
    f: &AggFuncAst,
    lower: impl Fn(&Expr) -> LogicalExpr,
) -> AggregateFunction {
    match f {
        AggFuncAst::ArgMax(by) => AggregateFunction::ArgMax(Box::new(lower(by))),
        AggFuncAst::ArgMin(by) => AggregateFunction::ArgMin(Box::new(lower(by))),
        AggFuncAst::Rrf(k) => AggregateFunction::Rrf(*k),
        AggFuncAst::Sum => AggregateFunction::Sum,
        AggFuncAst::Avg => AggregateFunction::Avg,
        AggFuncAst::Count => AggregateFunction::Count,
        AggFuncAst::Min => AggregateFunction::Min,
        AggFuncAst::Max => AggregateFunction::Max,
        AggFuncAst::AvgVec => AggregateFunction::AvgVec,
        AggFuncAst::SumVec => AggregateFunction::SumVec,
        AggFuncAst::Variance => AggregateFunction::Variance,
        AggFuncAst::Median => AggregateFunction::Median,
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
    map_expr(expr, &|e| match e {
        Expr::Ref(name) => rename.get(name).map(|n| Expr::Ref(n.clone())),
        _ => None,
    })
}

/// Rebuilds `expr` bottom-up, except that any node for which `f` returns
/// `Some(replacement)` is replaced whole (its children aren't visited).
fn map_expr(expr: &Expr, f: &dyn Fn(&Expr) -> Option<Expr>) -> Expr {
    if let Some(replacement) = f(expr) {
        return replacement;
    }
    match expr {
        Expr::Ref(_)
        | Expr::Int(_)
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
            lhs: Box::new(map_expr(lhs, f)),
            rhs: Box::new(map_expr(rhs, f)),
        },
        Expr::And(l, r) => Expr::And(Box::new(map_expr(l, f)), Box::new(map_expr(r, f))),
        Expr::Or(l, r) => Expr::Or(Box::new(map_expr(l, f)), Box::new(map_expr(r, f))),
        Expr::Not(e) => Expr::Not(Box::new(map_expr(e, f))),
        Expr::IsNull(e) => Expr::IsNull(Box::new(map_expr(e, f))),
        Expr::IsNotNull(e) => Expr::IsNotNull(Box::new(map_expr(e, f))),
        Expr::In { expr, list } => Expr::In {
            expr: Box::new(map_expr(expr, f)),
            list: list.iter().map(|e| map_expr(e, f)).collect(),
        },
        Expr::Between { expr, low, high } => Expr::Between {
            expr: Box::new(map_expr(expr, f)),
            low: Box::new(map_expr(low, f)),
            high: Box::new(map_expr(high, f)),
        },
        Expr::Field { base, field } => Expr::Field {
            base: Box::new(map_expr(base, f)),
            field: field.clone(),
        },
        Expr::Case {
            operand,
            branches,
            else_expr,
        } => Expr::Case {
            operand: operand.as_ref().map(|e| Box::new(map_expr(e, f))),
            branches: branches
                .iter()
                .map(|(c, r)| (map_expr(c, f), map_expr(r, f)))
                .collect(),
            else_expr: else_expr.as_ref().map(|e| Box::new(map_expr(e, f))),
        },
        Expr::Coalesce(args) => Expr::Coalesce(args.iter().map(|e| map_expr(e, f)).collect()),
        Expr::Nullif(a, b) => Expr::Nullif(Box::new(map_expr(a, f)), Box::new(map_expr(b, f))),
        Expr::ScalarFn { func, args } => Expr::ScalarFn {
            func: *func,
            args: args.iter().map(|e| map_expr(e, f)).collect(),
        },
        Expr::Cast { expr, to } => Expr::Cast {
            expr: Box::new(map_expr(expr, f)),
            to: *to,
        },
        Expr::VectorFn { func, args } => Expr::VectorFn {
            func: *func,
            args: args.iter().map(|e| map_expr(e, f)).collect(),
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
        LogicalExpr::AggregateExpr { expr, func, .. } => {
            collect_referenced_columns(expr, out);
            if let AggregateFunction::ArgMax(by) | AggregateFunction::ArgMin(by) = func {
                collect_referenced_columns(by, out);
            }
        }
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
    right_tables: &std::collections::HashSet<String>,
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
    let predicate = dsl_expr_to_logical_expr(&rewritten, schema_now, right_tables);

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

/// Lowers the two operands of a binary operator. A decimal literal
/// (`Expr::Scalar`) normally becomes an f32 `Float`; next to a `Float64`
/// operand it keeps its full f64 value instead, so `double_col >= 0.1234567891`
/// or `mass BETWEEN q.mass - 0.005 AND ...` compare at full precision rather
/// than against an f32-rounded constant.
fn lower_pair(
    lhs: &Expr,
    rhs: &Expr,
    schema: &crate::core::tuple::Schema,
    right_tables: &std::collections::HashSet<String>,
) -> (LogicalExpr, LogicalExpr) {
    let left = dsl_expr_to_logical_expr(lhs, schema, right_tables);
    let right = dsl_expr_to_logical_expr(rhs, schema, right_tables);
    let lt = crate::query::logical::infer_expr_type_full(&left, schema);
    let rt = crate::query::logical::infer_expr_type_full(&right, schema);
    let left = if rt == ValueType::Float64 {
        lower_beside(lhs, &rt, schema, right_tables)
    } else {
        left
    };
    let right = if lt == ValueType::Float64 {
        lower_beside(rhs, &lt, schema, right_tables)
    } else {
        right
    };
    (left, right)
}

/// Lowers `e`, keeping a decimal literal at f64 when it sits beside a
/// `Float64` value (see `lower_pair`).
fn lower_beside(
    e: &Expr,
    other: &ValueType,
    schema: &crate::core::tuple::Schema,
    right_tables: &std::collections::HashSet<String>,
) -> LogicalExpr {
    // Arithmetic made of literals (`180.06 - 0.005`) beside a DOUBLE: the
    // DOUBLE context applies to its operands too.
    if let (
        Expr::Infix {
            op: op @ (InfixOp::Add | InfixOp::Subtract | InfixOp::Multiply | InfixOp::Divide),
            lhs,
            rhs,
        },
        ValueType::Float64,
    ) = (e, other)
    {
        let sym = match op {
            InfixOp::Add => "+",
            InfixOp::Subtract => "-",
            InfixOp::Multiply => "*",
            _ => "/",
        };
        return LogicalExpr::BinaryExpr {
            left: Box::new(lower_beside(lhs, other, schema, right_tables)),
            op: sym.to_string(),
            right: Box::new(lower_beside(rhs, other, schema, right_tables)),
        };
    }
    let lowered = dsl_expr_to_logical_expr(e, schema, right_tables);
    widen_scalar_literal(e, lowered, other)
}

fn widen_scalar_literal(source: &Expr, lowered: LogicalExpr, other: &ValueType) -> LogicalExpr {
    match (source, other) {
        (Expr::Scalar(f), ValueType::Float64) => LogicalExpr::Literal(Value::Float64(*f)),
        _ => lowered,
    }
}

/// Convert a parsed DSL `Expr` into a `LogicalExpr` the physical evaluator
/// understands. `schema` is the schema of the row(s) this expression will
/// actually be evaluated against, and `right_tables` is the set of dataset
/// names that sit on the right side of a `JOIN` in the current query (empty
/// outside `execute_select`, where no join/qualifier-collision is possible)
/// — both exist solely to resolve `Expr::Field` (a qualified `table.col`
/// reference) correctly; see that arm below.
pub(super) fn dsl_expr_to_logical_expr(
    e: &Expr,
    schema: &crate::core::tuple::Schema,
    right_tables: &std::collections::HashSet<String>,
) -> LogicalExpr {
    match e {
        Expr::Ref(name) => LogicalExpr::Column(name.clone()),
        // `table.col` — a JOIN whose two sides share a bare column name
        // gets its right-side field renamed to `r_<name>` in the merged
        // schema (see `LogicalPlan::Join::schema()`); resolve to that
        // renamed field when `base` names a table on the join's right side
        // and the collision actually happened (a colliding field really is
        // named `r_<field>` here), otherwise fall back to the bare name —
        // correct for the left side, for a right-side field with no
        // collision, and for every non-JOIN context (`right_tables` empty).
        Expr::Field { base, field } => {
            if let Expr::Ref(qualifier) = base.as_ref() {
                if right_tables.contains(qualifier) {
                    let renamed = format!("r_{field}");
                    if schema.get_field_index(&renamed).is_some() {
                        return LogicalExpr::Column(renamed);
                    }
                }
            }
            LogicalExpr::Column(field.clone())
        }
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
            let (left, right) = lower_pair(lhs, rhs, schema, right_tables);
            LogicalExpr::BinaryExpr {
                left: Box::new(left),
                op: sym.to_string(),
                right: Box::new(right),
            }
        }
        Expr::And(lhs, rhs) => LogicalExpr::And(
            Box::new(dsl_expr_to_logical_expr(lhs, schema, right_tables)),
            Box::new(dsl_expr_to_logical_expr(rhs, schema, right_tables)),
        ),
        Expr::Or(lhs, rhs) => LogicalExpr::Or(
            Box::new(dsl_expr_to_logical_expr(lhs, schema, right_tables)),
            Box::new(dsl_expr_to_logical_expr(rhs, schema, right_tables)),
        ),
        Expr::Not(inner) => LogicalExpr::Not(Box::new(dsl_expr_to_logical_expr(
            inner,
            schema,
            right_tables,
        ))),
        Expr::IsNull(inner) => LogicalExpr::IsNull(Box::new(dsl_expr_to_logical_expr(
            inner,
            schema,
            right_tables,
        ))),
        Expr::IsNotNull(inner) => LogicalExpr::IsNotNull(Box::new(dsl_expr_to_logical_expr(
            inner,
            schema,
            right_tables,
        ))),
        Expr::In { expr, list } => {
            let value = dsl_expr_to_logical_expr(expr, schema, right_tables);
            let value_type = crate::query::logical::infer_expr_type_full(&value, schema);
            LogicalExpr::In {
                list: list
                    .iter()
                    .map(|e| lower_beside(e, &value_type, schema, right_tables))
                    .collect(),
                expr: Box::new(value),
            }
        }
        Expr::Between { expr, low, high } => {
            let value = dsl_expr_to_logical_expr(expr, schema, right_tables);
            let value_type = crate::query::logical::infer_expr_type_full(&value, schema);
            LogicalExpr::Between {
                low: Box::new(lower_beside(low, &value_type, schema, right_tables)),
                high: Box::new(lower_beside(high, &value_type, schema, right_tables)),
                expr: Box::new(value),
            }
        }
        Expr::Case {
            operand,
            branches,
            else_expr,
        } => LogicalExpr::Case {
            operand: operand
                .as_ref()
                .map(|e| Box::new(dsl_expr_to_logical_expr(e, schema, right_tables))),
            branches: branches
                .iter()
                .map(|(c, r)| {
                    (
                        dsl_expr_to_logical_expr(c, schema, right_tables),
                        dsl_expr_to_logical_expr(r, schema, right_tables),
                    )
                })
                .collect(),
            else_expr: else_expr
                .as_ref()
                .map(|e| Box::new(dsl_expr_to_logical_expr(e, schema, right_tables))),
        },
        Expr::Coalesce(args) => LogicalExpr::Coalesce(
            args.iter()
                .map(|e| dsl_expr_to_logical_expr(e, schema, right_tables))
                .collect(),
        ),
        Expr::Nullif(a, b) => LogicalExpr::Nullif(
            Box::new(dsl_expr_to_logical_expr(a, schema, right_tables)),
            Box::new(dsl_expr_to_logical_expr(b, schema, right_tables)),
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
                args: args
                    .iter()
                    .map(|e| dsl_expr_to_logical_expr(e, schema, right_tables))
                    .collect(),
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
                CastTarget::QVector(n, e) => LCast::QVector(*n, *e),
                CastTarget::Matrix(r, c) => LCast::Matrix(*r, *c),
                CastTarget::BitVector(n) => LCast::BitVector(*n),
                CastTarget::SparseVector(n) => LCast::SparseVector(*n),
            };
            LogicalExpr::Cast {
                expr: Box::new(dsl_expr_to_logical_expr(expr, schema, right_tables)),
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
                VectorFnKind::Real => LVk::Real,
                VectorFnKind::Imag => LVk::Imag,
                VectorFnKind::ComplexAbs => LVk::ComplexAbs,
                VectorFnKind::Phase => LVk::Phase,
                VectorFnKind::Conj => LVk::Conj,
                VectorFnKind::ComplexNew => LVk::ComplexNew,
                VectorFnKind::Tanimoto => LVk::Tanimoto,
                VectorFnKind::Jaccard => LVk::Jaccard,
                VectorFnKind::Hamming => LVk::Hamming,
                VectorFnKind::BitCount => LVk::BitCount,
                VectorFnKind::SpecCosine => LVk::SpecCosine,
                VectorFnKind::SpecCosineMod => LVk::SpecCosineMod,
                VectorFnKind::SpecMatches => LVk::SpecMatches,
                VectorFnKind::SpecEntropy => LVk::SpecEntropy,
                VectorFnKind::SpecClean => LVk::SpecClean,
                VectorFnKind::SparseNew => LVk::SparseNew,
            };
            LogicalExpr::VectorFn {
                func: lfunc,
                args: args
                    .iter()
                    .map(|e| dsl_expr_to_logical_expr(e, schema, right_tables))
                    .collect(),
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
        source: Some(DatasetSource::Named(s.source.clone())),
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
        // The projection can change the column set (e.g. `TRANSFORM ...
        // SELECT id, UPPER(name) AS name_upper`), so the dataset's schema
        // has to move with its rows -- leaving the old schema in place
        // here left the dataset permanently broken (any later read failed
        // with a value-count mismatch against the stale schema).
        ds.schema = schema.clone();
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
    let invalid = |msg: String| DslError::Engine {
        line: line_no,
        source: crate::engine::EngineError::InvalidOp(msg),
    };
    let schema = db
        .get_dataset(&s.dataset)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?
        .schema
        .clone();

    // Single dataset, no JOIN -- an empty right-table set makes any
    // `Expr::Field` qualifier resolve to its bare name.
    let no_right_tables = std::collections::HashSet::new();
    let predicate = s
        .filter
        .as_ref()
        .map(|f| dsl_expr_to_logical_expr(f, &schema, &no_right_tables));
    // Assignments go through the same evaluator as SELECT/WHERE, so every
    // expression form works (vector literals, functions, CAST, ...).
    let mut assignments = Vec::with_capacity(s.assignments.len());
    for (col_name, expr) in &s.assignments {
        let idx = schema.get_field_index(col_name).ok_or_else(|| {
            invalid(format!(
                "UPDATE: unknown column '{}' in '{}'",
                col_name, s.dataset
            ))
        })?;
        let lowered = dsl_expr_to_logical_expr(expr, &schema, &no_right_tables);
        crate::query::typecheck::check_expr(&lowered, &schema).map_err(invalid)?;
        assignments.push((idx, lowered));
    }

    // Compute and type-check every new value before changing anything, so
    // a bad assignment leaves the dataset untouched.
    let ds = db.get_dataset(&s.dataset).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let mut changes: Vec<(usize, Vec<(usize, Value)>)> = Vec::new();
    crate::query::row_error::clear();
    for (row_idx, row) in ds.rows.iter().enumerate() {
        if let Some(pred) = &predicate {
            if !crate::query::planner::evaluate_predicate(pred, row) {
                continue;
            }
        }
        let mut new_values = Vec::with_capacity(assignments.len());
        for (col_idx, expr) in &assignments {
            let field = &schema.fields[*col_idx];
            let value = crate::query::physical::evaluate_expression(expr, row);
            if let Some(e) = crate::query::row_error::take() {
                return Err(invalid(format!(
                    "UPDATE '{}' row {}: {}",
                    s.dataset, row_idx, e
                )));
            }
            let value = coerce_for_field(value, field)
                .map_err(|e| invalid(format!("UPDATE '{}' row {}: {}", s.dataset, row_idx, e)))?;
            new_values.push((*col_idx, value));
        }
        changes.push((row_idx, new_values));
    }

    if let Some(e) = crate::query::row_error::take() {
        return Err(invalid(format!("UPDATE '{}': {}", s.dataset, e)));
    }
    let updated = changes.len();
    let changed_columns: Vec<String> = assignments
        .iter()
        .map(|(i, _)| schema.fields[*i].name.clone())
        .collect();
    let ds = db
        .get_dataset_mut(&s.dataset)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    for (row_idx, new_values) in changes {
        for (col_idx, value) in new_values {
            ds.rows[row_idx].values[col_idx] = value;
        }
    }
    if updated > 0 {
        ds.rebuild_after_mutation(Some(&changed_columns))
            .map_err(invalid)?;
    }

    Ok(DslOutput::Message(format!(
        "Updated {} row(s) in '{}'",
        updated, s.dataset
    )))
}

/// Fits an `UPDATE`'s computed value to its column's type: numeric values
/// widen or narrow between `Int`/`Float`/`Float64` the way `INSERT`'s
/// literals do (a non-integral number into an `Int` column is an error,
/// never truncated), `NULL` needs a nullable column, and anything else must
/// already match.
fn coerce_for_field(value: Value, field: &crate::core::tuple::Field) -> Result<Value, String> {
    let coerced = match (&field.value_type, value) {
        (_, Value::Null) => Value::Null,
        (ValueType::Float, Value::Int(i)) => Value::Float(i as f32),
        (ValueType::Float, Value::Float64(f)) => Value::Float(f as f32),
        (ValueType::Float64, Value::Int(i)) => Value::Float64(i as f64),
        (ValueType::Float64, Value::Float(f)) => Value::Float64(f as f64),
        (ValueType::Int, Value::Float(f)) if f.fract() == 0.0 && f.is_finite() => {
            Value::Int(f as i64)
        }
        (ValueType::Int, Value::Float64(f)) if f.fract() == 0.0 && f.is_finite() => {
            Value::Int(f as i64)
        }
        (_, v) => v,
    };
    if field.is_compatible(&coerced) {
        Ok(coerced)
    } else if coerced.is_null() {
        Err(format!(
            "column '{}' is not nullable, but the new value is NULL",
            field.name
        ))
    } else {
        Err(format!(
            "column '{}' is {:?}, but the new value is {:?}",
            field.name,
            field.value_type,
            coerced.value_type()
        ))
    }
}

// ─── DELETE ───────────────────────────────────────────────────────────────────

pub(super) fn execute_delete(
    db: &mut TensorDb,
    s: DeleteStmt,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    // Single dataset, no JOIN -- an empty right-table set/schema makes any
    // `Expr::Field` qualifier resolve to its bare name, as before.
    let predicate: Option<RowPredicate> = s.filter.as_ref().map(|f| -> RowPredicate {
        let logical = dsl_expr_to_logical_expr(
            f,
            &crate::core::tuple::Schema::new(vec![]),
            &std::collections::HashSet::new(),
        );
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
        Some(pred) => {
            // Decide every row first, so a data error deletes nothing.
            crate::query::row_error::clear();
            let doomed: Vec<bool> = ds.rows.iter().map(&pred).collect();
            if let Some(e) = crate::query::row_error::take() {
                return Err(DslError::Engine {
                    line: line_no,
                    source: crate::engine::EngineError::InvalidOp(e),
                });
            }
            let mut doomed = doomed.into_iter();
            ds.rows.retain(|_| !doomed.next().unwrap_or(false));
        }
        None => ds.rows.clear(),
    }
    let deleted = before - ds.rows.len();
    if deleted > 0 {
        // Row ids shifted: every index, zone map and stat is stale.
        ds.rebuild_after_mutation(None)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::InvalidOp(e),
            })?;
    }

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
            // Any pairing touching Complex promotes to Complex -- checked
            // before Float64 below, same priority reasoning as
            // query/physical.rs's evaluate_expression.
            if matches!(l, Value::Complex(_)) || matches!(r, Value::Complex(_)) {
                return match (l.as_complex(), r.as_complex()) {
                    (Some(a), Some(b)) => match op {
                        InfixOp::Add => Value::Complex(a + b),
                        InfixOp::Subtract => Value::Complex(a - b),
                        InfixOp::Multiply => Value::Complex(a * b),
                        InfixOp::Divide => Value::Complex(a / b),
                        _ => Value::Null,
                    },
                    _ => Value::Null,
                };
            }
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
        //
        // Same gap covers `Expr::VectorFn` (COSINE_SIM/DOT/..., and Phase
        // 3's REAL/IMAG/ABS/PHASE/CONJ/COMPLEX) -- a computed/LAZY `ADD
        // COLUMN` using any of these also silently evaluates to Null,
        // pre-existing and not specific to the Complex additions (this
        // function has never dispatched any `VectorFn`). `SELECT`/`WHERE`/
        // ordinary computed columns are unaffected -- they route through
        // `query::physical::evaluate_expression`, which does dispatch
        // `VectorFn` (including the new Complex functions) correctly.
        _ => Value::Null,
    }
}
