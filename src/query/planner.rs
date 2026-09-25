use crate::core::tuple::Schema;
use crate::engine::{EngineError, TensorDb};
use crate::query::logical::{Expr, LogicalPlan};
use crate::query::physical::{
    AggregateExec, CosineFilterExec, DistinctExec, FilterExec, HashJoinExec, IndexScanExec,
    LimitExec, PartitionPrunedScanExec, PhysicalPlan, ProjectionExec, SeqScanExec,
    SimilarityJoinExec, SortExec, UnionExec, ValuesExec, VectorSearchExec,
};
use std::sync::Arc;

/// The parts of a `COSINE_SIM(col, query_vec) <op> threshold` conjunct
/// `Planner::match_cosine_threshold` extracts, once a matching `VECTOR`
/// index on `column` is confirmed to exist.
struct CosineThresholdMatch {
    column: String,
    query: Vec<f32>,
    threshold: f32,
    strict: bool,
}

pub struct Planner<'a> {
    db: &'a TensorDb,
}

impl<'a> Planner<'a> {
    pub fn new(db: &'a TensorDb) -> Self {
        Self { db }
    }

    pub fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<Box<dyn PhysicalPlan>, EngineError> {
        match logical_plan {
            LogicalPlan::Scan {
                dataset_name,
                schema,
            } => Ok(Box::new(SeqScanExec {
                dataset_name: dataset_name.clone(),
                schema: schema.clone(),
            })),
            LogicalPlan::Values { schema, rows, .. } => Ok(Box::new(ValuesExec {
                schema: schema.clone(),
                rows: rows.clone(),
            })),
            LogicalPlan::Filter { input, predicate } => {
                // OPTIMIZATION: Check if we can use an Index (replaces the
                // scan+filter outright -- these executors already apply the
                // full predicate themselves).
                if let LogicalPlan::Scan {
                    dataset_name,
                    schema,
                } = input.as_ref()
                {
                    if let Some(index_plan) =
                        self.try_optimize_filter(dataset_name, schema, predicate)
                    {
                        return Ok(index_plan);
                    }
                }

                // OPTIMIZATION: partition pruning. Unlike the index case
                // above, this only narrows which rows the scan below this
                // Filter produces -- the real predicate still has to run
                // afterward, since partition stats only prove a partition
                // *might* contain matches, never that every row in it does.
                let input_plan = match input.as_ref() {
                    LogicalPlan::Scan {
                        dataset_name,
                        schema,
                    } => match self.try_prune_partitions(dataset_name, schema, predicate) {
                        Some(pruned) => pruned,
                        None => self.create_physical_plan(input)?,
                    },
                    _ => self.create_physical_plan(input)?,
                };

                // Default: Filter Scan
                // We need to convert logical Expr to a physical predicate closure
                // This is tricky because closures need to be generic or boxed.
                // For MVP, we'll implement a simple interpreter for Expr inside predicate.
                let predicate_clone = predicate.clone();
                let predicate_fn = Box::new(move |row: &crate::core::tuple::Tuple| {
                    evaluate_expr(&predicate_clone, row)
                });

                Ok(Box::new(FilterExec {
                    input: input_plan,
                    predicate: predicate_fn,
                }))
            }
            LogicalPlan::Project { input, columns } => {
                let input_plan = self.create_physical_plan(input)?;
                let input_schema = input_plan.schema();

                let column_indices: Vec<usize> = columns
                    .iter()
                    .map(|name| {
                        input_schema.get_field_index(name).ok_or_else(|| {
                            EngineError::InvalidOp(format!("Column not found: {}", name))
                        })
                    })
                    .collect::<Result<_, _>>()?;

                let output_fields = column_indices
                    .iter()
                    .map(|&idx| input_schema.fields[idx].clone())
                    .collect();
                let output_schema = Arc::new(Schema::new(output_fields));

                Ok(Box::new(ProjectionExec {
                    input: input_plan,
                    output_schema,
                    column_indices,
                }))
            }
            LogicalPlan::VectorSearch {
                input: _, // Vector Search usually is a leaf for now, or replaces Scan
                column,
                query,
                k,
            } => {
                // Vector Search replaces the Scan entirely if we are searching on a dataset
                // But wait, LogicalPlan::VectorSearch takes input.
                // Usually VectorSearch IS the access method.
                // Let's assume input is Scan.
                // If input is not Scan, we might need to materialize input first?
                // For MVP: assume input is Scan(dataset).

                match logical_plan {
                    LogicalPlan::VectorSearch {
                        input,
                        column: _,
                        query: _,
                        k: _,
                    } => {
                        if let LogicalPlan::Scan {
                            dataset_name,
                            schema,
                        } = input.as_ref()
                        {
                            let resolved_index_type = self
                                .db
                                .get_dataset(dataset_name)
                                .ok()
                                .and_then(|ds| ds.get_index(column))
                                .map(|idx| format!("{:?}", idx.index_type()));
                            Ok(Box::new(VectorSearchExec {
                                dataset_name: dataset_name.clone(),
                                schema: schema.clone(),
                                column: column.clone(),
                                query: query.clone(),
                                k: *k,
                                resolved_index_type,
                            }))
                        } else {
                            Err(EngineError::InvalidOp(
                                "VectorSearch input must be a Scan for now".into(),
                            ))
                        }
                    }
                    _ => unreachable!(),
                }
            }
            LogicalPlan::Limit { input, n, offset } => {
                let input_plan = self.create_physical_plan(input)?;
                Ok(Box::new(LimitExec {
                    input: input_plan,
                    n: *n,
                    offset: *offset,
                }))
            }
            LogicalPlan::Sort { input, columns } => {
                let input_plan = self.create_physical_plan(input)?;
                Ok(Box::new(SortExec {
                    input: input_plan,
                    columns: columns.clone(),
                }))
            }
            LogicalPlan::Aggregate {
                input,
                group_expr,
                aggr_expr,
            } => {
                let input_plan = self.create_physical_plan(input)?;
                let schema = logical_plan.schema();
                Ok(Box::new(AggregateExec {
                    input: input_plan,
                    group_expr: group_expr.clone(),
                    aggr_expr: aggr_expr.clone(),
                    schema,
                }))
            }
            LogicalPlan::Join {
                left,
                right,
                left_col,
                right_col,
                join_type,
                right_dataset_name,
                similarity_threshold,
            } => {
                let left_plan = self.create_physical_plan(left)?;
                let right_plan = self.create_physical_plan(right)?;
                // Mark all output fields as nullable so NULL-padded rows
                // (from LEFT/RIGHT/FULL OUTER JOIN unmatched sides) pass validation.
                let base_schema = logical_plan.schema();
                let nullable_fields: Vec<crate::core::tuple::Field> = base_schema
                    .fields
                    .iter()
                    .map(|f| {
                        let mut nf = f.clone();
                        nf.nullable = true;
                        nf
                    })
                    .collect();
                let output_schema = Arc::new(Schema::new(nullable_fields));
                if let Some(threshold) = similarity_threshold {
                    Ok(Box::new(SimilarityJoinExec {
                        left: left_plan,
                        right: right_plan,
                        left_col: left_col.clone(),
                        right_col: right_col.clone(),
                        right_dataset_name: right_dataset_name.clone(),
                        threshold: *threshold,
                        join_type: *join_type,
                        output_schema,
                    }))
                } else {
                    Ok(Box::new(HashJoinExec {
                        left: left_plan,
                        right: right_plan,
                        left_col: left_col.clone(),
                        right_col: right_col.clone(),
                        join_type: *join_type,
                        output_schema,
                    }))
                }
            }
            LogicalPlan::Union { left, right, all } => {
                let left_plan = self.create_physical_plan(left)?;
                let right_plan = self.create_physical_plan(right)?;
                Ok(Box::new(UnionExec {
                    left: left_plan,
                    right: right_plan,
                    all: *all,
                }))
            }
            LogicalPlan::Distinct { input } => {
                let input_plan = self.create_physical_plan(input)?;
                Ok(Box::new(DistinctExec { input: input_plan }))
            }
        }
    }

    /// Pattern-matches a predicate shape and, only if a matching index
    /// actually exists on the referenced column, substitutes a specialized
    /// index-accelerated executor (`IndexScanExec`, `CosineFilterExec`)
    /// for the generic scan+filter path. Returns `None` otherwise, in
    /// which case the caller falls back to `SeqScanExec` + `FilterExec`,
    /// which evaluates the same predicate correctly without an index — so
    /// each specialized executor here is an optimization, not a second
    /// implementation of the predicate's semantics.
    fn try_optimize_filter(
        &self,
        dataset_name: &str,
        schema: &Schema,
        predicate: &Expr,
    ) -> Option<Box<dyn PhysicalPlan>> {
        if let Expr::BinaryExpr { left, op, right } = predicate {
            // Pattern 1: col = literal → hash index
            if op == "=" {
                if let (Expr::Column(col_name), Expr::Literal(val)) =
                    (left.as_ref(), right.as_ref())
                {
                    if let Ok(dataset) = self.db.get_dataset(dataset_name) {
                        if let Some(index) = dataset.get_index(col_name) {
                            if index.index_type() == crate::core::index::IndexType::Hash {
                                return Some(Box::new(IndexScanExec {
                                    dataset_name: dataset_name.to_string(),
                                    schema: Arc::new(schema.clone()),
                                    column: col_name.clone(),
                                    value: val.clone(),
                                }));
                            }
                        }
                    }
                }
            }

            // Pattern 2: COSINE_SIM(col, query_vec) > threshold → vector index,
            // the predicate's only conjunct.
            if let Some(m) = self.match_cosine_threshold(dataset_name, predicate) {
                return Some(Box::new(CosineFilterExec {
                    dataset_name: dataset_name.to_string(),
                    schema: Arc::new(schema.clone()),
                    column: m.column,
                    query: m.query,
                    threshold: m.threshold,
                    strict: m.strict,
                }));
            }
        }

        // Pattern 3: a top-level AND with a COSINE_SIM(...) > threshold conjunct
        // anywhere in it, index-accelerated, remaining conjuncts applied as a
        // post-filter -- fixes `WHERE COSINE_SIM(...) > t AND category = 'x'`
        // identically for a plain SELECT and for SEARCH's FILTER clause (both
        // route through this same LogicalPlan::Filter path), where previously
        // the whole predicate fell through to full SeqScanExec+FilterExec
        // brute force the moment it wasn't *exactly* the cosine comparison
        // alone. Deliberately implemented as physical-plan-only composition
        // (CosineFilterExec wrapped in FilterExec) rather than a new
        // LogicalPlan variant, matching this planner's existing convention
        // for every other index optimization here (IndexScanExec/
        // CosineFilterExec/PartitionPrunedScanExec are all physical-only too)
        // -- whether an index exists is runtime state the *logical* plan
        // shouldn't need to know about.
        if matches!(predicate, Expr::And(_, _)) {
            let conjuncts = flatten_and(predicate);
            for (i, conjunct) in conjuncts.iter().enumerate() {
                if let Some(m) = self.match_cosine_threshold(dataset_name, conjunct) {
                    let cosine_exec = Box::new(CosineFilterExec {
                        dataset_name: dataset_name.to_string(),
                        schema: Arc::new(schema.clone()),
                        column: m.column,
                        query: m.query,
                        threshold: m.threshold,
                        strict: m.strict,
                    });

                    let remaining: Vec<Expr> = conjuncts
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, e)| (*e).clone())
                        .collect();

                    return Some(match rebuild_and(remaining) {
                        Some(post_filter) => {
                            let predicate_fn = Box::new(move |row: &crate::core::tuple::Tuple| {
                                evaluate_expr(&post_filter, row)
                            });
                            Box::new(FilterExec {
                                input: cosine_exec,
                                predicate: predicate_fn,
                            })
                        }
                        // Every other conjunct was itself trivially always-true
                        // (shouldn't happen from real DSL input, but `flatten_and`
                        // doesn't assume it can't) -- the cosine exec alone is
                        // already the complete answer.
                        None => cosine_exec,
                    });
                }
            }
        }

        None
    }

    /// Recognizes `COSINE_SIM(col, query_vec) <op> threshold` (`>`/`>=`) against
    /// a column with an existing `VECTOR` index -- the shape both the
    /// single-predicate case and the AND-decomposition case above need to
    /// match identically. Returns `None` on any shape mismatch *or* a missing/
    /// wrong-type index (the caller falls back to brute force either way, same
    /// as before this existed).
    fn match_cosine_threshold(
        &self,
        dataset_name: &str,
        conjunct: &Expr,
    ) -> Option<CosineThresholdMatch> {
        let Expr::BinaryExpr { left, op, right } = conjunct else {
            return None;
        };
        if op != ">" && op != ">=" {
            return None;
        }
        let Expr::VectorFn {
            func: crate::query::logical::VectorFnKind::CosineSim,
            args,
        } = left.as_ref()
        else {
            return None;
        };
        if args.len() != 2 {
            return None;
        }
        let (Expr::Column(col_name), Expr::Literal(crate::core::value::Value::Vector(qvec))) =
            (&args[0], &args[1])
        else {
            return None;
        };
        let threshold = match right.as_ref() {
            Expr::Literal(crate::core::value::Value::Float(f)) => *f,
            Expr::Literal(crate::core::value::Value::Int(i)) => *i as f32,
            _ => return None,
        };

        let dataset = self.db.get_dataset(dataset_name).ok()?;
        let index = dataset.get_index(col_name)?;
        if index.index_type() != crate::core::index::IndexType::Vector {
            return None;
        }

        Some(CosineThresholdMatch {
            column: col_name.clone(),
            query: qvec.clone(),
            threshold,
            strict: op == ">",
        })
    }

    /// Recognizes `col <op> literal` (`<`, `<=`, `>`, `>=`, either operand
    /// order) shapes and, if the dataset has more than one partition's worth
    /// of per-column zone-map stats (`Dataset.partitions`, maintained by
    /// `Dataset::rebuild_partitions`/`add_row`), wraps the scan in a
    /// `PartitionPrunedScanExec` covering only the partitions whose
    /// `[min, max]` can't be ruled out. Returns `None` (falls back to a
    /// plain `SeqScanExec`) when there's nothing to prune, so the caller
    /// never has to reason about whether pruning "worked" -- the wrapping
    /// `FilterExec` still applies the real predicate either way.
    fn try_prune_partitions(
        &self,
        dataset_name: &str,
        schema: &Schema,
        predicate: &Expr,
    ) -> Option<Box<dyn PhysicalPlan>> {
        let (col_name, constraints) = extract_range_constraints(predicate)?;

        let dataset = self.db.get_dataset(dataset_name).ok()?;
        if dataset.partitions.len() < 2 {
            return None; // nothing to prune (or not worth a specialized exec)
        }

        let surviving: Vec<(usize, usize)> = dataset
            .partitions
            .iter()
            .filter(|p| {
                p.column_stats
                    .get(&col_name)
                    .map(|stats| match (&stats.min, &stats.max) {
                        (Some(min), Some(max)) => constraints.iter().all(|(op, literal)| {
                            partition_range_could_match(op, min, max, literal)
                        }),
                        // No non-null values seen in this partition for the
                        // column (e.g. all null) -- can't prove it's
                        // skippable, so keep it.
                        _ => true,
                    })
                    .unwrap_or(true) // no stats for this column -- keep, can't prove skippable
            })
            .map(|p| (p.start, p.end))
            .collect();

        if surviving.len() == dataset.partitions.len() {
            return None; // nothing prunable -- let the normal SeqScanExec path run
        }

        Some(Box::new(PartitionPrunedScanExec {
            dataset_name: dataset_name.to_string(),
            schema: Arc::new(schema.clone()),
            row_ranges: surviving,
        }))
    }
}

/// Extracts a single column name plus one or more `(op, literal)` range
/// constraints on it from a predicate shape this pruning pass understands:
/// a plain comparison (`col <op> literal`, either operand order) or
/// `col BETWEEN low AND high` (treated as the conjunction `col >= low AND
/// col <= high`, both of which must hold for a partition to survive). Each
/// `op` is always expressed relative to the column -- already flipped if
/// the literal was on the left of a `BinaryExpr`.
fn extract_range_constraints(
    predicate: &Expr,
) -> Option<(String, Vec<(String, crate::core::value::Value)>)> {
    match predicate {
        Expr::BinaryExpr { left, op, right } if matches!(op.as_str(), "<" | "<=" | ">" | ">=") => {
            match (left.as_ref(), right.as_ref()) {
                (Expr::Column(c), Expr::Literal(v)) => {
                    Some((c.clone(), vec![(op.clone(), v.clone())]))
                }
                (Expr::Literal(v), Expr::Column(c)) => Some((
                    c.clone(),
                    vec![(flip_comparison(op).to_string(), v.clone())],
                )),
                _ => None,
            }
        }
        Expr::Between { expr, low, high } => {
            let Expr::Column(c) = expr.as_ref() else {
                return None;
            };
            let Expr::Literal(low_v) = low.as_ref() else {
                return None;
            };
            let Expr::Literal(high_v) = high.as_ref() else {
                return None;
            };
            Some((
                c.clone(),
                vec![
                    (">=".to_string(), low_v.clone()),
                    ("<=".to_string(), high_v.clone()),
                ],
            ))
        }
        _ => None,
    }
}

/// Flattens a (possibly nested, left-associative) `AND` tree into its
/// individual conjuncts, e.g. `And(And(a, b), c)` (how `a AND b AND c`
/// parses) becomes `[a, b, c]`. A non-`And` expression flattens to the
/// single-element `[expr]`.
fn flatten_and(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::And(left, right) => {
            let mut conjuncts = flatten_and(left);
            conjuncts.extend(flatten_and(right));
            conjuncts
        }
        other => vec![other],
    }
}

/// Inverse of `flatten_and` (on owned conjuncts): rebuilds one predicate
/// `Expr` from a list, right-folding into nested `And`s. `None` for an
/// empty list (nothing left to filter by), the single element itself for a
/// one-element list (no `And` wrapper needed).
fn rebuild_and(mut conjuncts: Vec<Expr>) -> Option<Expr> {
    let mut result = conjuncts.pop()?;
    while let Some(next) = conjuncts.pop() {
        result = Expr::And(Box::new(next), Box::new(result));
    }
    Some(result)
}

/// Flips a comparison operator to restate `literal <op> col` as `col <op'>
/// literal`.
fn flip_comparison(op: &str) -> &str {
    match op {
        "<" => ">",
        "<=" => ">=",
        ">" => "<",
        ">=" => "<=",
        other => other,
    }
}

/// Could any value in a partition's `[min, max]` column range possibly
/// satisfy `col <op> literal`? Conservative: an incomparable pair (`None`
/// from `Value::compare`, e.g. mismatched types) is treated as "could
/// match" so this only ever prunes when it can prove a partition can't
/// contain a match.
fn partition_range_could_match(
    op: &str,
    min: &crate::core::value::Value,
    max: &crate::core::value::Value,
    literal: &crate::core::value::Value,
) -> bool {
    use std::cmp::Ordering;
    match op {
        "<" => !matches!(
            min.compare(literal),
            Some(Ordering::Greater) | Some(Ordering::Equal)
        ),
        "<=" => min.compare(literal) != Some(Ordering::Greater),
        ">" => !matches!(
            max.compare(literal),
            Some(Ordering::Less) | Some(Ordering::Equal)
        ),
        ">=" => max.compare(literal) != Some(Ordering::Less),
        _ => true,
    }
}

/// Public entry point for evaluating a logical predicate against a row.
pub fn evaluate_predicate(expr: &Expr, row: &crate::core::tuple::Tuple) -> bool {
    evaluate_expr(expr, row)
}

fn evaluate_expr(expr: &Expr, row: &crate::core::tuple::Tuple) -> bool {
    match expr {
        // A bare column reference used as a whole predicate (`WHERE active`,
        // no explicit `= true`) is only meaningful for a Bool column -- any
        // other type has no defined truthiness, so it never matches (same
        // "never silently wrong, just doesn't match" stance as an
        // incomparable BinaryExpr below).
        Expr::Column(name) => {
            matches!(row.get(name), Some(crate::core::value::Value::Bool(true)))
        }
        Expr::And(left, right) => evaluate_expr(left, row) && evaluate_expr(right, row),
        Expr::Or(left, right) => evaluate_expr(left, row) || evaluate_expr(right, row),
        Expr::Not(inner) => !evaluate_expr(inner, row),
        Expr::IsNull(inner) => matches!(
            eval_value(inner, row),
            Some(crate::core::value::Value::Null) | None
        ),
        Expr::IsNotNull(inner) => !matches!(
            eval_value(inner, row),
            Some(crate::core::value::Value::Null) | None
        ),
        Expr::In { expr, list } => {
            if let Some(val) = eval_value(expr, row) {
                list.iter().any(|item| {
                    eval_value(item, row)
                        .map(|v| val.equals(&v) == Some(true))
                        .unwrap_or(false)
                })
            } else {
                false
            }
        }
        Expr::Between { expr, low, high } => {
            let val = eval_value(expr, row);
            let lo = eval_value(low, row);
            let hi = eval_value(high, row);
            if let (Some(v), Some(l), Some(h)) = (val, lo, hi) {
                let ge = matches!(
                    v.compare(&l),
                    Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
                );
                let le = matches!(
                    v.compare(&h),
                    Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
                );
                ge && le
            } else {
                false
            }
        }
        Expr::BinaryExpr { left, op, right } => {
            let left_val = eval_value(left, row);
            let right_val = eval_value(right, row);

            if let (Some(l), Some(r)) = (left_val, right_val) {
                if op == "=" || op == "!=" {
                    // Value::equals(), not compare()'s Ordering -- see its
                    // doc comment (Complex has real equality but no order).
                    let eq = l.equals(&r);
                    return if op == "=" {
                        eq == Some(true)
                    } else {
                        eq == Some(false)
                    };
                }
                let ord = l.compare(&r);
                match op.as_str() {
                    ">" => ord == Some(std::cmp::Ordering::Greater),
                    "<" => ord == Some(std::cmp::Ordering::Less),
                    ">=" => matches!(
                        ord,
                        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
                    ),
                    "<=" => matches!(
                        ord,
                        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
                    ),
                    _ => false,
                }
            } else {
                false
            }
        }
        Expr::Case {
            operand,
            branches,
            else_expr,
        } => {
            let operand_val = operand
                .as_ref()
                .map(|e| eval_value(e, row).unwrap_or(crate::core::value::Value::Null));
            for (cond, result) in branches {
                let matched = if let Some(ref ov) = operand_val {
                    eval_value(cond, row)
                        .map(|cv| ov.compare(&cv) == Some(std::cmp::Ordering::Equal))
                        .unwrap_or(false)
                } else {
                    evaluate_expr(cond, row)
                };
                if matched {
                    return evaluate_expr(result, row);
                }
            }
            else_expr.as_ref().is_some_and(|e| evaluate_expr(e, row))
        }
        Expr::Coalesce(args) => {
            for arg in args {
                if let Some(v) = eval_value(arg, row) {
                    if !v.is_null() {
                        return true;
                    }
                }
            }
            false
        }
        Expr::ScalarFn { .. } | Expr::Cast { .. } | Expr::Nullif(_, _) => false,
        _ => false,
    }
}

fn eval_value(expr: &Expr, row: &crate::core::tuple::Tuple) -> Option<crate::core::value::Value> {
    use crate::query::physical::evaluate_expression;
    match expr {
        Expr::Column(name) => row.get(name).cloned(),
        Expr::Literal(val) => Some(val.clone()),
        // For complex exprs, delegate to the full evaluator
        Expr::ScalarFn { .. }
        | Expr::Cast { .. }
        | Expr::Case { .. }
        | Expr::Coalesce(_)
        | Expr::Nullif(_, _)
        | Expr::VecLiteral(_)
        | Expr::VectorFn { .. } => Some(evaluate_expression(expr, row)),
        _ => None,
    }
}

#[cfg(test)]
mod partition_pruning_tests {
    use super::*;
    use crate::core::tuple::{Field, Tuple};
    use crate::core::value::{Value, ValueType};
    use crate::engine::TensorDb;

    fn int(v: i64) -> Value {
        Value::Int(v)
    }

    #[test]
    fn range_could_match_is_correct_at_boundaries() {
        use std::cmp::Ordering;
        // min=10, max=20
        let (min, max) = (int(10), int(20));

        // "<" possible iff min < literal
        assert!(partition_range_could_match("<", &min, &max, &int(11))); // min(10) < 11
        assert!(!partition_range_could_match("<", &min, &max, &int(10))); // min(10) !< 10

        // "<=" possible iff min <= literal
        assert!(partition_range_could_match("<=", &min, &max, &int(10)));
        assert!(!partition_range_could_match("<=", &min, &max, &int(9)));

        // ">" possible iff max > literal
        assert!(partition_range_could_match(">", &min, &max, &int(19))); // max(20) > 19
        assert!(!partition_range_could_match(">", &min, &max, &int(20))); // max(20) !> 20

        // ">=" possible iff max >= literal
        assert!(partition_range_could_match(">=", &min, &max, &int(20)));
        assert!(!partition_range_could_match(">=", &min, &max, &int(21)));

        // Incomparable values (mismatched types) must never be used to prune.
        let mismatched = Value::String("x".to_string());
        assert_eq!(min.compare(&mismatched), None);
        assert!(partition_range_could_match("<", &min, &max, &mismatched));
        assert!(partition_range_could_match(">", &min, &max, &mismatched));

        let _ = Ordering::Less; // silence unused import if the above shrinks later
    }

    #[test]
    fn flip_comparison_swaps_direction() {
        assert_eq!(flip_comparison("<"), ">");
        assert_eq!(flip_comparison("<="), ">=");
        assert_eq!(flip_comparison(">"), "<");
        assert_eq!(flip_comparison(">="), "<=");
    }

    #[test]
    fn extract_range_constraints_handles_both_operand_orders_and_between() {
        let col_lit = Expr::BinaryExpr {
            left: Box::new(Expr::Column("age".to_string())),
            op: ">".to_string(),
            right: Box::new(Expr::Literal(int(30))),
        };
        let (col, constraints) = extract_range_constraints(&col_lit).unwrap();
        assert_eq!(col, "age");
        assert_eq!(constraints, vec![(">".to_string(), int(30))]);

        // Literal on the left: `30 < age` means `age > 30`.
        let lit_col = Expr::BinaryExpr {
            left: Box::new(Expr::Literal(int(30))),
            op: "<".to_string(),
            right: Box::new(Expr::Column("age".to_string())),
        };
        let (col, constraints) = extract_range_constraints(&lit_col).unwrap();
        assert_eq!(col, "age");
        assert_eq!(constraints, vec![(">".to_string(), int(30))]);

        let between = Expr::Between {
            expr: Box::new(Expr::Column("age".to_string())),
            low: Box::new(Expr::Literal(int(10))),
            high: Box::new(Expr::Literal(int(20))),
        };
        let (col, constraints) = extract_range_constraints(&between).unwrap();
        assert_eq!(col, "age");
        assert_eq!(
            constraints,
            vec![(">=".to_string(), int(10)), ("<=".to_string(), int(20))]
        );

        // Equality isn't a range predicate this pass handles.
        let eq = Expr::BinaryExpr {
            left: Box::new(Expr::Column("age".to_string())),
            op: "=".to_string(),
            right: Box::new(Expr::Literal(int(30))),
        };
        assert!(extract_range_constraints(&eq).is_none());
    }

    /// Builds a dataset with `row_count` rows (`id` ascending from 0),
    /// spanning several `BATCH_SIZE`-sized partitions, for pruning tests.
    fn dataset_with_ascending_ids(db: &mut TensorDb, name: &str, row_count: i64) {
        let schema = Arc::new(Schema::new(vec![Field::new("id", ValueType::Int)]));
        db.create_dataset(name.to_string(), schema.clone()).unwrap();
        for i in 0..row_count {
            db.insert_row(name, Tuple::new(schema.clone(), vec![int(i)]).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn try_prune_partitions_skips_partitions_that_cannot_match() {
        let mut db = TensorDb::new();
        // 2500 rows -> partitions [0,1024), [1024,2048), [2048,2500) (BATCH_SIZE=1024).
        dataset_with_ascending_ids(&mut db, "t", 2500);
        assert_eq!(db.get_dataset("t").unwrap().partitions.len(), 3);

        let planner = Planner::new(&db);
        let schema = (*db.get_dataset("t").unwrap().schema).clone();

        // Only the last partition (max=2499) can contain id > 2400; the
        // first two (max 1023 and 2047) are provably out of range.
        let predicate = Expr::BinaryExpr {
            left: Box::new(Expr::Column("id".to_string())),
            op: ">".to_string(),
            right: Box::new(Expr::Literal(int(2400))),
        };
        let plan = planner
            .try_prune_partitions("t", &schema, &predicate)
            .expect("expected pruning to activate");

        let rows = plan.execute(&db).unwrap();
        // The pruned scan itself returns every row in the surviving
        // partition(s) (2048..2500 = 452 rows) -- it narrows candidates,
        // it doesn't apply the predicate. If pruning hadn't fired, this
        // would be 2500 (every row in the dataset).
        assert_eq!(rows.len(), 452);
    }

    #[test]
    fn try_prune_partitions_returns_none_when_every_partition_could_match() {
        let mut db = TensorDb::new();
        dataset_with_ascending_ids(&mut db, "t", 2500);

        let planner = Planner::new(&db);
        let schema = (*db.get_dataset("t").unwrap().schema).clone();

        // Every partition's range [0,1023]/[1024,2047]/[2048,2499] satisfies
        // id > -1, so nothing is prunable.
        let predicate = Expr::BinaryExpr {
            left: Box::new(Expr::Column("id".to_string())),
            op: ">".to_string(),
            right: Box::new(Expr::Literal(int(-1))),
        };
        assert!(planner
            .try_prune_partitions("t", &schema, &predicate)
            .is_none());
    }

    #[test]
    fn end_to_end_query_result_is_exact_regardless_of_pruning() {
        let mut db = TensorDb::new();
        dataset_with_ascending_ids(&mut db, "t", 2500);

        let logical = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Scan {
                dataset_name: "t".to_string(),
                schema: db.get_dataset("t").unwrap().schema.clone(),
            }),
            predicate: Expr::BinaryExpr {
                left: Box::new(Expr::Column("id".to_string())),
                op: ">".to_string(),
                right: Box::new(Expr::Literal(int(2400))),
            },
        };

        let planner = Planner::new(&db);
        let plan = planner.create_physical_plan(&logical).unwrap();
        let rows = plan.execute(&db).unwrap();

        // ids 2401..=2499 -> 99 rows. This must hold whether or not
        // partition pruning fired: pruning only narrows what the wrapping
        // FilterExec has to look at, it never changes the answer.
        assert_eq!(rows.len(), 99);
    }
}

#[cfg(test)]
mod filtered_vector_search_tests {
    use super::*;
    use crate::core::value::Value;

    fn col(name: &str) -> Expr {
        Expr::Column(name.to_string())
    }

    fn lit_str(s: &str) -> Expr {
        Expr::Literal(Value::String(s.to_string()))
    }

    fn cosine_gt(threshold: f32) -> Expr {
        Expr::BinaryExpr {
            left: Box::new(Expr::VectorFn {
                func: crate::query::logical::VectorFnKind::CosineSim,
                args: vec![col("emb"), Expr::Literal(Value::Vector(vec![1.0, 0.0]))],
            }),
            op: ">".to_string(),
            right: Box::new(Expr::Literal(Value::Float(threshold))),
        }
    }

    #[test]
    fn flatten_and_flattens_left_associative_chain() {
        // a AND b AND c parses left-associative: And(And(a, b), c)
        let a = col("a");
        let b = col("b");
        let c = col("c");
        let tree = Expr::And(
            Box::new(Expr::And(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(c.clone()),
        );
        let flat = flatten_and(&tree);
        assert_eq!(flat.len(), 3);
        assert!(matches!(flat[0], Expr::Column(n) if n == "a"));
        assert!(matches!(flat[1], Expr::Column(n) if n == "b"));
        assert!(matches!(flat[2], Expr::Column(n) if n == "c"));
    }

    #[test]
    fn flatten_and_of_non_and_is_single_element() {
        let e = col("x");
        assert_eq!(flatten_and(&e).len(), 1);
    }

    #[test]
    fn rebuild_and_round_trips_through_flatten() {
        let a = col("a");
        let b = col("b");
        let c = col("c");
        let tree = Expr::And(
            Box::new(Expr::And(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(c.clone()),
        );
        let flat: Vec<Expr> = flatten_and(&tree).into_iter().cloned().collect();
        let rebuilt = rebuild_and(flat).unwrap();
        assert_eq!(flatten_and(&rebuilt).len(), 3);
    }

    #[test]
    fn rebuild_and_of_empty_is_none() {
        assert!(rebuild_and(vec![]).is_none());
    }

    #[test]
    fn rebuild_and_of_one_is_that_element_unwrapped() {
        let single = rebuild_and(vec![col("a")]).unwrap();
        assert!(!matches!(single, Expr::And(_, _)));
    }

    fn dataset_with_vector_index(db: &mut TensorDb, name: &str) {
        let schema = std::sync::Arc::new(crate::core::tuple::Schema::new(vec![
            crate::core::tuple::Field::new("category", crate::core::value::ValueType::String),
            crate::core::tuple::Field::new("emb", crate::core::value::ValueType::Vector(2)),
        ]));
        db.create_dataset(name.to_string(), schema.clone()).unwrap();
        for (cat, v) in [("a", vec![1.0, 0.0]), ("b", vec![0.0, 1.0])] {
            db.insert_row(
                name,
                crate::core::tuple::Tuple::new(
                    schema.clone(),
                    vec![Value::String(cat.to_string()), Value::Vector(v)],
                )
                .unwrap(),
            )
            .unwrap();
        }
        db.create_vector_index(name, "emb").unwrap();
    }

    #[test]
    fn match_cosine_threshold_finds_it_anywhere_in_an_and_chain() {
        let mut db = TensorDb::new();
        dataset_with_vector_index(&mut db, "docs");

        let planner = Planner::new(&db);

        // Cosine conjunct first.
        let and1 = Expr::And(Box::new(cosine_gt(0.5)), Box::new(lit_str("a")));
        assert!(planner.match_cosine_threshold("docs", &and1).is_none()); // the AND itself isn't a BinaryExpr
        assert!(planner
            .match_cosine_threshold("docs", flatten_and(&and1)[0])
            .is_some());

        // Cosine conjunct second -- order must not matter for try_optimize_filter.
        let and2 = Expr::And(
            Box::new(Expr::BinaryExpr {
                left: Box::new(col("category")),
                op: "=".to_string(),
                right: Box::new(lit_str("a")),
            }),
            Box::new(cosine_gt(0.5)),
        );
        let plan =
            planner.try_optimize_filter("docs", &db.get_dataset("docs").unwrap().schema, &and2);
        assert!(
            plan.is_some(),
            "AND with the cosine conjunct in second position should still be accelerated"
        );
        let debug = format!("{:?}", plan.unwrap());
        assert!(
            debug.contains("CosineFilterExec"),
            "expected the accelerated exec nested inside, got: {debug}"
        );
    }

    #[test]
    fn and_without_a_cosine_conjunct_or_without_an_index_falls_back() {
        let mut db = TensorDb::new();
        let schema = std::sync::Arc::new(crate::core::tuple::Schema::new(vec![
            crate::core::tuple::Field::new("category", crate::core::value::ValueType::String),
            crate::core::tuple::Field::new("region", crate::core::value::ValueType::String),
        ]));
        db.create_dataset("plain".to_string(), schema).unwrap();

        let and = Expr::And(
            Box::new(Expr::BinaryExpr {
                left: Box::new(col("category")),
                op: "=".to_string(),
                right: Box::new(lit_str("a")),
            }),
            Box::new(Expr::BinaryExpr {
                left: Box::new(col("region")),
                op: "=".to_string(),
                right: Box::new(lit_str("us")),
            }),
        );
        let planner = Planner::new(&db);
        let plan =
            planner.try_optimize_filter("plain", &db.get_dataset("plain").unwrap().schema, &and);
        assert!(
            plan.is_none(),
            "no cosine conjunct at all -- nothing to accelerate"
        );
    }
}
