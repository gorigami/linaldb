use crate::core::tensor::Tensor;
use crate::core::tuple::Schema;
use crate::core::value::Value;
use std::sync::Arc;

/// Represents a filter expression
#[derive(Debug, Clone)]
pub enum Expr {
    /// Column reference
    Column(String),
    /// Constants
    Literal(Value),
    /// Binary operation (e.g. =, >, <, >=, <=)
    BinaryExpr {
        left: Box<Expr>,
        op: String,
        right: Box<Expr>,
    },
    /// Logical AND of two predicates
    And(Box<Expr>, Box<Expr>),
    /// Logical OR of two predicates
    Or(Box<Expr>, Box<Expr>),
    /// Logical NOT of a predicate
    Not(Box<Expr>),
    /// `col IS NULL`
    IsNull(Box<Expr>),
    /// `col IS NOT NULL`
    IsNotNull(Box<Expr>),
    /// `expr IN (v1, v2, ...)`
    In { expr: Box<Expr>, list: Vec<Expr> },
    /// `expr BETWEEN low AND high`
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
    },
    /// Aggregation function
    AggregateExpr {
        func: AggregateFunction,
        expr: Box<Expr>,
        /// Optional output column name from `AS alias` in the DSL.
        alias: Option<String>,
    },
    /// `CASE [operand] WHEN cond THEN result … [ELSE default] END`
    Case {
        operand: Option<Box<Expr>>,
        branches: Vec<(Expr, Expr)>,
        else_expr: Option<Box<Expr>>,
    },
    /// `COALESCE(e1, e2, …)`
    Coalesce(Vec<Expr>),
    /// `NULLIF(a, b)`
    Nullif(Box<Expr>, Box<Expr>),
    /// Scalar functions: UPPER, LOWER, LENGTH, TRIM, CONCAT, SUBSTR
    ScalarFn { func: ScalarFnKind, args: Vec<Expr> },
    /// `CAST(expr AS type)`
    Cast { expr: Box<Expr>, to: CastTarget },
    /// Inline vector literal: `[0.1, 0.2, 0.3]`
    VecLiteral(Vec<f64>),
    /// SQL-style vector function: `COSINE_SIM(emb, [0.1, 0.2])`, `NORMALIZE(emb)`
    VectorFn { func: VectorFnKind, args: Vec<Expr> },
    /// Inline matrix literal: `[[0.1, 0.2], [0.3, 0.4]]`
    MatLiteral(Vec<Vec<f64>>),
}

/// Vector/tensor functions usable in SQL expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorFnKind {
    Normalize,
    L2Norm,
    CosineSim,
    Dot,
    VecAdd,
    VecScale,
    Matmul,
    Transpose,
    MatShape,
    Flatten,
    /// `DISTANCE(a, b)` — Euclidean distance, SQL-callable form of the
    /// standalone `DISTANCE a TO b` tensor-DSL keyword (§3).
    Distance,
    Real,
    Imag,
    ComplexAbs,
    Phase,
    Conj,
    ComplexNew,
    Tanimoto,
    Jaccard,
    Hamming,
    BitCount,
    SpecCosine,
    SpecCosineMod,
    SpecMatches,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarFnKind {
    Upper,
    Lower,
    Length,
    Trim,
    Concat,
    Substr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastTarget {
    Int,
    Float,
    Double,
    Text,
    Bool,
    Vector(usize),
    Matrix(usize, usize),
    BitVector(Option<usize>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// The default (unaliased) output column name for an aggregate expression,
/// e.g. `AVG(score)` -- used both by `LogicalPlan::Aggregate::schema()`
/// below and by HAVING-clause alias resolution in `dsl::executor::query`,
/// so the two can never drift out of sync (a HAVING clause naming a bare
/// aggregate call must match exactly what an unaliased SELECT would have
/// produced for that same aggregate).
pub fn aggregate_default_name(func: &AggregateFunction, inner: &Expr) -> String {
    let col_name = match inner {
        Expr::Column(n) => n.clone(),
        _ => "val".to_string(),
    };
    let func_name = match func {
        AggregateFunction::ArgMax(_) => "ARG_MAX".to_string(),
        AggregateFunction::ArgMin(_) => "ARG_MIN".to_string(),
        AggregateFunction::Rrf(_) => "RRF".to_string(),
        other => format!("{:?}", other).to_uppercase(),
    };
    format!("{}({})", func_name, col_name)
}

#[derive(Debug, Clone)]
pub enum AggregateFunction {
    Sum,
    Avg,
    Count,
    Min,
    Max,
    /// Element-wise vector average across group rows
    AvgVec,
    /// Element-wise vector sum across group rows
    SumVec,
    /// Population variance of a scalar (Int/Float/Float64) column
    Variance,
    /// Median of a scalar (Int/Float/Float64) column
    Median,
    /// The aggregated expression's value on the group row with the largest
    /// `by` (the boxed expression). First row wins ties; NULL `by` is skipped.
    ArgMax(Box<Expr>),
    /// Same as `ArgMax`, for the smallest `by`.
    ArgMin(Box<Expr>),
    /// Reciprocal rank fusion: sum of `1 / (k + rank)` over the group.
    Rrf(f64),
}

#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan a dataset
    Scan {
        dataset_name: String,
        schema: Arc<Schema>,
    },
    /// Rows already computed earlier in the same query -- a CTE (`WITH name
    /// AS (...)`) or a `FROM (SELECT ...) AS name` subquery. They live only
    /// in the query that produced them and never touch the database's
    /// catalog, so a `SELECT` never needs to write to the database.
    /// `rows` is shared (`Arc`) so a CTE referenced several times isn't
    /// copied per reference. `name` is used only for display (`EXPLAIN`).
    Values {
        name: String,
        schema: Arc<Schema>,
        rows: Arc<Vec<crate::core::tuple::Tuple>>,
    },
    /// Filter rows
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    /// Projection (Select columns)
    Project {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
    },
    /// Vector Search (K-NN)
    VectorSearch {
        input: Box<LogicalPlan>,
        column: String,
        query: Tensor,
        k: usize,
    },
    /// Top-k for many query vectors at once (`SEARCH ... QUERIES`). Each
    /// query is `(query_id, vector)`; `schema` is the output schema:
    /// `query_id`, `rank`, `score`, `row_id`, then the dataset's columns.
    BatchVectorSearch {
        dataset_name: String,
        column: String,
        queries: Arc<QueryBatch>,
        k: usize,
        schema: Arc<Schema>,
        /// `PREFILTER`: when set, each query's top-k is an exact ranking
        /// over the rows passing the predicate, instead of an index search.
        prefilter: Option<Arc<Prefilter>>,
        /// Emit only the dataset's columns (a single-query `SEARCH ...
        /// PREFILTER`, which keeps plain `SEARCH`'s output shape).
        rows_only: bool,
    },
    /// Sort rows by one or more columns
    Sort {
        input: Box<LogicalPlan>,
        columns: Vec<(String, bool)>,
    },
    /// Limit rows (with optional offset)
    Limit {
        input: Box<LogicalPlan>,
        n: usize,
        offset: usize,
    },
    /// Join two datasets on an equi-join condition, or (when
    /// `similarity_threshold` is `Some`) a
    /// `COSINE_SIM(left_col, right_col) > threshold` similarity condition.
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        left_col: String,
        right_col: String,
        join_type: JoinType,
        /// The right side's dataset name — needed to look up a Vector
        /// index on `right_col` for an index-accelerated similarity join.
        right_dataset_name: String,
        similarity_threshold: Option<f32>,
    },
    /// Aggregate rows
    Aggregate {
        input: Box<LogicalPlan>,
        group_expr: Vec<Expr>,
        aggr_expr: Vec<Expr>,
    },
    /// UNION / UNION ALL of two query results
    Union {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        all: bool,
    },
    /// DISTINCT — deduplicate rows
    Distinct { input: Box<LogicalPlan> },
}

impl LogicalPlan {
    pub fn schema(&self) -> Arc<Schema> {
        match self {
            LogicalPlan::Scan { schema, .. } => schema.clone(),
            LogicalPlan::Values { schema, .. } => schema.clone(),
            LogicalPlan::Filter { input, .. } => input.schema(),
            LogicalPlan::Project { input, columns } => {
                let input_schema = input.schema();
                // Construct new schema from selected columns
                // This is a simplification; normally we'd validate here or during construction
                let fields = columns
                    .iter()
                    .filter_map(|name| input_schema.get_field(name).cloned())
                    .collect();
                Arc::new(Schema::new(fields))
            }
            LogicalPlan::VectorSearch { input, .. } => input.schema(),
            LogicalPlan::BatchVectorSearch { schema, .. } => schema.clone(),
            LogicalPlan::Sort { input, .. } => input.schema(),
            LogicalPlan::Limit { input, .. } => input.schema(),
            LogicalPlan::Join { left, right, .. } => {
                let left_schema = left.schema();
                let right_schema = right.schema();
                let left_names: std::collections::HashSet<&str> =
                    left_schema.fields.iter().map(|f| f.name.as_str()).collect();
                let mut fields = left_schema.fields.clone();
                for f in &right_schema.fields {
                    if left_names.contains(f.name.as_str()) {
                        let mut renamed = f.clone();
                        renamed.name = format!("r_{}", f.name);
                        fields.push(renamed);
                    } else {
                        fields.push(f.clone());
                    }
                }
                Arc::new(Schema::new(fields))
            }
            LogicalPlan::Union { left, .. } => left.schema(),
            LogicalPlan::Distinct { input } => input.schema(),
            LogicalPlan::Aggregate {
                input,
                group_expr,
                aggr_expr,
            } => {
                // Schema consists of Group keys + Aggregation results
                let mut fields = Vec::new();
                // 1. Group keys
                let input_schema = input.schema();
                for expr in group_expr {
                    if let Expr::Column(name) = expr {
                        let typ = infer_expr_type_full(expr, &input_schema);
                        fields.push(crate::core::tuple::Field::new(name, typ));
                    }
                }
                // 2. Aggregates
                for expr in aggr_expr {
                    if let Expr::AggregateExpr {
                        func,
                        expr: inner,
                        alias,
                    } = expr
                    {
                        let name = alias
                            .clone()
                            .unwrap_or_else(|| aggregate_default_name(func, inner));
                        let mut typ = crate::core::value::ValueType::Int; // Default

                        // Infer for SUM/MIN/MAX if inner is likely Vector (not perfect, but MVP)
                        match func {
                            super::logical::AggregateFunction::Sum
                            | super::logical::AggregateFunction::Min
                            | super::logical::AggregateFunction::Max => {
                                let input_schema = input.schema();
                                typ = infer_expr_type_full(inner.as_ref(), &input_schema);
                            }
                            super::logical::AggregateFunction::Avg => {
                                // AVG mirrors SUM/MIN/MAX's element-wise Vector/Matrix
                                // behavior (see AggregateExec's finalization in
                                // physical.rs), but a scalar Int/Float input always
                                // averages down to a Float, never an Int.
                                let input_schema = input.schema();
                                typ = match infer_expr_type_full(inner.as_ref(), &input_schema) {
                                    t @ (crate::core::value::ValueType::Vector(_)
                                    | crate::core::value::ValueType::Matrix(_, _)) => t,
                                    crate::core::value::ValueType::Float64 => {
                                        crate::core::value::ValueType::Float64
                                    }
                                    // Caught by Phase 3's wildcard-arm audit
                                    // -- without this, AVG(complex_col) would
                                    // report ValueType::Float in its schema
                                    // even though AggregateExec's real
                                    // finalization above correctly produces
                                    // a Value::Complex, a schema/value
                                    // mismatch that fails Tuple construction.
                                    crate::core::value::ValueType::Complex => {
                                        crate::core::value::ValueType::Complex
                                    }
                                    _ => crate::core::value::ValueType::Float,
                                };
                            }
                            super::logical::AggregateFunction::AvgVec
                            | super::logical::AggregateFunction::SumVec => {
                                // Was hardcoded to Vector(0) regardless of the
                                // real input width -- every AVG_VEC/SUM_VEC
                                // column reported dimension 0 in its schema
                                // (SHOW SCHEMA, error messages, this crate's
                                // own table-cell type header) even though the
                                // actual computed vector had the correct
                                // element count. Mirrors AVG's own inference
                                // just above.
                                let input_schema = input.schema();
                                typ = match infer_expr_type_full(inner.as_ref(), &input_schema) {
                                    t @ (crate::core::value::ValueType::Vector(_)
                                    | crate::core::value::ValueType::Matrix(_, _)) => t,
                                    _ => crate::core::value::ValueType::Vector(0),
                                };
                            }
                            super::logical::AggregateFunction::ArgMax(_)
                            | super::logical::AggregateFunction::ArgMin(_) => {
                                // The selected value keeps the aggregated
                                // expression's own type.
                                let input_schema = input.schema();
                                typ = infer_expr_type_full(inner.as_ref(), &input_schema);
                            }
                            super::logical::AggregateFunction::Rrf(_) => {
                                typ = crate::core::value::ValueType::Float64;
                            }
                            super::logical::AggregateFunction::Variance
                            | super::logical::AggregateFunction::Median => {
                                // Always Float64 -- both are computed in f64
                                // throughout (Welford's algorithm / sorted
                                // interpolation in `physical.rs`), regardless
                                // of the input column's own numeric type.
                                typ = crate::core::value::ValueType::Float64;
                            }
                            _ => {}
                        }

                        fields.push(crate::core::tuple::Field::new(&name, typ));
                    }
                }
                Arc::new(Schema::new(fields))
            }
        }
    }
}

/// Name prefix for a query's own columns inside a `PREFILTER` predicate
/// (`q.mass` becomes the column `"\0q.mass"`): a NUL can't appear in a real
/// column name, so these never collide with the searched dataset's columns.
pub const QUERY_COLUMN_PREFIX: &str = "\0q.";

/// A batch search's `(query_id, vector)` pairs. `Debug` prints only the
/// count, so `EXPLAIN` stays readable for thousands of queries.
#[derive(Clone)]
pub struct QueryBatch(pub Vec<(Value, Vec<f32>)>);

impl std::fmt::Debug for QueryBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} queries", self.0.len())
    }
}

/// A `SEARCH ... PREFILTER` predicate, ready to evaluate per (query, row).
#[derive(Clone)]
pub struct Prefilter {
    /// Over `combined_schema`: the searched dataset's columns followed by
    /// the query columns (`QUERY_COLUMN_PREFIX` + name).
    pub predicate: Expr,
    pub combined_schema: Arc<Schema>,
    /// Per query (aligned with the search's queries), the values of the
    /// query columns; empty when queries don't come from a dataset.
    pub query_values: Vec<Vec<Value>>,
    /// A conjunct the SORTED index on `column` can answer, with bounds that
    /// depend only on the query: `(column, [(op, bound expression)])`.
    pub sorted_range: Option<(String, Vec<(String, Expr)>)>,
    /// `APPROX`: rank the passing rows through the HNSW graph
    /// (`HnswIndex::search_filtered`) instead of an exact scan.
    pub approximate: bool,
}

// Leaves out the per-query values and the combined schema, which can be
// large; `EXPLAIN` only needs the predicate and how it's narrowed.
impl std::fmt::Debug for Prefilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prefilter")
            .field("predicate", &self.predicate)
            .field("sorted_range", &self.sorted_range)
            .field("approximate", &self.approximate)
            .finish()
    }
}

// Helper to fix BinaryExpr destructuring in infer_expr_type
pub(crate) fn infer_expr_type_full(expr: &Expr, schema: &Schema) -> crate::core::value::ValueType {
    use crate::core::value::ValueType;
    match expr {
        Expr::Column(name) => schema
            .get_field(name)
            .map(|f| f.value_type.clone())
            .unwrap_or(ValueType::Null),
        Expr::Literal(val) => val.value_type(),
        Expr::BinaryExpr { left, right, .. } => {
            let l = infer_expr_type_full(left, schema);
            let r = infer_expr_type_full(right, schema);

            match (l, r) {
                (ValueType::Matrix(r, c), _) => ValueType::Matrix(r, c),
                (_, ValueType::Matrix(r, c)) => ValueType::Matrix(r, c),
                (ValueType::Vector(d), _) => ValueType::Vector(d),
                (_, ValueType::Vector(d)) => ValueType::Vector(d),
                // Float64 always wins the promotion, even against a plain
                // Float, mirroring the runtime arithmetic rule.
                (ValueType::Float64, _) | (_, ValueType::Float64) => ValueType::Float64,
                (ValueType::Float, _) | (_, ValueType::Float) => ValueType::Float,
                (ValueType::Int, ValueType::Int) => ValueType::Int,
                _ => ValueType::Int,
            }
        }
        Expr::And(_, _)
        | Expr::Or(_, _)
        | Expr::Not(_)
        | Expr::IsNull(_)
        | Expr::IsNotNull(_)
        | Expr::In { .. }
        | Expr::Between { .. } => ValueType::Bool,
        Expr::AggregateExpr {
            func, expr: inner, ..
        } => match func {
            AggregateFunction::Avg => match infer_expr_type_full(inner, schema) {
                t @ (ValueType::Vector(_) | ValueType::Matrix(_, _)) => t,
                ValueType::Float64 => ValueType::Float64,
                ValueType::Complex => ValueType::Complex,
                _ => ValueType::Float,
            },
            AggregateFunction::Count => ValueType::Int,
            AggregateFunction::AvgVec | AggregateFunction::SumVec => {
                match infer_expr_type_full(inner, schema) {
                    t @ (ValueType::Vector(_) | ValueType::Matrix(_, _)) => t,
                    _ => ValueType::Vector(0),
                }
            }
            AggregateFunction::Variance | AggregateFunction::Median | AggregateFunction::Rrf(_) => {
                ValueType::Float64
            }
            AggregateFunction::ArgMax(_) | AggregateFunction::ArgMin(_) => {
                infer_expr_type_full(inner, schema)
            }
            _ => ValueType::Int,
        },
        Expr::VecLiteral(v) => ValueType::Vector(v.len()),
        Expr::MatLiteral(rows) => {
            let r = rows.len();
            let c = rows.first().map_or(0, |row| row.len());
            ValueType::Matrix(r, c)
        }
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
        },
        Expr::Case {
            else_expr,
            branches,
            ..
        } => {
            if let Some(branch) = branches.first() {
                infer_expr_type_full(&branch.1, schema)
            } else if let Some(e) = else_expr {
                infer_expr_type_full(e, schema)
            } else {
                ValueType::Null
            }
        }
        Expr::Coalesce(args) => args
            .first()
            .map(|e| infer_expr_type_full(e, schema))
            .unwrap_or(ValueType::Null),
        Expr::Nullif(a, _) => infer_expr_type_full(a, schema),
        Expr::ScalarFn { func, .. } => match func {
            ScalarFnKind::Length => ValueType::Int,
            _ => ValueType::String,
        },
        Expr::Cast { to, .. } => match to {
            CastTarget::Int => ValueType::Int,
            CastTarget::Float => ValueType::Float,
            CastTarget::Double => ValueType::Float64,
            CastTarget::Text | CastTarget::Bool => ValueType::String,
            CastTarget::Vector(n) => ValueType::Vector(*n),
            CastTarget::Matrix(r, c) => ValueType::Matrix(*r, *c),
            CastTarget::BitVector(n) => ValueType::BitVector(n.unwrap_or(0)),
        },
    }
}
