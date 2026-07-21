/// A fully-parsed LINALDB DSL statement.
///
/// Every variant maps to exactly one top-level command. The types are
/// deliberately decoupled from engine internals — the executor is responsible
/// for the mapping (e.g. `TensorKindAst` → `engine::TensorKind`).
#[derive(Debug, Clone)]
pub enum Statement {
    // ─── Tensor construction ──────────────────────────────────────────────────
    /// `DEFINE <name> AS [STRICT] TENSOR [dims] VALUES [values]`
    DefineTensor(DefineTensorStmt),
    /// `VECTOR <name> = [values]`
    Vector(VectorStmt),
    /// `MATRIX <name> = [[row], [row], ...]`
    Matrix(MatrixStmt),

    // ─── Assignment / computation ────────────────────────────────────────────
    /// `LET <name> = <expr>` or `LAZY LET <name> = <expr>`
    Let(LetStmt),
    /// `DERIVE <name> FROM <expr>`
    Derive(DeriveStmt),

    // ─── Zero-copy semantics ─────────────────────────────────────────────────
    /// `BIND <alias> TO <source>`
    Bind(BindStmt),
    /// `ATTACH <tensor> TO <dataset>.<column>`
    Attach(AttachStmt),

    // ─── Dataset operations ──────────────────────────────────────────────────
    /// `DATASET <name> COLUMNS (...)` or `DATASET <name> FROM <path>`
    CreateDataset(CreateDatasetStmt),
    /// `ALTER DATASET <name> ADD COLUMN ...`
    AlterDataset(AlterDatasetStmt),
    /// `INSERT INTO <dataset> (...)`
    InsertInto(InsertIntoStmt),
    /// `SELECT <cols> FROM <dataset> [WHERE ...] [LIMIT ...]`
    Select(SelectStmt),
    /// `MATERIALIZE <name>`
    Materialize(MaterializeStmt),
    /// `DELIVER <dataset> [TO <path>]`
    Deliver(DeliverStmt),

    // ─── Introspection ───────────────────────────────────────────────────────
    /// `SHOW <target>`
    Show(ShowStmt),
    /// `EXPLAIN <name>`
    Explain(ExplainStmt),
    /// `AUDIT <target>`
    Audit(AuditStmt),

    // ─── Persistence ─────────────────────────────────────────────────────────
    /// `SAVE TENSOR|DATASET|PIPELINE <name> [TO <path>]` — the kind keyword is required, not optional
    Save(SaveStmt),
    /// `LOAD TENSOR|DATASET|PIPELINE <name> [FROM <path>]` — the kind keyword is required, not optional
    Load(LoadStmt),
    /// `LIST TENSORS` / `LIST DATASETS` / etc.
    List(ListStmt),
    /// `IMPORT DATASET FROM <path>` (durable) or `USE DATASET FROM <path>` (ephemeral)
    Import(ImportStmt),
    /// `IMPORT CSV FROM <path> [AS <name>]`
    ImportCsv(ImportCsvStmt),
    /// `EXPORT [CSV] <name> TO <path>`
    Export(ExportStmt),

    // ─── Database management ─────────────────────────────────────────────────
    /// `CREATE DATABASE <name>`
    CreateDatabase(CreateDatabaseStmt),
    /// `DROP DATABASE <name>`
    DropDatabase(DropDatabaseStmt),
    /// `USE <name>`
    UseDatabase(UseDatabaseStmt),

    // ─── Index management ────────────────────────────────────────────────────
    /// `CREATE INDEX ON <dataset>(<column>)`
    CreateIndex(CreateIndexStmt),

    // ─── Metadata ────────────────────────────────────────────────────────────
    /// `SET DATASET <name> <key> = <value>`
    SetMetadata(SetMetadataStmt),

    // ─── Search ──────────────────────────────────────────────────────────────
    /// `SEARCH <tensor> [IN <dataset>] [TOP <k>]`
    Search(SearchStmt),

    // ─── Data mutation ───────────────────────────────────────────────────────
    /// `UPDATE <dataset> SET col = expr [, ...] [WHERE ...]`
    Update(UpdateStmt),
    /// `DELETE FROM <dataset> [WHERE ...]`
    Delete(DeleteStmt),

    // ─── Pipeline ────────────────────────────────────────────────────────────
    /// `TRANSFORM <source> SELECT ... [WHERE ...] [INTO <target>]`
    Transform(TransformStmt),
    /// `DEFINE PIPELINE <name> AS step [THEN step ...]`
    DefinePipeline(DefinePipelineStmt),
    /// `APPLY PIPELINE <name> ON <source> [INTO <target>]`
    ApplyPipeline(ApplyPipelineStmt),
    /// `DROP PIPELINE <name>`
    DropPipeline(String),
    /// `DESCRIBE PIPELINE <name>`
    DescribePipeline(String),

    // ─── Session ─────────────────────────────────────────────────────────────
    /// `RESET`
    Reset,
}

impl Statement {
    /// Returns `true` for statements that only read state and never mutate it.
    /// Used to gate shared-reference execution paths.
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            Statement::Explain(_)
                | Statement::Audit(_)
                | Statement::List(_)
                | Statement::Deliver(_)
        )
    }
}

// ─── Statement structs ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DefineTensorStmt {
    pub name: String,
    pub kind: TensorKindAst,
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VectorStmt {
    pub name: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MatrixStmt {
    pub name: String,
    /// Row-major flat data; shape is `rows × cols`.
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LetStmt {
    pub name: String,
    pub lazy: bool,
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub struct DeriveStmt {
    pub name: String,
    pub source_expr: Expr,
}

#[derive(Debug, Clone)]
pub struct BindStmt {
    pub alias: String,
    pub target: String,
}

#[derive(Debug, Clone)]
pub struct AttachStmt {
    pub tensor: String,
    pub dataset: String,
    pub column: String,
}

#[derive(Debug, Clone)]
pub struct CreateDatasetStmt {
    pub name: String,
    pub columns: Vec<ColumnDef>,
    /// `DATASET <name> FROM <source> [clauses]` variant.
    pub from: Option<DatasetFromClause>,
}

/// Clause data for `DATASET target FROM source [FILTER …] [SELECT …] [GROUP BY …] [ORDER BY …] [LIMIT …]`.
#[derive(Debug, Clone)]
pub struct DatasetFromClause {
    pub source: String,
    pub filter: Option<Expr>,
    pub select: Option<Vec<SelectExpr>>,
    pub group_by: Vec<String>,
    pub having: Option<Expr>,
    pub order_by: Option<OrderByClause>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Scalar value on the right-hand side of a DEFAULT clause.
#[derive(Debug, Clone)]
pub enum FilterValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct AlterDatasetStmt {
    pub dataset: String,
    pub operation: AlterOp,
}

#[derive(Debug, Clone)]
pub enum AlterOp {
    AddColumn(ColumnDef),
    AddComputedColumn {
        name: String,
        expr: Box<Expr>,
        lazy: bool,
    },
}

#[derive(Debug, Clone)]
pub struct InsertIntoStmt {
    pub dataset: String,
    pub row: InsertRow,
}

/// Named (`INSERT INTO ds (col=val, ...)`) or positional (`INSERT INTO ds VALUES (v1, v2, ...)`) form.
#[derive(Debug, Clone)]
pub enum InsertRow {
    Named(Vec<(String, InsertValue)>),
    Positional(Vec<InsertValue>),
}

#[derive(Debug, Clone)]
pub enum InsertValue {
    Scalar(f64),
    Text(String),
    TensorRef(String),
    Null,
    Bool(bool),
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
}

/// The source of a SELECT — either a named dataset or a subquery.
#[derive(Debug, Clone)]
pub enum DatasetSource {
    Named(String),
    Subquery {
        query: Box<SelectStmt>,
        alias: String,
    },
}

#[derive(Debug, Clone)]
pub enum SetOpKind {
    Union,
    UnionAll,
}

/// `PARTITION BY col, ... ORDER BY col ASC/DESC, ...` clause for window functions.
#[derive(Debug, Clone)]
pub struct WindowSpec {
    pub partition_by: Vec<String>,
    pub order_by: Vec<(String, bool)>,
}

/// Window function variants.
#[derive(Debug, Clone)]
pub enum WindowFunc {
    RowNumber,
    Rank,
    DenseRank,
    Lag { col: String, offset: usize },
    Lead { col: String, offset: usize },
    Sum(Box<Expr>),
    Avg(Box<Expr>),
    Count(Box<Expr>),
    Min(Box<Expr>),
    Max(Box<Expr>),
}

#[derive(Debug, Clone)]
pub struct SelectStmt {
    pub ctes: Vec<(String, SelectStmt)>,
    pub distinct: bool,
    pub source: DatasetSource,
    pub joins: Vec<JoinClause>,
    pub columns: SelectColumns,
    pub filter: Option<Expr>,
    pub group_by: Vec<String>,
    pub having: Option<Expr>,
    pub order_by: Option<OrderByClause>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub union: Option<(SetOpKind, Box<SelectStmt>)>,
}

#[derive(Debug, Clone)]
pub struct JoinClause {
    pub kind: JoinKind,
    pub dataset: String,
    /// For an equi-join: `left_col = right_col`. For a similarity join
    /// (`similarity_threshold` is `Some`): the two Vector columns compared
    /// via `COSINE_SIM(left_col, right_col) > threshold`.
    pub left_col: String,
    pub right_col: String,
    /// `Some(threshold)` for `ON COSINE_SIM(a.col, b.col) > threshold`;
    /// `None` for a plain equi-join.
    pub similarity_threshold: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
    Full,
}

#[derive(Debug, Clone)]
pub enum SelectColumns {
    All,
    Named(Vec<SelectExpr>),
}

#[derive(Debug, Clone)]
pub struct OrderByClause {
    pub columns: Vec<(String, bool)>,
}

#[derive(Debug, Clone)]
pub struct MaterializeStmt {
    pub target: String,
}

#[derive(Debug, Clone)]
pub struct DeliverStmt {
    pub dataset: String,
    pub path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ShowStmt {
    pub target: ShowTarget,
}

/// A single item in a SELECT column list.
#[derive(Debug, Clone)]
pub enum SelectExpr {
    Column(String),
    Aggregate {
        func: AggFuncAst,
        expr: Box<Expr>,
        /// Optional `AS alias` — honored by the resulting output column name.
        alias: Option<String>,
    },
    /// Window function: `fn() OVER (PARTITION BY … ORDER BY …) AS alias`
    Window {
        func: WindowFunc,
        spec: WindowSpec,
        alias: String,
    },
    /// Arbitrary computed expression with optional alias: `UPPER(name) AS uname`
    Computed {
        expr: Box<Expr>,
        alias: Option<String>,
    },
}

/// Aggregate functions recognised in SELECT columns.
#[derive(Debug, Clone, PartialEq)]
pub enum AggFuncAst {
    Sum,
    Avg,
    Count,
    Min,
    Max,
    /// Element-wise vector average: `AVG_VEC(embedding)`
    AvgVec,
    /// Element-wise vector sum: `SUM_VEC(embedding)`
    SumVec,
}

/// What `SHOW` should display.
#[derive(Debug, Clone)]
pub enum ShowTarget {
    /// `SHOW ALL` / `SHOW ALL TENSORS`
    All,
    /// `SHOW ALL DATASETS`
    AllDatasets,
    /// `SHOW DATABASES` / `SHOW ALL DATABASES`
    AllDatabases,
    /// `SHOW SCHEMA <name>`
    Schema(String),
    /// `SHOW SHAPE <name>`
    Shape(String),
    /// `SHOW LINEAGE <name>`
    Lineage(String),
    /// `SHOW INDEXES` / `SHOW INDEXES <dataset>`
    Indexes(Option<String>),
    /// `SHOW DATASET METADATA <name>`
    DatasetMetadata(String),
    /// `SHOW DATASET VERSIONS <name>`
    DatasetVersions(String),
    /// `SHOW "<literal string>"`
    StringLiteral(String),
    /// `SHOW <name>` — tensor or dataset by name
    Named(String),
    /// `SHOW PIPELINES`
    Pipelines,
}

/// What `EXPLAIN` should show a query plan for.
#[derive(Debug, Clone)]
pub enum ExplainTarget {
    /// `EXPLAIN [PLAN] DATASET <name>` — simple full-scan plan
    Dataset(String),
    /// `EXPLAIN [PLAN] DATASET <name> FROM <source> [FILTER …] [SELECT …] …`
    DatasetQuery {
        name: String,
        from: DatasetFromClause,
    },
    /// `EXPLAIN [PLAN] SEARCH <ds> ON <col> QUERY <q> LIMIT <k>`
    Search(SearchStmt),
    /// `EXPLAIN [PLAN] SELECT …`
    Select(SelectStmt),
}

#[derive(Debug, Clone)]
pub struct ExplainStmt {
    pub target: ExplainTarget,
}

#[derive(Debug, Clone)]
pub struct AuditStmt {
    pub target: String,
}

#[derive(Debug, Clone)]
pub struct SaveStmt {
    pub kind: PersistKind,
    pub name: String,
    pub path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LoadStmt {
    pub kind: PersistKind,
    pub name: String,
    pub path: Option<String>,
}

#[derive(Debug, Clone)]
pub enum PersistKind {
    Tensor,
    Dataset,
    Pipeline,
}

#[derive(Debug, Clone)]
pub struct ListStmt {
    pub target: ListTarget,
}

#[derive(Debug, Clone)]
pub enum ListTarget {
    Tensors,
    Datasets,
    DatasetVersions(String),
    DatasetPackages,
    /// `LIST PIPELINES` — alias for `SHOW PIPELINES`, added for naming
    /// symmetry with `LIST TENSORS`/`LIST DATASETS`.
    Pipelines,
}

#[derive(Debug, Clone)]
pub struct ImportStmt {
    /// `false` = `IMPORT DATASET FROM <path>` (persisted)
    /// `true`  = `USE DATASET FROM <path>` (session-only)
    pub ephemeral: bool,
    pub path: String,
    pub name: Option<String>,
    /// Optional `FIELDS (name1, name2, ...)` clause: explicitly pick which
    /// named fields/arrays to ingest from a source that contains several
    /// (e.g. an HDF5 file bundling arrays of different shapes). `None`
    /// keeps the connector's default behavior (first-length-group wins,
    /// everything else is skipped with a warning); `Some` makes the
    /// selection explicit and turns "requested fields don't share a
    /// combinable shape" into a hard error instead of a silent/warned skip,
    /// since the user has no fallback expectation once they've named exactly
    /// what they want.
    pub fields: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ExportStmt {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct ImportCsvStmt {
    pub path: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CreateDatabaseStmt {
    pub name: String,
    pub if_not_exists: bool,
}

#[derive(Debug, Clone)]
pub struct DropDatabaseStmt {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone)]
pub struct UseDatabaseStmt {
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct CreateIndexStmt {
    pub dataset: String,
    pub column: String,
    pub kind: IndexKindAst,
}

#[derive(Debug, Clone)]
pub enum IndexKindAst {
    Default,
    BTree,
    Hash,
    Vector,
}

#[derive(Debug, Clone)]
pub struct SetMetadataStmt {
    pub dataset: String,
    pub key: String,
    pub value: String,
}

/// How the query vector is supplied in a SEARCH statement.
#[derive(Debug, Clone)]
pub enum SearchQuery {
    /// `QUERY <tensor_name>` — reference to an in-memory named tensor.
    TensorRef(String),
    /// `QUERY [v1, v2, …]` — inline vector literal.
    Inline(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct SearchStmt {
    /// Dataset to search in.
    pub dataset: String,
    /// Vector column to search against.
    pub column: String,
    /// Query vector: either a named tensor or an inline literal.
    pub query: SearchQuery,
    /// Number of nearest neighbours to return.
    pub top_k: usize,
    /// Optional output dataset name (defaults to `"search_results"`).
    pub target: Option<String>,
}

// ─── Shared sub-types ─────────────────────────────────────────────────────────

/// A column definition in `DATASET ... COLUMNS (...)` or `ADD COLUMN`.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub col_type: ColType,
    pub nullable: bool,
    /// Explicit `DEFAULT <val>` from the DDL statement; `None` means use type-appropriate zero.
    pub default_val: Option<FilterValue>,
}

/// Column types as parsed from the DSL.
#[derive(Debug, Clone)]
pub enum ColType {
    Int,
    Float,
    String,
    Bool,
    /// `Vector(n)` — 1-D tensor of length n
    Vector(usize),
    /// `Matrix(rows, cols)` — 2-D tensor
    Matrix(usize, usize),
    /// `Tensor(dims)` — N-D tensor
    Tensor(Vec<usize>),
}

/// Tensor kind as expressed in the DSL. Decoupled from `engine::TensorKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKindAst {
    Normal,
    Strict,
    Lazy,
}

// ─── Expression types ─────────────────────────────────────────────────────────

/// An expression — the right-hand side of `LET`, `DERIVE`, and other
/// value-producing statements.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A named tensor or variable: `a`, `my_tensor`
    Ref(String),
    /// An integer literal: `42`, `-7`
    Int(i64),
    /// A float literal: `3.14`, `1.0e-5`
    Scalar(f64),
    /// A string literal: `"hello"`
    StringLit(String),
    /// A boolean literal: `true`, `false`
    Bool(bool),
    /// Infix arithmetic: `a + b`, `x * 2.0`
    Infix {
        op: InfixOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Logical AND: `a > 5 AND b < 10`
    And(Box<Expr>, Box<Expr>),
    /// Logical OR: `a = 1 OR b = 2`
    Or(Box<Expr>, Box<Expr>),
    /// Logical NOT: `NOT active`
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
    /// Prefix named operation: `ADD a b`, `CORRELATE a WITH b`
    Call(CallExpr),
    /// Subscript: `t[0, 1]`, `t[0:5, *]`
    Index {
        base: Box<Expr>,
        indices: Vec<IndexSpec>,
    },
    /// Field/column access: `dataset.column`
    Field { base: Box<Expr>, field: String },
    /// Dataset constructor used in `LET ds = dataset("name")`
    DatasetRef(String),
    /// `CASE [expr] WHEN cond THEN result … [ELSE default] END`
    Case {
        operand: Option<Box<Expr>>,
        branches: Vec<(Expr, Expr)>,
        else_expr: Option<Box<Expr>>,
    },
    /// `COALESCE(e1, e2, …)` — first non-NULL
    Coalesce(Vec<Expr>),
    /// `NULLIF(a, b)` — NULL if a == b, else a
    Nullif(Box<Expr>, Box<Expr>),
    /// Scalar string/type functions: `UPPER(s)`, `CAST(x AS INT)`, etc.
    ScalarFn { func: ScalarFnKind, args: Vec<Expr> },
    /// `CAST(expr AS type)`
    Cast { expr: Box<Expr>, to: CastTarget },
    /// Inline vector literal: `[0.1, 0.2, 0.3]`
    VecLiteral(Vec<f64>),
    /// SQL-style vector function: `COSINE_SIM(embedding, [0.1, 0.2, 0.3])`
    VectorFn { func: VectorFnKind, args: Vec<Expr> },
    /// Inline matrix literal: `[[0.1, 0.2], [0.3, 0.4]]`
    MatLiteral(Vec<Vec<f64>>),
}

/// Vector/tensor functions usable directly in SQL expressions.
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
    /// `FLATTEN(expr)` — flatten a Matrix (or a no-op on an already-flat
    /// Vector) row-major into a 1D Vector.
    Flatten,
    /// `DISTANCE(a, b)` — Euclidean distance, SQL-callable form of the
    /// standalone `DISTANCE a TO b` tensor-DSL keyword (§3).
    Distance,
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
    Text,
    Bool,
    /// `CAST(expr AS VECTOR(n))` — reshape/flatten to a Vector of length `n`.
    Vector(usize),
    /// `CAST(expr AS MATRIX(r, c))` — reshape/flatten to a Matrix of shape `r x c`.
    Matrix(usize, usize),
}

/// Infix arithmetic operators (symbols: `+`, `-`, `*`, `/`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfixOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Eq,
    NotEq,
    Gt,
    Lt,
    GtEq,
    LtEq,
}

/// Named operation calls used in `LET` and `DERIVE` expressions.
/// Each variant encodes the operation's argument structure.
#[derive(Debug, Clone)]
pub enum CallExpr {
    // ── Two-operand operations ──────────────────────────────────────────────
    /// `ADD a b`
    Add(Box<Expr>, Box<Expr>),
    /// `SUBTRACT a b`
    Subtract(Box<Expr>, Box<Expr>),
    /// `MULTIPLY a b`
    Multiply(Box<Expr>, Box<Expr>),
    /// `DIVIDE a b`
    Divide(Box<Expr>, Box<Expr>),
    /// `CORRELATE a WITH b`
    Correlate(Box<Expr>, Box<Expr>),
    /// `SIMILARITY a WITH b`
    Similarity(Box<Expr>, Box<Expr>),
    /// `DISTANCE a TO b`
    Distance(Box<Expr>, Box<Expr>),
    /// `MATMUL a b`
    Matmul(Box<Expr>, Box<Expr>),

    // ── Single-operand operations ───────────────────────────────────────────
    /// `NORMALIZE a`
    Normalize(Box<Expr>),
    /// `TRANSPOSE a`
    Transpose(Box<Expr>),
    /// `FLATTEN a`
    Flatten(Box<Expr>),
    /// `SUM a`
    Sum(Box<Expr>),
    /// `MEAN a`
    Mean(Box<Expr>),
    /// `STDEV a`
    Stdev(Box<Expr>),
    /// `FFT a` — real-to-complex forward FFT. `a` must be a rank-1 Vector;
    /// result is a `Matrix(2, N/2+1)` (row 0 = real parts, row 1 =
    /// imaginary parts). See SIGNAL_PROCESSING_PLAN.md for the convention.
    Fft(Box<Expr>),
    /// `IFFT a` — complex-to-real inverse FFT. `a` must be a `Matrix(2, M)`
    /// spectrum (as `FFT` produces); result is a real `Vector`. Assumes the
    /// original signal length was even (`2*(M-1)`) -- the spectrum alone
    /// can't distinguish an even- from an odd-length source signal (both
    /// produce the same `M`), and there is no side-channel carrying the
    /// true length through the DSL layer today.
    Ifft(Box<Expr>),
    /// `MAGNITUDE a` — power/magnitude spectrum: `sqrt(re^2 + im^2)` per
    /// bin. `a` must be a `Matrix(2, M)` spectrum (as `FFT` produces);
    /// result is a real `Vector(M)`. The convenience most whitening/PSD
    /// work needs without touching phase.
    Magnitude(Box<Expr>),
    /// `PSD a WINDOW <n>` — power spectral density estimate via averaged
    /// periodograms (simplified: non-overlapping chunks, no window
    /// function -- see `core::signal::psd`'s doc comment). `a` must be a
    /// rank-1 Vector; result is a real `Vector(n/2+1)`.
    Psd { input: Box<Expr>, window: usize },
    /// `WHITEN a WITH b` — flattens `a`'s noise spectrum against a PSD
    /// estimate `b` (as `PSD` produces). `b` must have exactly
    /// `a.len()/2+1` entries (see `core::signal::whiten`'s doc comment for
    /// why -- interpolating a differently-sized PSD onto `a` is not
    /// implemented). Result is a real `Vector` the same length as `a`.
    Whiten { signal: Box<Expr>, psd: Box<Expr> },
    /// `BANDPASS a FROM low_hz TO high_hz WITH RATE sample_rate` —
    /// brick-wall zeroing of every FFT bin outside `[low_hz, high_hz]`,
    /// then inverse-transformed back to the time domain. `a` must be a
    /// rank-1 Vector; result is a real `Vector` the same length as `a`.
    Bandpass {
        input: Box<Expr>,
        low_hz: f64,
        high_hz: f64,
        sample_rate: f64,
    },
    /// `SCALE a BY <factor>`
    Scale { input: Box<Expr>, factor: f64 },
    /// `RESHAPE a TO [dims]`
    Reshape { input: Box<Expr>, shape: Vec<usize> },

    // ── N-ary operations ────────────────────────────────────────────────────
    /// `STACK t1 t2 t3 ...`
    Stack(Vec<Expr>),
}

#[derive(Debug, Clone)]
pub struct TransformStmt {
    pub source: String,
    pub columns: SelectColumns,
    pub filter: Option<Expr>,
    /// If Some, write result to this dataset; if None, replace source in-place.
    pub target: Option<String>,
}

#[derive(Debug, Clone)]
pub struct UpdateStmt {
    pub dataset: String,
    /// `(column_name, new_value_expr)` pairs
    pub assignments: Vec<(String, Expr)>,
    pub filter: Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct DeleteStmt {
    pub dataset: String,
    pub filter: Option<Expr>,
}

/// A single step in a named pipeline.
#[derive(Debug, Clone)]
pub enum PipelineStep {
    /// `SELECT expr AS alias, ...`
    Select(Vec<SelectExpr>),
    /// `WHERE condition` or `FILTER condition`
    Filter(Expr),
    /// `ORDER BY col ASC/DESC, ...`
    OrderBy(Vec<(String, bool)>),
    /// `LIMIT n`
    Limit(usize),
    /// `NORMALIZE col_name` — normalize a vector column in-place
    NormalizeCol(String),
}

#[derive(Debug, Clone)]
pub struct DefinePipelineStmt {
    pub name: String,
    pub steps: Vec<PipelineStep>,
    /// Original DSL source text — populated by the executor entry-point for persistence.
    pub source: String,
}

/// A pipeline stored in the session registry.
#[derive(Debug, Clone)]
pub struct StoredPipeline {
    pub steps: Vec<PipelineStep>,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct ApplyPipelineStmt {
    pub pipeline: String,
    pub source: String,
    pub into: Option<String>,
}

/// A single dimension specifier inside a subscript expression `t[...]`.
#[derive(Debug, Clone)]
pub enum IndexSpec {
    /// `*` or `:` — select the entire dimension
    All,
    /// A single index: `0`, `3`
    Index(usize),
    /// A half-open range: `0:5`
    Range(usize, usize),
}
