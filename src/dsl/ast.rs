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
    /// `SAVE [TENSOR|DATASET] <name> [TO <path>]`
    Save(SaveStmt),
    /// `LOAD [TENSOR|DATASET] <name> [FROM <path>]`
    Load(LoadStmt),
    /// `LIST TENSORS` / `LIST DATASETS` / etc.
    List(ListStmt),
    /// `IMPORT DATASET FROM <path>` (durable) or `USE DATASET FROM <path>` (ephemeral)
    Import(ImportStmt),
    /// `EXPORT <name> TO <path>`
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
            Statement::Show(_) | Statement::Explain(_) | Statement::Audit(_) | Statement::List(_)
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
    pub filter: Option<DatasetFilter>,
    pub select: Option<Vec<SelectExpr>>,
    pub group_by: Vec<String>,
    pub having: Option<DatasetFilter>,
    pub order_by: Option<OrderByClause>,
    pub limit: Option<usize>,
}

/// A simple `col op value` predicate used in FILTER / HAVING clauses of `DATASET FROM`.
#[derive(Debug, Clone)]
pub struct DatasetFilter {
    pub column: String,
    pub op: CmpOp,
    pub value: FilterValue,
}

/// Comparison operators supported in FILTER / HAVING predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    NotEq,
    Gt,
    GtEq,
    Lt,
    LtEq,
}

/// Scalar value on the right-hand side of a FILTER predicate or DEFAULT clause.
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

#[derive(Debug, Clone)]
pub struct SelectStmt {
    pub dataset: String,
    pub columns: SelectColumns,
    pub filter: Option<Expr>,
    pub group_by: Vec<String>,
    pub having: Option<Expr>,
    pub order_by: Option<OrderByClause>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum SelectColumns {
    All,
    Named(Vec<SelectExpr>),
}

#[derive(Debug, Clone)]
pub struct OrderByClause {
    pub column: String,
    pub ascending: bool,
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

/// A single item in a SELECT column list — either a plain column or an aggregate call.
#[derive(Debug, Clone)]
pub enum SelectExpr {
    Column(String),
    Aggregate { func: AggFuncAst, expr: Box<Expr> },
}

/// Aggregate functions recognised in SELECT columns.
#[derive(Debug, Clone, PartialEq)]
pub enum AggFuncAst {
    Sum,
    Avg,
    Count,
    Min,
    Max,
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
}

/// What `EXPLAIN` should show a query plan for.
#[derive(Debug, Clone)]
pub enum ExplainTarget {
    /// `EXPLAIN [PLAN] DATASET <name>` — simple dataset scan
    Dataset(String),
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
}

#[derive(Debug, Clone)]
pub struct ImportStmt {
    /// `false` = `IMPORT DATASET FROM <path>` (persisted)
    /// `true`  = `USE DATASET FROM <path>` (session-only)
    pub ephemeral: bool,
    pub path: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExportStmt {
    pub name: String,
    pub path: String,
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
    /// Infix arithmetic: `a + b`, `x * 2.0`
    Infix {
        op: InfixOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
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
    /// `SCALE a BY <factor>`
    Scale { input: Box<Expr>, factor: f64 },
    /// `RESHAPE a TO [dims]`
    Reshape { input: Box<Expr>, shape: Vec<usize> },

    // ── N-ary operations ────────────────────────────────────────────────────
    /// `STACK t1 t2 t3 ...`
    Stack(Vec<Expr>),
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
