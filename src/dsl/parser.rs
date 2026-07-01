use logos::Span;

use crate::dsl::ast::*;
use crate::dsl::lexer::{tokenize, Token};

// ─── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ParseError {
    /// Byte offset in the source string where the error was detected.
    pub offset: usize,
    pub msg: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (at byte {})", self.msg, self.offset)
    }
}

impl ParseError {
    /// Convert to a `DslError::Parse` using the given line number.
    pub fn into_dsl_error(self, line: usize) -> crate::dsl::DslError {
        crate::dsl::DslError::Parse { line, msg: self.msg }
    }
}

// ─── Parser ───────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<(Token, Span)>,
    pos: usize,
    end: usize,
}

impl Parser {
    fn new(tokens: Vec<(Token, Span)>, source_len: usize) -> Self {
        Self { tokens, pos: 0, end: source_len }
    }

    // ── Cursor primitives ────────────────────────────────────────────────────

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|(t, _)| t)
    }

    fn current_span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|(_, s)| s.clone())
            .unwrap_or(self.end..self.end)
    }

    fn advance(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let (tok, _) = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(tok)
        } else {
            None
        }
    }

    /// Returns `true` if the current token has the same variant as `expected`
    /// (ignoring inner values).
    fn at(&self, expected: &Token) -> bool {
        self.peek()
            .map(|t| same_variant(t, expected))
            .unwrap_or(false)
    }

    /// Returns `true` if the current token is `Token::Ident(name)`.
    fn at_ident(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Token::Ident(s)) if s == name)
    }

    /// Returns `true` if the current token is any `Token::Ident(_)`.
    fn at_any_ident(&self) -> bool {
        matches!(self.peek(), Some(Token::Ident(_)))
    }

    fn eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    // ── Consuming helpers ────────────────────────────────────────────────────

    /// Consume the current token if its variant matches `expected`.
    fn eat(&mut self, expected: &Token) -> Result<Span, ParseError> {
        if self.at(expected) {
            let span = self.current_span();
            self.pos += 1;
            Ok(span)
        } else {
            Err(self.unexpected(&format!("{}", expected)))
        }
    }

    fn eat_ident(&mut self) -> Result<String, ParseError> {
        match self.advance_if_ident() {
            Some(s) => Ok(s),
            None => Err(self.unexpected("identifier")),
        }
    }

    fn advance_if_ident(&mut self) -> Option<String> {
        if matches!(self.peek(), Some(Token::Ident(_))) {
            if let Some(Token::Ident(s)) = self.advance() {
                return Some(s);
            }
        }
        None
    }

    fn eat_str(&mut self) -> Result<String, ParseError> {
        if matches!(self.peek(), Some(Token::Str(_))) {
            if let Some(Token::Str(s)) = self.advance() {
                return Ok(s);
            }
        }
        Err(self.unexpected("string literal"))
    }

    fn eat_int(&mut self) -> Result<i64, ParseError> {
        if matches!(self.peek(), Some(Token::Int(_))) {
            if let Some(Token::Int(n)) = self.advance() {
                return Ok(n);
            }
        }
        Err(self.unexpected("integer literal"))
    }

    /// Consume a numeric literal (int or float), returning `f64`.
    /// A leading `-` is consumed as a unary negation.
    fn eat_number(&mut self) -> Result<f64, ParseError> {
        match self.peek() {
            Some(Token::Float(_)) => {
                if let Some(Token::Float(f)) = self.advance() {
                    return Ok(f);
                }
                unreachable!()
            }
            Some(Token::Int(_)) => {
                if let Some(Token::Int(n)) = self.advance() {
                    return Ok(n as f64);
                }
                unreachable!()
            }
            Some(Token::Minus) => {
                self.advance();
                Ok(-self.eat_number()?)
            }
            _ => Err(self.unexpected("numeric literal")),
        }
    }

    fn eat_usize(&mut self) -> Result<usize, ParseError> {
        let n = self.eat_int()?;
        n.try_into().map_err(|_| self.error(format!("expected non-negative integer, got {}", n)))
    }

    // ── Error helpers ────────────────────────────────────────────────────────

    fn error(&self, msg: impl Into<String>) -> ParseError {
        ParseError { offset: self.current_span().start, msg: msg.into() }
    }

    fn unexpected(&self, expected: &str) -> ParseError {
        let found = match self.peek() {
            Some(t) => format!("{}", t),
            None => "end of input".to_string(),
        };
        self.error(format!("expected {}, found {}", expected, found))
    }
}

/// Discriminant-level equality: two tokens are "the same" if they are the same
/// variant, regardless of any inner value.
fn same_variant(a: &Token, b: &Token) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Parse a single DSL statement from `source`.
///
/// Returns `Err` if the source fails to lex or is not a valid statement.
/// Trailing tokens after a complete statement are treated as an error.
pub fn parse(source: &str) -> Result<Statement, ParseError> {
    let tokens = tokenize(source).map_err(|offset| ParseError {
        offset,
        msg: format!("unrecognised character at byte {}", offset),
    })?;

    let mut p = Parser::new(tokens, source.len());

    if p.eof() {
        return Err(ParseError { offset: 0, msg: "empty input".to_string() });
    }

    let stmt = p.parse_statement()?;

    if !p.eof() {
        return Err(p.error(format!(
            "unexpected token after statement: {}",
            p.peek().map(|t| format!("{}", t)).unwrap_or_default()
        )));
    }

    Ok(stmt)
}

// ─── Statement dispatch ───────────────────────────────────────────────────────

impl Parser {
    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        match self.peek() {
            Some(Token::Define)      => self.parse_define_tensor(),
            Some(Token::Vector)      => self.parse_vector(),
            Some(Token::Matrix)      => self.parse_matrix(),
            Some(Token::Let)
            | Some(Token::Lazy)      => self.parse_let(),
            Some(Token::Derive)      => self.parse_derive(),
            Some(Token::Show)        => self.parse_show(),
            Some(Token::Bind)        => self.parse_bind(),
            Some(Token::Attach)      => self.parse_attach(),
            Some(Token::Dataset)     => self.parse_create_dataset(),
            Some(Token::Insert)      => self.parse_insert_into(),
            Some(Token::Select)      => self.parse_select(),
            Some(Token::Materialize) => self.parse_materialize(),
            Some(Token::Deliver)     => self.parse_deliver(),
            Some(Token::Explain)     => self.parse_explain(),
            Some(Token::Audit)       => self.parse_audit(),
            Some(Token::Save)        => self.parse_save(),
            Some(Token::Load)        => self.parse_load(),
            Some(Token::List)        => self.parse_list(),
            Some(Token::Import)      => self.parse_import(),
            Some(Token::Export)      => self.parse_export(),
            Some(Token::Create)      => self.parse_create(),
            Some(Token::Alter)       => self.parse_alter(),
            Some(Token::Drop)        => self.parse_drop(),
            Some(Token::Use)         => self.parse_use(),
            Some(Token::Set)         => self.parse_set(),
            Some(Token::Search)      => self.parse_search(),
            Some(Token::Reset)       => { self.advance(); Ok(Statement::Reset) }
            _ => Err(self.unexpected("a statement keyword")),
        }
    }
}

// ─── Statement parsers ────────────────────────────────────────────────────────

impl Parser {
    // DEFINE <name> AS [STRICT] TENSOR [dims] VALUES [values]
    fn parse_define_tensor(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Define)?;
        let name = self.eat_ident()?;
        self.eat(&Token::As)?;

        let kind = if self.at(&Token::Strict) {
            self.advance();
            self.eat(&Token::Tensor)?;
            TensorKindAst::Strict
        } else {
            self.eat(&Token::Tensor)?;
            TensorKindAst::Normal
        };

        let shape = self.parse_usize_list()?;
        self.eat(&Token::Values)?;
        let values = self.parse_f64_list()?;

        Ok(Statement::DefineTensor(DefineTensorStmt { name, kind, shape, values }))
    }

    // VECTOR <name> = [values]
    fn parse_vector(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Vector)?;
        let name = self.eat_ident()?;
        self.eat(&Token::Eq)?;
        let values = self.parse_f64_list()?;
        Ok(Statement::Vector(VectorStmt { name, values }))
    }

    // MATRIX <name> = [[r0], [r1], ...]
    fn parse_matrix(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Matrix)?;
        let name = self.eat_ident()?;
        self.eat(&Token::Eq)?;

        self.eat(&Token::LBracket)?;
        let mut rows_data: Vec<Vec<f64>> = Vec::new();
        while !self.at(&Token::RBracket) && !self.eof() {
            rows_data.push(self.parse_f64_list()?);
            if self.at(&Token::Comma) { self.advance(); }
        }
        self.eat(&Token::RBracket)?;

        if rows_data.is_empty() {
            return Err(self.error("matrix must have at least one row"));
        }
        let cols = rows_data[0].len();
        for (i, row) in rows_data.iter().enumerate() {
            if row.len() != cols {
                return Err(self.error(format!(
                    "row {} has {} values, expected {}",
                    i, row.len(), cols
                )));
            }
        }
        let rows = rows_data.len();
        let values: Vec<f64> = rows_data.into_iter().flatten().collect();

        Ok(Statement::Matrix(MatrixStmt { name, rows, cols, values }))
    }

    // LET <name> = <expr>
    // LAZY LET <name> = <expr>
    // LET LAZY <name> = <expr>
    fn parse_let(&mut self) -> Result<Statement, ParseError> {
        let lazy = match self.peek() {
            Some(Token::Lazy) => {
                self.advance();
                self.eat(&Token::Let)?;
                true
            }
            Some(Token::Let) => {
                self.advance();
                let lazy = self.at(&Token::Lazy);
                if lazy { self.advance(); }
                lazy
            }
            _ => return Err(self.unexpected("LET or LAZY LET")),
        };

        let name = self.eat_ident()?;
        self.eat(&Token::Eq)?;
        let expr = self.parse_expr()?;

        Ok(Statement::Let(LetStmt { name, lazy, expr }))
    }

    // DERIVE <name> FROM <expr>
    fn parse_derive(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Derive)?;
        let name = self.eat_ident()?;
        self.eat(&Token::From)?;
        let source_expr = self.parse_expr()?;
        Ok(Statement::Derive(DeriveStmt { name, source_expr }))
    }

    // BIND <alias> TO <target>
    fn parse_bind(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Bind)?;
        let alias = self.eat_ident()?;
        self.eat(&Token::To)?;
        let target = self.eat_ident()?;
        Ok(Statement::Bind(BindStmt { alias, target }))
    }

    // ATTACH <tensor> TO <dataset>.<column>
    fn parse_attach(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Attach)?;
        let tensor = self.eat_ident()?;
        self.eat(&Token::To)?;
        let dataset = self.eat_ident()?;
        self.eat(&Token::Dot)?;
        let column = self.eat_ident()?;
        Ok(Statement::Attach(AttachStmt { tensor, dataset, column }))
    }

    // SHOW <target>
    fn parse_show(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Show)?;

        let target = match self.peek() {
            Some(Token::All) => {
                self.advance();
                match self.peek() {
                    Some(Token::Datasets)  => { self.advance(); ShowTarget::AllDatasets }
                    Some(Token::Databases) => { self.advance(); ShowTarget::AllDatabases }
                    Some(Token::Tensors)   => { self.advance(); ShowTarget::All }
                    _                      => ShowTarget::All,
                }
            }
            Some(Token::Databases) => { self.advance(); ShowTarget::AllDatabases }
            Some(Token::Schema)    => {
                self.advance();
                ShowTarget::Schema(self.eat_ident()?)
            }
            Some(Token::Shape)     => {
                self.advance();
                ShowTarget::Shape(self.eat_ident()?)
            }
            Some(Token::Lineage)   => {
                self.advance();
                ShowTarget::Lineage(self.eat_ident()?)
            }
            Some(Token::Indexes)   => {
                self.advance();
                let ds = if self.at_any_ident() { Some(self.eat_ident()?) } else { None };
                ShowTarget::Indexes(ds)
            }
            Some(Token::Dataset)   => {
                self.advance();
                match self.peek() {
                    Some(Token::Metadata) => {
                        self.advance();
                        ShowTarget::DatasetMetadata(self.eat_ident()?)
                    }
                    Some(Token::Versions) => {
                        self.advance();
                        ShowTarget::DatasetVersions(self.eat_ident()?)
                    }
                    _ => return Err(self.unexpected("METADATA or VERSIONS after SHOW DATASET")),
                }
            }
            Some(Token::Str(_))    => ShowTarget::StringLiteral(self.eat_str()?),
            Some(Token::Ident(_))  => ShowTarget::Named(self.eat_ident()?),
            _ => return Err(self.unexpected("a SHOW target")),
        };

        Ok(Statement::Show(ShowStmt { target }))
    }

    // DATASET <name> COLUMNS (...) | DATASET <name> FROM <src>
    fn parse_create_dataset(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Dataset)?;
        let name = self.eat_ident()?;

        match self.peek() {
            Some(Token::Columns) => {
                self.advance();
                let columns = self.parse_column_list()?;
                Ok(Statement::CreateDataset(CreateDatasetStmt { name, columns, from: None }))
            }
            Some(Token::From) => {
                self.advance();
                let src = self.eat_str()?;
                Ok(Statement::CreateDataset(CreateDatasetStmt {
                    name,
                    columns: vec![],
                    from: Some(src),
                }))
            }
            _ => Err(self.unexpected("COLUMNS or FROM after DATASET name")),
        }
    }

    // ALTER DATASET <name> ADD COLUMN <coldef>
    fn parse_alter(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Alter)?;
        self.eat(&Token::Dataset)?;
        let dataset = self.eat_ident()?;
        self.eat(&Token::Add)?;
        self.eat(&Token::Column)?;
        let col = self.parse_column_def()?;
        Ok(Statement::AlterDataset(AlterDatasetStmt {
            dataset,
            operation: AlterOp::AddColumn(col),
        }))
    }

    // INSERT INTO <dataset> (col = val, ...)
    fn parse_insert_into(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Insert)?;
        self.eat(&Token::Into)?;
        let dataset = self.eat_ident()?;

        self.eat(&Token::LParen)?;
        let mut row = Vec::new();
        while !self.at(&Token::RParen) && !self.eof() {
            let col = self.eat_ident()?;
            self.eat(&Token::Eq)?;
            let val = self.parse_insert_value()?;
            row.push((col, val));
            if self.at(&Token::Comma) { self.advance(); }
        }
        self.eat(&Token::RParen)?;

        Ok(Statement::InsertInto(InsertIntoStmt { dataset, row }))
    }

    fn parse_insert_value(&mut self) -> Result<InsertValue, ParseError> {
        match self.peek() {
            Some(Token::Null)    => { self.advance(); Ok(InsertValue::Null) }
            Some(Token::Str(_))  => Ok(InsertValue::Text(self.eat_str()?)),
            Some(Token::Float(_))
            | Some(Token::Int(_))
            | Some(Token::Minus) => Ok(InsertValue::Scalar(self.eat_number()?)),
            Some(Token::Ident(_))=> Ok(InsertValue::TensorRef(self.eat_ident()?)),
            _ => Err(self.unexpected("a value (number, string, identifier, or NULL)")),
        }
    }

    // SELECT [* | col, ...] FROM <dataset> [WHERE expr] [GROUP BY col, ...] [HAVING expr]
    //                                       [ORDER BY col [ASC|DESC]] [LIMIT n]
    fn parse_select(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Select)?;

        let columns = if self.at(&Token::Star) {
            self.advance();
            SelectColumns::All
        } else {
            let mut cols = vec![self.eat_ident()?];
            while self.at(&Token::Comma) {
                self.advance();
                cols.push(self.eat_ident()?);
            }
            SelectColumns::Named(cols)
        };

        self.eat(&Token::From)?;
        let dataset = self.eat_ident()?;

        let mut filter = None;
        let mut group_by = Vec::new();
        let mut having = None;
        let mut order_by = None;
        let mut limit = None;

        loop {
            match self.peek() {
                Some(Token::Where) | Some(Token::Filter) => {
                    self.advance();
                    filter = Some(self.parse_expr()?);
                }
                Some(Token::Group) => {
                    self.advance();
                    if self.at(&Token::By) { self.advance(); }
                    group_by.push(self.eat_ident()?);
                    while self.at(&Token::Comma) {
                        self.advance();
                        group_by.push(self.eat_ident()?);
                    }
                }
                Some(Token::Having) => {
                    self.advance();
                    having = Some(self.parse_expr()?);
                }
                Some(Token::Order) => {
                    self.advance();
                    if self.at(&Token::By) { self.advance(); }
                    let column = self.eat_ident()?;
                    // Optional ASC/DESC
                    let ascending = !self.at_ident("DESC");
                    if self.at_ident("ASC") || self.at_ident("DESC") { self.advance(); }
                    order_by = Some(OrderByClause { column, ascending });
                }
                Some(Token::Limit) => {
                    self.advance();
                    limit = Some(self.eat_usize()?);
                }
                _ => break,
            }
        }

        Ok(Statement::Select(SelectStmt {
            dataset,
            columns,
            filter,
            group_by,
            having,
            order_by,
            limit,
        }))
    }

    // MATERIALIZE <name> | MATERIALIZE <dataset>.<column>
    fn parse_materialize(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Materialize)?;
        let name = self.eat_ident()?;
        let target = if self.at(&Token::Dot) {
            self.advance();
            let col = self.eat_ident()?;
            format!("{}.{}", name, col)
        } else {
            name
        };
        Ok(Statement::Materialize(MaterializeStmt { target }))
    }

    // DELIVER <dataset> [TO <path>]
    fn parse_deliver(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Deliver)?;
        let dataset = self.eat_ident()?;
        let path = if self.at(&Token::To) {
            self.advance();
            Some(self.eat_str()?)
        } else {
            None
        };
        Ok(Statement::Deliver(DeliverStmt { dataset, path }))
    }

    // EXPLAIN <name>
    fn parse_explain(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Explain)?;
        let target = self.eat_ident()?;
        Ok(Statement::Explain(ExplainStmt { target }))
    }

    // AUDIT <name>
    fn parse_audit(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Audit)?;
        let target = self.eat_ident()?;
        Ok(Statement::Audit(AuditStmt { target }))
    }

    // SAVE TENSOR <name> [TO <path>]
    // SAVE DATASET <name> [TO <path>]
    fn parse_save(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Save)?;
        let kind = self.parse_persist_kind()?;
        let name = self.eat_ident()?;
        let path = if self.at(&Token::To) {
            self.advance();
            Some(self.eat_str()?)
        } else {
            None
        };
        Ok(Statement::Save(SaveStmt { kind, name, path }))
    }

    // LOAD TENSOR <name> [FROM <path>]
    // LOAD DATASET <name> [FROM <path>]
    fn parse_load(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Load)?;
        let kind = self.parse_persist_kind()?;
        let name = self.eat_ident()?;
        let path = if self.at(&Token::From) {
            self.advance();
            Some(self.eat_str()?)
        } else {
            None
        };
        Ok(Statement::Load(LoadStmt { kind, name, path }))
    }

    fn parse_persist_kind(&mut self) -> Result<PersistKind, ParseError> {
        match self.peek() {
            Some(Token::Tensor)  => { self.advance(); Ok(PersistKind::Tensor) }
            Some(Token::Dataset) => { self.advance(); Ok(PersistKind::Dataset) }
            _ => Err(self.unexpected("TENSOR or DATASET")),
        }
    }

    // LIST TENSORS
    // LIST DATASETS
    // LIST DATASET VERSIONS <name>
    fn parse_list(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::List)?;
        let target = match self.peek() {
            Some(Token::Tensors)  => { self.advance(); ListTarget::Tensors }
            Some(Token::Datasets) => { self.advance(); ListTarget::Datasets }
            Some(Token::Dataset)  => {
                self.advance();
                if self.at(&Token::Versions) {
                    self.advance();
                    ListTarget::DatasetVersions(self.eat_ident()?)
                } else if self.at_ident("PACKAGES") {
                    self.advance();
                    ListTarget::DatasetPackages
                } else {
                    ListTarget::Datasets
                }
            }
            _ => return Err(self.unexpected("TENSORS, DATASETS, or DATASET VERSIONS")),
        };
        Ok(Statement::List(ListStmt { target }))
    }

    // IMPORT DATASET FROM <path>
    fn parse_import(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Import)?;
        self.eat(&Token::Dataset)?;
        self.eat(&Token::From)?;
        let path = self.eat_str()?;
        Ok(Statement::Import(ImportStmt { ephemeral: false, path, name: None }))
    }

    // EXPORT <name> TO <path>
    fn parse_export(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Export)?;
        let name = self.eat_ident()?;
        self.eat(&Token::To)?;
        let path = self.eat_str()?;
        Ok(Statement::Export(ExportStmt { name, path }))
    }

    // USE <database_name>
    // USE DATASET FROM <path>   (ephemeral import)
    fn parse_use(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Use)?;
        match self.peek() {
            Some(Token::Dataset) => {
                self.advance();
                self.eat(&Token::From)?;
                let path = self.eat_str()?;
                Ok(Statement::Import(ImportStmt { ephemeral: true, path, name: None }))
            }
            Some(Token::Ident(_)) => {
                Ok(Statement::UseDatabase(UseDatabaseStmt { name: self.eat_ident()? }))
            }
            _ => Err(self.unexpected("a database name or DATASET FROM")),
        }
    }

    // CREATE DATABASE <name>
    // CREATE [VECTOR] INDEX <idx_name> ON <dataset>(<column>)
    fn parse_create(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Create)?;
        match self.peek() {
            Some(Token::Database) => {
                self.advance();
                let name = self.eat_ident()?;
                Ok(Statement::CreateDatabase(CreateDatabaseStmt { name }))
            }
            Some(Token::Index) => {
                self.advance();
                let (dataset, column, kind) = self.parse_index_target(IndexKindAst::Default)?;
                Ok(Statement::CreateIndex(CreateIndexStmt { dataset, column, kind }))
            }
            Some(Token::Ident(_)) if self.at_ident("VECTOR") => {
                self.advance(); // VECTOR
                self.eat(&Token::Index)?;
                let (dataset, column, _) = self.parse_index_target(IndexKindAst::Default)?;
                Ok(Statement::CreateIndex(CreateIndexStmt {
                    dataset,
                    column,
                    kind: IndexKindAst::Hash, // vector index maps to hash-style
                }))
            }
            _ => Err(self.unexpected("DATABASE or INDEX after CREATE")),
        }
    }

    // <idx_name> ON <dataset>(<column>)   — called after consuming INDEX keyword
    fn parse_index_target(
        &mut self,
        default_kind: IndexKindAst,
    ) -> Result<(String, String, IndexKindAst), ParseError> {
        // Optional index name (ignored for now, not stored in AST — could be added)
        let _idx_name = if self.at_any_ident() && !self.at_ident("ON") {
            Some(self.eat_ident()?)
        } else {
            None
        };

        // ON keyword
        if self.at_ident("ON") { self.advance(); }

        let dataset = self.eat_ident()?;
        self.eat(&Token::LParen)?;
        let column = self.eat_ident()?;
        self.eat(&Token::RParen)?;

        Ok((dataset, column, default_kind))
    }

    // DROP DATABASE <name>
    fn parse_drop(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Drop)?;
        self.eat(&Token::Database)?;
        let name = self.eat_ident()?;
        Ok(Statement::DropDatabase(DropDatabaseStmt { name }))
    }

    // SET DATASET <name> <key> = <value>
    fn parse_set(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Set)?;
        self.eat(&Token::Dataset)?;
        let dataset = self.eat_ident()?;
        let key = self.eat_ident()?;
        self.eat(&Token::Eq)?;
        let value = self.eat_str()?;
        Ok(Statement::SetMetadata(SetMetadataStmt { dataset, key, value }))
    }

    // SEARCH <query_tensor> [IN <dataset>] [TOP <k>]
    fn parse_search(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Search)?;
        let query_tensor = self.eat_ident()?;

        let dataset = if self.at_ident("IN") {
            self.advance();
            Some(self.eat_ident()?)
        } else {
            None
        };

        let top_k = if self.at_ident("TOP") {
            self.advance();
            Some(self.eat_usize()?)
        } else {
            None
        };

        Ok(Statement::Search(SearchStmt { query_tensor, dataset, top_k }))
    }
}

// ─── Expression parser (Pratt) ────────────────────────────────────────────────

impl Parser {
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_pratt(0)
    }

    /// Pratt parser — `min_bp` is the minimum left binding power to continue.
    fn parse_pratt(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_expr_atom()?;

        loop {
            // Postfix: field access and subscript — always bind tighter than infix
            match self.peek() {
                Some(Token::Dot) => {
                    self.advance();
                    let field = self.eat_ident()?;
                    lhs = Expr::Field { base: Box::new(lhs), field };
                    continue;
                }
                Some(Token::LBracket) => {
                    self.advance();
                    let indices = self.parse_index_specs()?;
                    self.eat(&Token::RBracket)?;
                    lhs = Expr::Index { base: Box::new(lhs), indices };
                    continue;
                }
                _ => {}
            }

            // Infix operators with precedence
            let (left_bp, right_bp, op) = match self.peek() {
                Some(Token::Plus)  => (1u8, 2u8, InfixOp::Add),
                Some(Token::Minus) => (1,   2,   InfixOp::Subtract),
                Some(Token::Star)  => (3,   4,   InfixOp::Multiply),
                Some(Token::Slash) => (3,   4,   InfixOp::Divide),
                _                  => break,
            };

            if left_bp < min_bp { break; }

            self.advance();
            let rhs = self.parse_pratt(right_bp)?;
            lhs = Expr::Infix { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }

        Ok(lhs)
    }

    fn parse_expr_atom(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            Some(Token::Float(_)) => {
                if let Some(Token::Float(f)) = self.advance() { return Ok(Expr::Scalar(f)); }
                unreachable!()
            }
            Some(Token::Int(_)) => {
                if let Some(Token::Int(n)) = self.advance() { return Ok(Expr::Scalar(n as f64)); }
                unreachable!()
            }
            Some(Token::Str(_)) => {
                return Ok(Expr::StringLit(self.eat_str()?));
            }
            // Unary minus: -<atom>
            Some(Token::Minus) => {
                self.advance();
                let inner = self.parse_pratt(5)?; // high bp so - binds tightly
                return Ok(match inner {
                    Expr::Scalar(n) => Expr::Scalar(-n),
                    other => Expr::Call(CallExpr::Scale {
                        input: Box::new(other),
                        factor: -1.0,
                    }),
                });
            }
            // Parenthesized expression
            Some(Token::LParen) => {
                self.advance();
                let e = self.parse_pratt(0)?;
                self.eat(&Token::RParen)?;
                return Ok(e);
            }
            // Named prefix operations
            Some(Token::Add)
            | Some(Token::Subtract)
            | Some(Token::Multiply)
            | Some(Token::Divide)
            | Some(Token::Correlate)
            | Some(Token::Similarity)
            | Some(Token::Distance)
            | Some(Token::Matmul)
            | Some(Token::Normalize)
            | Some(Token::Transpose)
            | Some(Token::Flatten)
            | Some(Token::Sum)
            | Some(Token::Mean)
            | Some(Token::Stdev)
            | Some(Token::Scale)
            | Some(Token::Reshape)
            | Some(Token::Stack) => return self.parse_call_expr(),
            // Identifier (variable ref or dataset constructor)
            Some(Token::Ident(_)) => {
                let name = self.eat_ident()?;
                // dataset("name") constructor
                if name == "dataset" && self.at(&Token::LParen) {
                    self.advance();
                    let ds = self.eat_str()?;
                    self.eat(&Token::RParen)?;
                    return Ok(Expr::DatasetRef(ds));
                }
                return Ok(Expr::Ref(name));
            }
            _ => {}
        }
        Err(self.unexpected("an expression"))
    }

    fn parse_call_expr(&mut self) -> Result<Expr, ParseError> {
        let call = match self.advance() {
            // Two-operand (juxtaposition: OP a b)
            Some(Token::Add) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Add(Box::new(a), Box::new(b))
            }
            Some(Token::Subtract) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Subtract(Box::new(a), Box::new(b))
            }
            Some(Token::Multiply) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Multiply(Box::new(a), Box::new(b))
            }
            Some(Token::Divide) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Divide(Box::new(a), Box::new(b))
            }
            // Two-operand with keyword separator
            Some(Token::Correlate) => {
                let a = self.parse_simple_expr()?;
                self.eat(&Token::With)?;
                let b = self.parse_simple_expr()?;
                CallExpr::Correlate(Box::new(a), Box::new(b))
            }
            Some(Token::Similarity) => {
                let a = self.parse_simple_expr()?;
                self.eat(&Token::With)?;
                let b = self.parse_simple_expr()?;
                CallExpr::Similarity(Box::new(a), Box::new(b))
            }
            Some(Token::Distance) => {
                let a = self.parse_simple_expr()?;
                self.eat(&Token::To)?;
                let b = self.parse_simple_expr()?;
                CallExpr::Distance(Box::new(a), Box::new(b))
            }
            Some(Token::Matmul) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Matmul(Box::new(a), Box::new(b))
            }
            // Single-operand
            Some(Token::Normalize) => CallExpr::Normalize(Box::new(self.parse_simple_expr()?)),
            Some(Token::Transpose)  => CallExpr::Transpose(Box::new(self.parse_simple_expr()?)),
            Some(Token::Flatten)    => CallExpr::Flatten(Box::new(self.parse_simple_expr()?)),
            Some(Token::Sum)        => CallExpr::Sum(Box::new(self.parse_simple_expr()?)),
            Some(Token::Mean)       => CallExpr::Mean(Box::new(self.parse_simple_expr()?)),
            Some(Token::Stdev)      => CallExpr::Stdev(Box::new(self.parse_simple_expr()?)),
            // Single-operand with trailing parameter
            Some(Token::Scale) => {
                let input = self.parse_simple_expr()?;
                self.eat(&Token::By)?;
                let factor = self.eat_number()?;
                CallExpr::Scale { input: Box::new(input), factor }
            }
            Some(Token::Reshape) => {
                let input = self.parse_simple_expr()?;
                self.eat(&Token::To)?;
                let shape = self.parse_usize_list()?;
                CallExpr::Reshape { input: Box::new(input), shape }
            }
            // N-ary: STACK t1 t2 t3 ...
            Some(Token::Stack) => {
                let mut operands = vec![self.parse_simple_expr()?];
                while self.can_start_simple_expr() {
                    operands.push(self.parse_simple_expr()?);
                }
                if operands.len() < 2 {
                    return Err(self.error("STACK requires at least 2 operands"));
                }
                CallExpr::Stack(operands)
            }
            _ => return Err(self.unexpected("a named operation")),
        };
        Ok(Expr::Call(call))
    }

    /// A "simple" expression: identifier, literal, subscript, field access, or
    /// a parenthesized full expression. Used as operands to named operations to
    /// avoid ambiguity with multi-token argument lists.
    fn parse_simple_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = match self.peek() {
            Some(Token::Float(_)) => {
                if let Some(Token::Float(f)) = self.advance() { Expr::Scalar(f) } else { unreachable!() }
            }
            Some(Token::Int(_)) => {
                if let Some(Token::Int(n)) = self.advance() { Expr::Scalar(n as f64) } else { unreachable!() }
            }
            Some(Token::Minus) => {
                self.advance();
                Expr::Scalar(-self.eat_number()?)
            }
            Some(Token::Ident(_)) => Expr::Ref(self.eat_ident()?),
            Some(Token::LParen) => {
                self.advance();
                let e = self.parse_pratt(0)?;
                self.eat(&Token::RParen)?;
                e
            }
            _ => return Err(self.unexpected(
                "a simple expression (identifier, literal, or parenthesized expression)"
            )),
        };

        // Allow one level of postfix per simple expr
        if self.at(&Token::LBracket) {
            self.advance();
            let indices = self.parse_index_specs()?;
            self.eat(&Token::RBracket)?;
            expr = Expr::Index { base: Box::new(expr), indices };
        } else if self.at(&Token::Dot) {
            self.advance();
            let field = self.eat_ident()?;
            expr = Expr::Field { base: Box::new(expr), field };
        }

        Ok(expr)
    }

    /// Returns `true` if the current token could start a `parse_simple_expr`.
    fn can_start_simple_expr(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Ident(_))
                | Some(Token::Int(_))
                | Some(Token::Float(_))
                | Some(Token::Minus)
                | Some(Token::LParen)
        )
    }
}

// ─── Helper parsers ───────────────────────────────────────────────────────────

impl Parser {
    /// `[n1, n2, n3]` → `Vec<usize>`
    fn parse_usize_list(&mut self) -> Result<Vec<usize>, ParseError> {
        self.eat(&Token::LBracket)?;
        let mut dims = Vec::new();
        while !self.at(&Token::RBracket) && !self.eof() {
            dims.push(self.eat_usize()?);
            if self.at(&Token::Comma) { self.advance(); }
        }
        self.eat(&Token::RBracket)?;
        Ok(dims)
    }

    /// `[v1, v2, v3]` → `Vec<f64>`
    fn parse_f64_list(&mut self) -> Result<Vec<f64>, ParseError> {
        self.eat(&Token::LBracket)?;
        let mut vals = Vec::new();
        while !self.at(&Token::RBracket) && !self.eof() {
            vals.push(self.eat_number()?);
            if self.at(&Token::Comma) { self.advance(); }
        }
        self.eat(&Token::RBracket)?;
        Ok(vals)
    }

    /// Comma-separated index specifiers inside `[...]`:
    /// `0`, `0:5`, `*`, `:`
    fn parse_index_specs(&mut self) -> Result<Vec<IndexSpec>, ParseError> {
        let mut specs = Vec::new();
        while !self.at(&Token::RBracket) && !self.eof() {
            let spec = if self.at(&Token::Star) || self.at(&Token::Colon) {
                self.advance();
                IndexSpec::All
            } else {
                let start = self.eat_usize()?;
                if self.at(&Token::Colon) {
                    self.advance();
                    let end = self.eat_usize()?;
                    IndexSpec::Range(start, end)
                } else {
                    IndexSpec::Index(start)
                }
            };
            specs.push(spec);
            if self.at(&Token::Comma) { self.advance(); }
        }
        Ok(specs)
    }

    /// `(colname: coltype [NOT NULLABLE], ...)` for DATASET COLUMNS
    fn parse_column_list(&mut self) -> Result<Vec<ColumnDef>, ParseError> {
        self.eat(&Token::LParen)?;
        let mut cols = Vec::new();
        while !self.at(&Token::RParen) && !self.eof() {
            cols.push(self.parse_column_def()?);
            if self.at(&Token::Comma) { self.advance(); }
        }
        self.eat(&Token::RParen)?;
        Ok(cols)
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef, ParseError> {
        let name = self.eat_ident()?;
        self.eat(&Token::Colon)?;
        let col_type = self.parse_col_type()?;

        let nullable = if self.at(&Token::Not) {
            self.advance();
            self.eat(&Token::Nullable)?;
            false
        } else if self.at(&Token::Nullable) {
            self.advance();
            true
        } else {
            true // default
        };

        Ok(ColumnDef { name, col_type, nullable })
    }

    fn parse_col_type(&mut self) -> Result<ColType, ParseError> {
        match self.peek() {
            Some(Token::Vector) => {
                self.advance();
                self.eat(&Token::LParen)?;
                let n = self.eat_usize()?;
                self.eat(&Token::RParen)?;
                Ok(ColType::Vector(n))
            }
            Some(Token::Matrix) => {
                self.advance();
                self.eat(&Token::LParen)?;
                let rows = self.eat_usize()?;
                self.eat(&Token::Comma)?;
                let cols = self.eat_usize()?;
                self.eat(&Token::RParen)?;
                Ok(ColType::Matrix(rows, cols))
            }
            Some(Token::Tensor) => {
                self.advance();
                let dims = self.parse_usize_list()?;
                Ok(ColType::Tensor(dims))
            }
            Some(Token::Ident(_)) => {
                let t = self.eat_ident()?;
                match t.to_uppercase().as_str() {
                    "INT" | "INTEGER" | "INT32" | "INT64" => Ok(ColType::Int),
                    "FLOAT" | "FLOAT32" | "FLOAT64" | "DOUBLE" => Ok(ColType::Float),
                    "STRING" | "TEXT" | "VARCHAR" => Ok(ColType::String),
                    "BOOL" | "BOOLEAN" => Ok(ColType::Bool),
                    // Mixed-case aliases (e.g. Vector(128), Matrix(4,4) in schema defs)
                    "VECTOR" => {
                        self.eat(&Token::LParen)?;
                        let n = self.eat_usize()?;
                        self.eat(&Token::RParen)?;
                        Ok(ColType::Vector(n))
                    }
                    "MATRIX" => {
                        self.eat(&Token::LParen)?;
                        let rows = self.eat_usize()?;
                        self.eat(&Token::Comma)?;
                        let cols = self.eat_usize()?;
                        self.eat(&Token::RParen)?;
                        Ok(ColType::Matrix(rows, cols))
                    }
                    "TENSOR" => Ok(ColType::Tensor(self.parse_usize_list()?)),
                    other => Err(self.error(format!("unknown column type '{}'", other))),
                }
            }
            _ => Err(self.unexpected(
                "a column type: Int, Float, String, Bool, Vector(n), Matrix(r,c), Tensor[...]",
            )),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> Statement {
        parse(src).unwrap_or_else(|e| panic!("parse failed for {:?}: {}", src, e))
    }

    fn parse_err(src: &str) -> ParseError {
        parse(src).expect_err(&format!("expected parse error for {:?}", src))
    }

    // ── Tensor construction ──────────────────────────────────────────────────

    #[test]
    fn vector_literal() {
        let stmt = parse_ok("VECTOR v = [1, 2, 3]");
        let Statement::Vector(s) = stmt else { panic!("expected Vector") };
        assert_eq!(s.name, "v");
        assert_eq!(s.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn matrix_literal() {
        let stmt = parse_ok("MATRIX m = [[1, 2], [3, 4]]");
        let Statement::Matrix(s) = stmt else { panic!("expected Matrix") };
        assert_eq!(s.name, "m");
        assert_eq!(s.rows, 2);
        assert_eq!(s.cols, 2);
        assert_eq!(s.values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn define_tensor_normal() {
        let stmt = parse_ok("DEFINE v AS TENSOR [3] VALUES [1, 0, 0]");
        let Statement::DefineTensor(s) = stmt else { panic!("expected DefineTensor") };
        assert_eq!(s.name, "v");
        assert_eq!(s.kind, TensorKindAst::Normal);
        assert_eq!(s.shape, vec![3]);
        assert_eq!(s.values, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn define_tensor_strict() {
        let stmt = parse_ok("DEFINE v AS STRICT TENSOR [3] VALUES [1, 0, 0]");
        let Statement::DefineTensor(s) = stmt else { panic!("expected DefineTensor") };
        assert_eq!(s.kind, TensorKindAst::Strict);
    }

    // ── LET / assignment ─────────────────────────────────────────────────────

    #[test]
    fn let_named_op() {
        let stmt = parse_ok("LET result = ADD a b");
        let Statement::Let(s) = stmt else { panic!("expected Let") };
        assert_eq!(s.name, "result");
        assert!(!s.lazy);
        assert!(matches!(s.expr, Expr::Call(CallExpr::Add(..))));
    }

    #[test]
    fn let_lazy_prefix() {
        let stmt = parse_ok("LAZY LET trend = STDEV sensor");
        let Statement::Let(s) = stmt else { panic!("expected Let") };
        assert!(s.lazy);
        assert!(matches!(s.expr, Expr::Call(CallExpr::Stdev(..))));
    }

    #[test]
    fn let_lazy_suffix() {
        let stmt = parse_ok("LET LAZY trend = MEAN sensor");
        let Statement::Let(s) = stmt else { panic!("expected Let") };
        assert!(s.lazy);
    }

    #[test]
    fn let_infix_add() {
        let stmt = parse_ok("LET c = a + b");
        let Statement::Let(s) = stmt else { panic!("expected Let") };
        assert!(matches!(s.expr, Expr::Infix { op: InfixOp::Add, .. }));
    }

    #[test]
    fn let_infix_precedence() {
        // a + b * 2.0  should parse as  a + (b * 2.0)
        let stmt = parse_ok("LET c = a + b * 2.0");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Infix { op: InfixOp::Add, lhs, rhs } = s.expr else { panic!("expected Add at root") };
        assert!(matches!(*lhs, Expr::Ref(_)));
        assert!(matches!(*rhs, Expr::Infix { op: InfixOp::Multiply, .. }));
    }

    #[test]
    fn let_subscript() {
        let stmt = parse_ok("LET x = t[0, 1]");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Index { base, indices } = s.expr else { panic!("expected Index") };
        assert!(matches!(*base, Expr::Ref(ref n) if n == "t"));
        assert_eq!(indices.len(), 2);
        assert!(matches!(indices[0], IndexSpec::Index(0)));
        assert!(matches!(indices[1], IndexSpec::Index(1)));
    }

    #[test]
    fn let_range_subscript() {
        let stmt = parse_ok("LET x = t[0:5, *]");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Index { indices, .. } = s.expr else { panic!() };
        assert!(matches!(indices[0], IndexSpec::Range(0, 5)));
        assert!(matches!(indices[1], IndexSpec::All));
    }

    #[test]
    fn let_field_access() {
        let stmt = parse_ok("LET x = ds.col");
        let Statement::Let(s) = stmt else { panic!() };
        assert!(matches!(s.expr, Expr::Field { .. }));
    }

    #[test]
    fn let_correlate() {
        let stmt = parse_ok("LET sim = CORRELATE a WITH b");
        let Statement::Let(s) = stmt else { panic!() };
        assert!(matches!(s.expr, Expr::Call(CallExpr::Correlate(..))));
    }

    #[test]
    fn let_scale() {
        let stmt = parse_ok("LET s = SCALE a BY 0.5");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Call(CallExpr::Scale { factor, .. }) = s.expr else { panic!() };
        assert!((factor - 0.5).abs() < 1e-9);
    }

    #[test]
    fn let_reshape() {
        let stmt = parse_ok("LET r = RESHAPE a TO [2, 3]");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Call(CallExpr::Reshape { shape, .. }) = s.expr else { panic!() };
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn let_stack() {
        let stmt = parse_ok("LET s = STACK a b c");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Call(CallExpr::Stack(ops)) = s.expr else { panic!() };
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn let_dataset_constructor() {
        let stmt = parse_ok(r#"LET ds = dataset("my_ds")"#);
        let Statement::Let(s) = stmt else { panic!() };
        assert!(matches!(s.expr, Expr::DatasetRef(ref n) if n == "my_ds"));
    }

    // ── Semantics ────────────────────────────────────────────────────────────

    #[test]
    fn bind_statement() {
        let stmt = parse_ok("BIND alias TO source");
        let Statement::Bind(s) = stmt else { panic!() };
        assert_eq!(s.alias, "alias");
        assert_eq!(s.target, "source");
    }

    #[test]
    fn attach_statement() {
        let stmt = parse_ok("ATTACH my_tensor TO my_dataset.my_col");
        let Statement::Attach(s) = stmt else { panic!() };
        assert_eq!(s.tensor, "my_tensor");
        assert_eq!(s.dataset, "my_dataset");
        assert_eq!(s.column, "my_col");
    }

    #[test]
    fn derive_statement() {
        let stmt = parse_ok("DERIVE clean FROM NORMALIZE raw");
        let Statement::Derive(s) = stmt else { panic!() };
        assert_eq!(s.name, "clean");
        assert!(matches!(s.source_expr, Expr::Call(CallExpr::Normalize(..))));
    }

    // ── SHOW ─────────────────────────────────────────────────────────────────

    #[test]
    fn show_schema() {
        let stmt = parse_ok("SHOW SCHEMA my_dataset");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::Schema(ref n) if n == "my_dataset"));
    }

    #[test]
    fn show_all() {
        let stmt = parse_ok("SHOW ALL");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::All));
    }

    #[test]
    fn show_all_datasets() {
        let stmt = parse_ok("SHOW ALL DATASETS");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::AllDatasets));
    }

    #[test]
    fn show_named() {
        let stmt = parse_ok("SHOW my_tensor");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::Named(ref n) if n == "my_tensor"));
    }

    #[test]
    fn show_dataset_metadata() {
        let stmt = parse_ok("SHOW DATASET METADATA foo");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::DatasetMetadata(ref n) if n == "foo"));
    }

    // ── Dataset ops ──────────────────────────────────────────────────────────

    #[test]
    fn create_dataset_columns() {
        let stmt = parse_ok("DATASET diagnostics COLUMNS (id: Int, emb: Vector(128))");
        let Statement::CreateDataset(s) = stmt else { panic!() };
        assert_eq!(s.name, "diagnostics");
        assert_eq!(s.columns.len(), 2);
        assert_eq!(s.columns[0].name, "id");
        assert!(matches!(s.columns[0].col_type, ColType::Int));
        assert!(matches!(s.columns[1].col_type, ColType::Vector(128)));
    }

    #[test]
    fn create_dataset_matrix_column() {
        let stmt = parse_ok("DATASET t COLUMNS (feat: Matrix(4, 4))");
        let Statement::CreateDataset(s) = stmt else { panic!() };
        assert!(matches!(s.columns[0].col_type, ColType::Matrix(4, 4)));
    }

    #[test]
    fn select_basic() {
        let stmt = parse_ok("SELECT col1, col2 FROM my_ds");
        let Statement::Select(s) = stmt else { panic!() };
        assert_eq!(s.dataset, "my_ds");
        let SelectColumns::Named(cols) = s.columns else { panic!() };
        assert_eq!(cols, vec!["col1", "col2"]);
    }

    #[test]
    fn select_with_limit() {
        let stmt = parse_ok("SELECT * FROM my_ds LIMIT 10");
        let Statement::Select(s) = stmt else { panic!() };
        assert!(matches!(s.columns, SelectColumns::All));
        assert_eq!(s.limit, Some(10));
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    #[test]
    fn save_tensor() {
        let stmt = parse_ok(r#"SAVE TENSOR my_t TO "data/my_t.json""#);
        let Statement::Save(s) = stmt else { panic!() };
        assert!(matches!(s.kind, PersistKind::Tensor));
        assert_eq!(s.name, "my_t");
        assert_eq!(s.path, Some("data/my_t.json".into()));
    }

    #[test]
    fn load_dataset() {
        let stmt = parse_ok(r#"LOAD DATASET ds FROM "data/ds""#);
        let Statement::Load(s) = stmt else { panic!() };
        assert!(matches!(s.kind, PersistKind::Dataset));
        assert_eq!(s.path, Some("data/ds".into()));
    }

    #[test]
    fn use_database() {
        let stmt = parse_ok("USE analytics");
        let Statement::UseDatabase(s) = stmt else { panic!() };
        assert_eq!(s.name, "analytics");
    }

    #[test]
    fn use_dataset_from() {
        let stmt = parse_ok(r#"USE DATASET FROM "data/foo.h5ad""#);
        let Statement::Import(s) = stmt else { panic!() };
        assert!(s.ephemeral);
        assert_eq!(s.path, "data/foo.h5ad");
    }

    #[test]
    fn import_dataset_from() {
        let stmt = parse_ok(r#"IMPORT DATASET FROM "data/foo.parquet""#);
        let Statement::Import(s) = stmt else { panic!() };
        assert!(!s.ephemeral);
    }

    // ── Database management ──────────────────────────────────────────────────

    #[test]
    fn create_database() {
        let stmt = parse_ok("CREATE DATABASE mydb");
        let Statement::CreateDatabase(s) = stmt else { panic!() };
        assert_eq!(s.name, "mydb");
    }

    #[test]
    fn drop_database() {
        let stmt = parse_ok("DROP DATABASE mydb");
        let Statement::DropDatabase(s) = stmt else { panic!() };
        assert_eq!(s.name, "mydb");
    }

    // ── Misc ─────────────────────────────────────────────────────────────────

    #[test]
    fn reset_statement() {
        let stmt = parse_ok("RESET");
        assert!(matches!(stmt, Statement::Reset));
    }

    #[test]
    fn is_read_only() {
        assert!(parse_ok("SHOW ALL").is_read_only());
        assert!(parse_ok("EXPLAIN foo").is_read_only());
        assert!(parse_ok("LIST TENSORS").is_read_only());
        assert!(!parse_ok("LET x = ADD a b").is_read_only());
        assert!(!parse_ok("VECTOR v = [1, 2]").is_read_only());
    }

    // ── Error cases ──────────────────────────────────────────────────────────

    #[test]
    fn unknown_command_errors() {
        let e = parse_err("UNKNOWN foo bar");
        assert!(e.msg.contains("expected a statement keyword"));
    }

    #[test]
    fn empty_input_errors() {
        let e = parse_err("");
        assert!(e.msg.contains("empty input"));
    }

    #[test]
    fn missing_eq_in_let_errors() {
        let e = parse_err("LET x ADD a b");
        assert!(e.msg.contains('=') || e.msg.contains("expected"));
    }

    #[test]
    fn trailing_tokens_error() {
        let e = parse_err("RESET RESET");
        assert!(e.msg.contains("unexpected token"));
    }
}
